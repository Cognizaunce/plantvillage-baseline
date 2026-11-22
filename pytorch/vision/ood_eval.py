"""Out-of-distribution (OOD) evaluation: can the model tell "not a tomato leaf I know"?

For each model and each dataset, scores every in-distribution (ID) tomato image and every
OOD image (other crops), then reports how well each score separates the two groups:

    MSP     max softmax probability            (Hendrycks & Gimpel, 2017)
    Energy  logsumexp of the logits, T = 1     (Liu et al., 2020)
    Higher score = "looks like a tomato leaf". Inputs below a threshold get rejected.

Metrics per score:
    AUROC          1.0 = perfect separation, 0.5 = guessing
    FPR@95TPR      share of OOD images still accepted when 95% of tomato test images are kept
    Rejection rule threshold fixed on the *validation* tomato images (keep 95%), then applied
    to test: reports how many test tomato leaves are kept and how many OOD leaves are rejected.

Read-only: loads checkpoints, writes only ood_metrics_<dataset>.json in each model_dir.

Run from pytorch/vision/:
    python ood_eval.py
    python ood_eval.py --model_dirs experiments/joint_mobilenetv3 --datasets fieldplant
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import utils
from model.data_loader import make_transform
from model.net import Net

IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}
DEFAULT_MODELS = ['experiments/plantvillage_mobilenetv3',
                  'experiments/fieldplant_finetune_mobilenetv3',
                  'experiments/joint_mobilenetv3']
DATASETS = {'plantvillage': Path('data/tomato_plantvillage'),
            'fieldplant': Path('data/tomato_fieldplant')}


class ImageFolderFlat(Dataset):
    """All images under the given folders (recursive); labels are not needed for OOD scoring."""

    def __init__(self, folders, transform):
        self.files = sorted(p for d in folders for p in Path(d).rglob('*')
                            if p.suffix.lower() in IMAGE_EXTS)
        self.groups = [p.parent.name for p in self.files]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.files[i]) as image:
            return self.transform(image.convert('RGB')), 0


def load_model(model_dir, restore, device):
    state = torch.load(Path(model_dir) / f'{restore}.pth.tar', map_location='cpu',
                       weights_only=True)
    params = utils.Params(Path(model_dir) / 'params.json')
    params.__dict__.update(state['params'])
    params.pretrained = False  # weights come from the checkpoint; skip the ImageNet download
    model = Net(params)
    model.load_state_dict(state['state_dict'])
    return model.to(device).eval(), params


@torch.no_grad()
def scores(model, folders, params, device, batch_size):
    dataset = ImageFolderFlat(folders, make_transform(params, training=False))
    if len(dataset) == 0:
        return None, None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=params.num_workers)
    msp, energy = [], []
    for images, _ in loader:
        logits = model(images.to(device)).float()
        msp.append(torch.softmax(logits, 1).max(1).values.cpu())
        energy.append(torch.logsumexp(logits, 1).cpu())
    return {'msp': torch.cat(msp).numpy(), 'energy': torch.cat(energy).numpy()}, dataset.groups


def auroc(id_scores, ood_scores):
    """Probability a random ID image scores higher than a random OOD image (ties count half)."""
    values = np.concatenate([id_scores, ood_scores])
    order = values.argsort(kind='mergesort')
    ranks = np.empty(len(values))
    sorted_vals = values[order]
    i = 0
    while i < len(values):  # average ranks over ties
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_id, n_ood = len(id_scores), len(ood_scores)
    return float((ranks[:n_id].sum() - n_id * (n_id + 1) / 2) / (n_id * n_ood))


def fpr_at_tpr(id_scores, ood_scores, tpr=0.95):
    threshold = np.quantile(id_scores, 1 - tpr)
    return float(np.mean(ood_scores >= threshold))


def evaluate(model, params, data_dir, device, batch_size, keep_rate):
    ood_root = data_dir / 'ood'
    ood_folders = sorted(d for d in ood_root.iterdir() if d.is_dir()) if ood_root.exists() else []
    id_val, _ = scores(model, [data_dir / 'val_signs'], params, device, batch_size)
    id_test, _ = scores(model, [data_dir / 'test_signs'], params, device, batch_size)
    ood, groups = scores(model, ood_folders, params, device, batch_size)
    if id_test is None or ood is None:
        raise SystemExit(f'No tomato test images or no OOD images under {data_dir}')
    crops = sorted(set(groups))
    groups = np.array(groups)
    result = {'id_test_count': len(id_test['msp']), 'ood_count': len(ood['msp']),
              'ood_per_crop_count': {g: int((groups == g).sum()) for g in crops}}
    for name in ('msp', 'energy'):
        threshold = float(np.quantile(id_val[name], 1 - keep_rate))
        result[name] = {
            'auroc': auroc(id_test[name], ood[name]),
            'fpr_at_95tpr': fpr_at_tpr(id_test[name], ood[name]),
            'threshold_from_val': threshold,
            'test_tomato_kept': float(np.mean(id_test[name] >= threshold)),
            'ood_rejected': float(np.mean(ood[name] < threshold)),
            'auroc_per_crop': {g: auroc(id_test[name], ood[name][groups == g]) for g in crops},
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model_dirs', nargs='+', default=DEFAULT_MODELS)
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument('--restore_file', default='best', choices=['best', 'last'])
    parser.add_argument('--keep_rate', type=float, default=0.95,
                        help='Share of validation tomato images the threshold must keep.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    device = str(utils.get_device(args.device))

    summary = []
    for model_dir in args.model_dirs:
        if not (Path(model_dir) / f'{args.restore_file}.pth.tar').exists():
            print(f'Skipping {model_dir}: no {args.restore_file}.pth.tar')
            continue
        model, params = load_model(model_dir, args.restore_file, device)
        for name in args.datasets:
            print(f'Scoring {model_dir} on {name} ...', flush=True)
            result = evaluate(model, params, DATASETS[name], device, args.batch_size,
                              args.keep_rate)
            result.update(model_dir=model_dir, dataset=name, keep_rate=args.keep_rate)
            utils.save_dict_to_json(result, Path(model_dir) / f'ood_metrics_{name}.json')
            summary.append(result)

    print(f'\n{"model":32s} {"data":12s} {"score":7s} {"AUROC":>6s} {"FPR@95":>7s}'
          f' {"kept":>6s} {"OOD rej":>8s}')
    for r in summary:
        for s in ('msp', 'energy'):
            m = r[s]
            print(f'{Path(r["model_dir"]).name:32s} {r["dataset"]:12s} {s:7s}'
                  f' {m["auroc"]:6.3f} {m["fpr_at_95tpr"]:7.3f}'
                  f' {m["test_tomato_kept"]:6.3f} {m["ood_rejected"]:8.3f}')
    for r in summary:
        crops = ', '.join(f'{c}={v:.3f}' for c, v in r['energy']['auroc_per_crop'].items())
        print(f'  energy AUROC per crop, {Path(r["model_dir"]).name} / {r["dataset"]}: {crops}')


if __name__ == '__main__':
    main()
