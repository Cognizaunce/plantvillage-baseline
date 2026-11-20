"""Evaluate a selected checkpoint. Use validation for selection; reserve test for final evaluation."""
import argparse
import json
from pathlib import Path
import torch
import utils
from model.net import Net, loss_fn
from model.data_loader import fetch_dataloader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', default='data/64x64_SIGNS')
    parser.add_argument('--model_dir', default='experiments/base_model')
    parser.add_argument('--restore_file', default='best', choices=['best', 'last'])
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--tag', default='', help='Added to the metrics file name, e.g. fieldplant')
    args = parser.parse_args()
    checkpoint = Path(args.model_dir) / (args.restore_file + '.pth.tar')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    params = utils.Params(Path(args.model_dir) / 'params.json')
    params.__dict__.update(state['params'])  # Architecture must match the trained checkpoint.
    params.device = str(utils.get_device(args.device))
    utils.seed_everything(params.seed)
    loaders = fetch_dataloader([args.split], args.data_dir, params)
    batches = loaders[args.split]
    model = Net(params).to(params.device)
    model.load_state_dict(state['state_dict'])
    metrics = utils.run_epoch(model, loss_fn, batches, params.device, per_class=True)
    metrics.update(epoch=state['epoch'], split=args.split)
    destination = Path(args.model_dir) / (f'metrics_{args.split}_{args.restore_file}'
                                          + (f'_{args.tag}' if args.tag else '') + '.json')
    utils.save_dict_to_json(metrics, destination)
    print({k: metrics[k] for k in ['loss', 'accuracy', 'macro_f1', 'count', 'epoch', 'split']})
    classes_file = Path(args.data_dir) / 'classes.json'
    names = (json.loads(classes_file.read_text(encoding="utf-8"))["classes"]
             if classes_file.exists() else {})
    for i, (f1, recall) in enumerate(zip(metrics['per_class_f1'], metrics['per_class_recall'])):
        print(f'  {i} {str(names.get(str(i), "")):45s} F1={f1:.3f} recall={recall:.3f}')


if __name__ == '__main__':
    main()
