"""Merge PlantVillage and FieldPlant splits into a single joint dataset.

Creates data/tomato_joint/{train,val,test}_signs/ by symlinking (Linux/Mac)
or copying (Windows / --copy flag) files from both source datasets.
FieldPlant files are prefixed with 'fp_' to avoid filename collisions.

Then train with:
    python train.py --data_dir data/tomato_joint \\
        --model_dir experiments/joint_mobilenetv3 \\
        --init_from experiments/plantvillage_mobilenetv3/best.pth.tar

Run from pytorch/vision/:
    python build_joint_dataset.py
"""
import argparse
import json
import os
import shutil
from pathlib import Path

PV_DIR = Path('data/tomato_plantvillage')
FP_DIR = Path('data/tomato_fieldplant')
SPLITS = ('train', 'val', 'test')


def link_or_copy(src, dst, use_copy):
    if use_copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def build_split(split, out_dir, use_copy, pv_dir, fp_dir):
    out_split = out_dir / f'{split}_signs'
    out_split.mkdir(parents=True, exist_ok=True)

    pv_src = pv_dir / f'{split}_signs'
    fp_src = fp_dir / f'{split}_signs'

    counts = {}

    for f in sorted(pv_src.iterdir()):
        if f.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        dst = out_split / f.name
        if not dst.exists():
            link_or_copy(f, dst, use_copy)
        label = int(f.name.split('_', 1)[0])
        counts[label] = counts.get(label, 0) + 1

    for f in sorted(fp_src.iterdir()):
        if f.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        dst = out_split / f'fp_{f.name}'
        if not dst.exists():
            link_or_copy(f, dst, use_copy)
        label = int(f.name.split('_', 1)[0])
        counts[label] = counts.get(label, 0) + 1

    total = sum(counts.values())
    dist = ', '.join(f'{k}:{v}' for k, v in sorted(counts.items()))
    print(f'  {split:5s}  {total:5d}  {dist}')
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pv_dir',   default=str(PV_DIR))
    parser.add_argument('--fp_dir',   default=str(FP_DIR))
    parser.add_argument('--out_dir',  default='data/tomato_joint')
    parser.add_argument('--copy', action='store_true',
                        help='Copy files instead of symlinking (required on Windows).')
    parser.add_argument('--overwrite', action='store_true',
                        help='Delete and recreate the output directory.')
    args = parser.parse_args()

    pv_dir = Path(args.pv_dir)
    fp_dir = Path(args.fp_dir)
    out_dir = Path(args.out_dir)

    for d, name in [(pv_dir, '--pv_dir'), (fp_dir, '--fp_dir')]:
        if not d.exists():
            raise SystemExit(f'{name} not found: {d}')

    use_copy = args.copy or os.name == 'nt'  # always copy on Windows
    if use_copy:
        print('Windows detected — copying files (not symlinking).')

    if out_dir.exists():
        if args.overwrite:
            shutil.rmtree(out_dir)
            print(f'Removed existing {out_dir}')
        else:
            raise SystemExit(
                f'{out_dir} already exists. Use --overwrite to recreate it.')

    print(f'Building joint dataset -> {out_dir}')
    for split in SPLITS:
        build_split(split, out_dir, use_copy, pv_dir, fp_dir)

    # Copy classes.json from PlantVillage (same 10 classes)
    classes_src = pv_dir / 'classes.json'
    if classes_src.exists():
        shutil.copy2(classes_src, out_dir / 'classes.json')

    print(f'\nDone. Train with:')
    print(f'  python train.py --data_dir {out_dir} \\')
    print(f'      --model_dir experiments/joint_mobilenetv3 \\')
    print(f'      --init_from experiments/plantvillage_mobilenetv3/best.pth.tar')


if __name__ == '__main__':
    main()
