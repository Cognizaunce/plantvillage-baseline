"""Download the raw PlantVillage and FieldPlant datasets into data/raw.

PlantVillage: sparse git clone of github.com/spMohanty/PlantVillage-Dataset (raw/color only).
FieldPlant:   Roboflow COCO export of plant-disease-detection/fieldplant (CC BY 4.0).
              Needs a free Roboflow API key in the ROBOFLOW_API_KEY environment variable
              and `pip install roboflow`. Without a key, the script prints manual steps.

Existing downloads are left alone. Then run build_tomato_dataset.py.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

PLANTVILLAGE_URL = 'https://github.com/spMohanty/PlantVillage-Dataset.git'
FIELDPLANT_WORKSPACE, FIELDPLANT_PROJECT = 'plant-disease-detection', 'fieldplant'
FIELDPLANT_PAGE = 'https://universe.roboflow.com/plant-disease-detection/fieldplant'


def run(command):
    print('$', ' '.join(command))
    subprocess.run(command, check=True)


def download_plantvillage(raw_dir, tomato_only):
    target = raw_dir/'PlantVillage-Dataset'
    if (target/'raw'/'color').exists():
        print(f'PlantVillage already present at {target}; skipping.')
        return
    if shutil.which('git') is None:
        sys.exit('git is required to download PlantVillage. Install git, or download the '
                 'raw/color folder manually into ' + str(target))
    if target.exists():
        sys.exit(f'{target} exists but is incomplete; delete it and rerun.')
    run(['git', 'clone', '--depth', '1', '--filter=blob:none', '--sparse',
         PLANTVILLAGE_URL, str(target)])
    if tomato_only:
        # Tomato classes only (about 18k images). No PlantVillage OOD crops in this case.
        run(['git', '-C', str(target), 'sparse-checkout', 'set', '--no-cone',
             '/raw/color/Tomato___*/'])
    else:
        # All 38 colour classes (about 54k images); non-tomato crops are used for OOD.
        run(['git', '-C', str(target), 'sparse-checkout', 'set', 'raw/color'])
    print(f'PlantVillage colour images are in {target/"raw"/"color"}')


def download_fieldplant(raw_dir, version):
    target = raw_dir/'FieldPlant'
    if target.exists() and any(target.rglob('*.json')):
        print(f'FieldPlant already present at {target}; skipping.')
        return
    key = os.environ.get('ROBOFLOW_API_KEY')
    manual = (f'Manual option: open {FIELDPLANT_PAGE}, choose "Download Dataset", '
              f'format "COCO" (no extra augmentation), and unzip it into {target} '
              '(so it contains train/, valid/, test/ with _annotations.coco.json files).')
    if not key:
        print('ROBOFLOW_API_KEY is not set, so FieldPlant was not downloaded.')
        print('Get a free key at https://app.roboflow.com (Settings > API Keys), then:')
        print('  pip install roboflow')
        print('  export ROBOFLOW_API_KEY=...   (PowerShell: $env:ROBOFLOW_API_KEY="...")')
        print(manual)
        return
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit('Run `pip install roboflow` first. ' + manual)
    project = Roboflow(api_key=key).workspace(FIELDPLANT_WORKSPACE).project(FIELDPLANT_PROJECT)
    if version is None:
        numbers = [int(str(v.version).rstrip('/').split('/')[-1]) for v in project.versions()]
        if not numbers:
            sys.exit('No FieldPlant versions found. ' + manual)
        version = max(numbers)
    print(f'Downloading FieldPlant version {version} (COCO format) to {target}')
    project.version(version).download('coco', location=str(target))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=['plantvillage', 'fieldplant', 'all'], default='all')
    parser.add_argument('--raw_dir', default='data/raw')
    parser.add_argument('--tomato_only', action='store_true',
                        help='PlantVillage: fetch only tomato classes (no OOD crops).')
    parser.add_argument('--fieldplant_version', type=int, default=None,
                        help='Roboflow version number; default is the latest.')
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset in {'plantvillage', 'all'}:
        download_plantvillage(raw_dir, args.tomato_only)
    if args.dataset in {'fieldplant', 'all'}:
        download_fieldplant(raw_dir, args.fieldplant_version)


if __name__ == '__main__':
    main()
