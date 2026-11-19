"""Filter PlantVillage or FieldPlant to tomato leaves and write stratified train/val/test splits.

Output uses the same layout as model/data_loader.py, so train.py works unchanged:
    <output_dir>/{train,val,test}_signs/<label>_<id>.jpg
    <output_dir>/classes.json          label index -> original class name
    <output_dir>/split_manifest.json   seed, settings, every source file per split, SHA-256
    <output_dir>/ood/<crop>/...        non-tomato leaves kept for OOD evaluation (never trained on)

PlantVillage: class folders named <Crop>___<Disease>. Exact duplicate files are removed
before splitting (a file found in two classes is dropped).
FieldPlant: Roboflow COCO export with leaf bounding boxes. Each tomato box is cropped to a
leaf image. Splits are made per source photo, so leaves from one photo never appear in two
splits. Roboflow augmented copies of a photo are collapsed to one.

Labels: both sources use the 10 PlantVillage tomato classes from tomato_label_map.json,
so a label index means the same disease in both datasets. FieldPlant classes are renamed
to their PlantVillage match; FieldPlant tomato classes mapped to null (no PlantVillage
match, e.g. Brown_Spots, bacterial_wilt) are left out entirely.

The split is stratified by class with a fixed seed. Reruns require a fresh output directory.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
from PIL import Image

IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}
CROP_ALIASES = {'manioc': 'cassava', 'maize': 'corn'}
SPLITS = ('train', 'val', 'test')


def crop_name(class_name):
    crop = class_name.replace(' ', '_').split('_')[0].lower()
    return CROP_ALIASES.get(crop, crop)


def is_tomato(class_name):
    return crop_name(class_name) == 'tomato'


def norm(name):
    return name.strip().lower().replace(' ', '_')


def apply_label_map(source, tomato, label_map):
    """Return (kept items with PlantVillage class names, class list, dropped counts)."""
    classes = label_map['plantvillage_classes']
    if source == 'plantvillage':
        found = sorted({i['class_name'] for i in tomato})
        if found != sorted(classes):
            raise ValueError(f'PlantVillage tomato classes {found} do not match '
                             f'tomato_label_map.json {classes}')
        return tomato, classes, {}
    mapping = {norm(k): v for k, v in label_map['fieldplant_to_plantvillage'].items()}
    unknown = sorted({i['class_name'] for i in tomato if norm(i['class_name']) not in mapping})
    if unknown:
        raise ValueError(f'FieldPlant classes {unknown} are missing from tomato_label_map.json; '
                         'map each to a PlantVillage class or null.')
    bad = sorted({v for v in mapping.values() if v is not None and v not in classes})
    if bad:
        raise ValueError(f'Mapped classes {bad} are not in plantvillage_classes.')
    kept, dropped = [], Counter()
    for item in tomato:
        target = mapping[norm(item['class_name'])]
        if target is None:
            dropped[item['class_name']] += 1
        else:
            kept.append(dict(item, class_name=target, original_class=item['class_name']))
    if dropped:
        print(f'Left out FieldPlant classes with no PlantVillage match: {dict(dropped)}')
    return kept, classes, dict(dropped)


def file_digest(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def collect_plantvillage(root):
    """Return items for every image in <Crop>___<Disease> folders under root."""
    folders = defaultdict(list)
    for folder in sorted(p for p in root.rglob('*') if p.is_dir() and '___' in p.name):
        if any(f.suffix.lower() in IMAGE_EXTS for f in folder.iterdir()):
            folders[folder.name].append(folder)
    if not folders:
        raise ValueError(f'No PlantVillage class folders (<Crop>___<Disease>) under {root}')
    items = []
    for name, candidates in sorted(folders.items()):
        if len(candidates) > 1:  # e.g. color/grayscale/segmented variants side by side
            colour = [c for c in candidates if 'color' in c.as_posix().lower()]
            if len(colour) != 1:
                raise ValueError(f'Class {name} found in several folders {candidates}; '
                                 'point --raw_dir at one variant (e.g. raw/color).')
            candidates = colour
        for path in sorted(candidates[0].iterdir()):
            if path.suffix.lower() in IMAGE_EXTS:
                items.append(dict(src=path, class_name=name, group=str(path), box=None))
    return dedupe_exact(items)


def dedupe_exact(items):
    by_hash = defaultdict(list)
    for item in items:
        by_hash[file_digest(item['src'])].append(item)
    kept, dropped = [], 0
    for copies in by_hash.values():
        if len({c['class_name'] for c in copies}) > 1:
            dropped += len(copies)  # same pixels with conflicting labels
        else:
            kept.append(copies[0])
            dropped += len(copies) - 1
    if dropped:
        print(f'Removed {dropped} exact duplicate / conflicting files.')
    return sorted(kept, key=lambda i: str(i['src']))


def collect_fieldplant(root, min_box):
    """Return one item per annotated leaf box in Roboflow COCO exports under root."""
    jsons = sorted(root.rglob('*.json'))
    if not jsons:
        raise ValueError(f'No COCO annotation JSON files under {root}')
    items, seen_photos, skipped_small = [], set(), 0
    for json_path in jsons:
        coco = json.loads(json_path.read_text(encoding='utf-8'))
        if not {'images', 'annotations', 'categories'} <= coco.keys():
            continue
        categories = {c['id']: c['name'] for c in coco['categories']}
        images = {i['id']: i for i in coco['images']}
        boxes = defaultdict(list)
        for ann in coco['annotations']:
            boxes[ann['image_id']].append(ann)
        for image_id, image in sorted(images.items()):
            # Roboflow names files <original>_jpg.rf.<hash>.jpg; augmented copies share <original>.
            photo = Path(image['file_name']).name.split('.rf.')[0]
            if photo in seen_photos or image_id not in boxes:
                continue
            seen_photos.add(photo)
            for ann in boxes[image_id]:
                x, y, w, h = ann['bbox']
                if min(w, h) < min_box:
                    skipped_small += 1
                    continue
                items.append(dict(src=json_path.parent/image['file_name'],
                                  class_name=categories[ann['category_id']],
                                  group=photo, box=[x, y, w, h]))
    if not items:
        raise ValueError(f'No usable annotations found under {root}')
    if skipped_small:
        print(f'Skipped {skipped_small} boxes smaller than {min_box}px.')
    return items


def split_by_group(items, rng, val_fraction, test_fraction):
    """Stratified split of groups (photos); each group's label is its most common class."""
    groups = defaultdict(list)
    for item in items:
        groups[item['group']].append(item)
    by_label = defaultdict(list)
    for key, members in sorted(groups.items()):
        counts = Counter(m['class_name'] for m in members)
        label = min(counts, key=lambda c: (-counts[c], c))
        by_label[label].append(key)
    assignment = {}
    for label, keys in sorted(by_label.items()):
        rng.shuffle(keys)
        n = len(keys)
        n_test = round(n*test_fraction)
        n_val = round(n*val_fraction)
        if n >= 3:
            n_test, n_val = max(1, n_test), max(1, n_val)
        else:
            print(f'Warning: class {label} has only {n} source image(s); all go to train.')
            n_test = n_val = 0
        for i, key in enumerate(keys):
            assignment[key] = 'test' if i < n_test else 'val' if i < n_test+n_val else 'train'
    splits = {s: [] for s in SPLITS}
    for item in items:
        splits[assignment[item['group']]].append(item)
    return splits


def load_image(item, image_size, margin):
    with Image.open(item['src']) as image:
        image = image.convert('RGB')
        if item['box'] is not None:  # square crop around the box so leaves are not squashed
            x, y, w, h = item['box']
            side = max(w, h)*(1+margin)
            cx, cy = x+w/2, y+h/2
            left, top = max(0, cx-side/2), max(0, cy-side/2)
            right, bottom = min(image.width, cx+side/2), min(image.height, cy+side/2)
            image = image.crop((round(left), round(top), round(right), round(bottom)))
        return image.resize((image_size, image_size), Image.Resampling.BILINEAR)


def record(item, raw_dir, name):
    entry = dict(file=name, source=item['src'].relative_to(raw_dir).as_posix(),
                 class_name=item['class_name'])
    if 'original_class' in item:
        entry['original_class'] = item['original_class']
    if item['box'] is not None:
        entry['box_xywh'] = [round(v, 1) for v in item['box']]
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', choices=['plantvillage', 'fieldplant'], required=True)
    parser.add_argument('--raw_dir', default=None,
                        help='Default: data/raw/PlantVillage-Dataset/raw/color or data/raw/FieldPlant')
    parser.add_argument('--output_dir', default=None,
                        help='Default: data/tomato_<source>')
    parser.add_argument('--seed', type=int, default=4110)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--val_fraction', type=float, default=0.15)
    parser.add_argument('--test_fraction', type=float, default=0.15)
    parser.add_argument('--ood_per_class', type=int, default=100,
                        help='Non-tomato images kept per class for OOD; 0 = none, -1 = all.')
    parser.add_argument('--min_box', type=float, default=32, help='FieldPlant: min box side (px).')
    parser.add_argument('--box_margin', type=float, default=0.1, help='FieldPlant: crop padding.')
    parser.add_argument('--label_map', default=str(Path(__file__).with_name('tomato_label_map.json')))
    args = parser.parse_args()
    if not (0 < args.val_fraction < 1 and 0 < args.test_fraction < 1
            and args.val_fraction+args.test_fraction < 1) or args.image_size < 16:
        raise ValueError('Choose 0 < val/test fractions, val+test < 1, and image_size >= 16.')
    default_raw = {'plantvillage': 'data/raw/PlantVillage-Dataset/raw/color',
                   'fieldplant': 'data/raw/FieldPlant'}[args.source]
    raw_dir = Path(args.raw_dir or default_raw)
    destination = Path(args.output_dir or f'data/tomato_{args.source}')
    if destination.exists() and any(p.name != '.gitkeep' for p in destination.iterdir()):
        raise FileExistsError('Use a fresh output directory to prevent stale split membership.')
    if not raw_dir.exists():
        raise FileNotFoundError(f'{raw_dir} not found; run download_data.py first.')

    items = (collect_plantvillage(raw_dir) if args.source == 'plantvillage'
             else collect_fieldplant(raw_dir, args.min_box))
    tomato = [i for i in items if is_tomato(i['class_name'])]
    others = [i for i in items if not is_tomato(i['class_name'])]
    if not tomato:
        raise ValueError('No tomato classes found.')
    label_map = json.loads(Path(args.label_map).read_text(encoding='utf-8'))
    tomato, class_names, dropped = apply_label_map(args.source, tomato, label_map)
    if not tomato:
        raise ValueError('No tomato leaves left after applying the label map.')
    label_of = {name: index for index, name in enumerate(class_names)}

    rng = random.Random(args.seed)
    splits = split_by_group(tomato, rng, args.val_fraction, args.test_fraction)
    records, counter = {}, 0
    for split in SPLITS:
        output = destination/f'{split}_signs'
        output.mkdir(parents=True, exist_ok=True)
        records[split] = []
        for item in splits[split]:
            name = f'{label_of[item["class_name"]]}_{args.source}_{counter:06d}.jpg'
            counter += 1
            load_image(item, args.image_size, args.box_margin).save(output/name, quality=95)
            records[split].append(record(item, raw_dir, f'{split}_signs/{name}'))

    # OOD pool: other crops' leaves, sampled per class. Also add your own non-leaf photos
    # under ood/non_leaf/ (that folder is not filled by this script).
    records['ood'] = []
    if args.ood_per_class != 0:
        by_class = defaultdict(list)
        for item in others:
            by_class[item['class_name']].append(item)
        for name, members in sorted(by_class.items()):
            rng.shuffle(members)
            if args.ood_per_class > 0:
                members = members[:args.ood_per_class]
            output = destination/'ood'/crop_name(name)
            output.mkdir(parents=True, exist_ok=True)
            for item in members:
                file = f'{args.source}_{counter:06d}.jpg'
                counter += 1
                load_image(item, args.image_size, args.box_margin).save(output/file, quality=95)
                records['ood'].append(record(item, raw_dir, f'ood/{crop_name(name)}/{file}'))
    (destination/'ood'/'non_leaf').mkdir(parents=True, exist_ok=True)

    (destination/'classes.json').write_text(json.dumps(
        {'num_classes': len(class_names), 'classes': {str(i): n for i, n in enumerate(class_names)}},
        indent=2)+'\n', encoding='utf-8')
    payload = json.dumps(records, sort_keys=True).encode()
    manifest = dict(source=args.source, raw_dir=raw_dir.as_posix(), seed=args.seed,
                    image_size=args.image_size, val_fraction=args.val_fraction,
                    test_fraction=args.test_fraction, ood_per_class=args.ood_per_class,
                    min_box=args.min_box, box_margin=args.box_margin, label_map=label_map,
                    dropped_classes=dropped, files=records,
                    split_sha256=hashlib.sha256(payload).hexdigest())
    (destination/'split_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n',
                                                   encoding='utf-8')

    print(f'{len(class_names)} PlantVillage tomato classes -> set "num_classes": '
          f'{len(class_names)} in params.json')
    for split in SPLITS:
        counts = Counter(r['class_name'] for r in records[split])
        print(f'  {split:5s} {len(records[split]):6d}  ' +
              ', '.join(f'{label_of[n]}:{counts[n]}' for n in class_names if counts[n]))
    ood_counts = Counter(Path(r['file']).parent.name for r in records['ood'])
    print(f'  ood   {len(records["ood"]):6d}  {dict(sorted(ood_counts.items()))}')
    print(f'Wrote {destination} (classes.json, split_manifest.json)')


if __name__ == '__main__':
    main()
