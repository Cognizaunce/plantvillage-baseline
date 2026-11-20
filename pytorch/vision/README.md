# Image Classification with PyTorch

This example trains a small CNN. Run all commands below from `pytorch/vision`
after installing the root requirements.

## Try the code without a download

```bash
python make_toy_dataset.py --output_dir data/toy
python train.py --data_dir data/toy --model_dir experiments/base_model --device cpu
python evaluate.py --data_dir data/toy --model_dir experiments/base_model --split val --device cpu
python predict.py --image data/toy/val_signs/0_val_000.png --model_dir experiments/base_model --device cpu
```

The toy images are synthetic colour/stripe patterns for checking the pipeline
only. They are not hand signs or a suitable project dataset.

## Use the hand-sign image format

The original image example uses RGB hand-sign images for six classes (0–5).
The [SIGNS dataset link](https://drive.google.com/file/d/1ufiR6hUKhXoAyiBNsySPkUwlvE_wfEHC/view)
is provided as an optional starting point; check availability and reuse rights.
Expected structure and filename labels:

```text
data/SIGNS/train_signs/0_IMG_5864.jpg
data/SIGNS/test_signs/0_IMG_5942.jpg
```

```bash
python build_dataset.py --data_dir data/SIGNS --output_dir data/64x64_SIGNS --seed 4110
python train.py --data_dir data/64x64_SIGNS --model_dir experiments/base_model
```

Data preparation resizes images and creates a reproducible train/validation split
from the raw training partition. It preserves the supplied test partition and
writes a split manifest. Use a new output directory for another preparation run.
Check duplicates and subject/source leakage for your actual dataset.

## Tomato leaf data (PlantVillage and FieldPlant)

`download_data.py` fetches the raw data into `data/raw` (ignored by Git).
PlantVillage needs `git`. FieldPlant comes from Roboflow
(https://universe.roboflow.com/plant-disease-detection/fieldplant, CC BY 4.0) and
needs a free API key plus `pip install roboflow`. Without a key the script prints
how to download the COCO export by hand into `data/raw/FieldPlant`.

```bash
python download_data.py --dataset all            # add --tomato_only to skip other PlantVillage crops
export ROBOFLOW_API_KEY=...                      # PowerShell: $env:ROBOFLOW_API_KEY="..."
python download_data.py --dataset fieldplant

python build_tomato_dataset.py --source plantvillage --seed 4110   # -> data/tomato_plantvillage
python build_tomato_dataset.py --source fieldplant --seed 4110     # -> data/tomato_fieldplant
```

`build_tomato_dataset.py` keeps only tomato classes and writes a stratified
70/15/15 train/val/test split (change with `--val_fraction` / `--test_fraction`)
in the `{split}_signs/<label>_<id>.jpg` format the loader already reads, so
`train.py`, `evaluate.py` and `predict.py` need no changes. Set `num_classes`
in `params.json` to the number the script prints (10 for PlantVillage) and
`image_size` as needed (images are saved at 224x224 by default).

- PlantVillage: exact duplicate files are removed before splitting.
- FieldPlant: each tomato leaf box is cropped to its own image, and splits are
  made per source photo so leaves from one photo never land in two splits.
  FieldPlant classes are renamed to their PlantVillage match using
  `tomato_label_map.json` (healthy, Mosaic, Leaf_Curling -> Yellow Leaf Curl
  Virus, Mildiou -> Late blight). Brown_Spots and bacterial_wilt have no
  match and are left out, so those diseases come only from PlantVillage.
  Both datasets share the same 10 label indices.
- `ood/<crop>/` holds up to `--ood_per_class` non-tomato leaves per class
  (apple, corn, cassava, ...) for OOD evaluation only. Put non-leaf photos in
  `ood/non_leaf/` yourself.
- `classes.json` maps label indices to class names. `split_manifest.json`
  records the seed, settings, every source file per split and a SHA-256 of
  the split, so the same seed and raw data give the same split.

## Run experiments

Edit `experiments/base_model/params.json` before training. Copy its configuration
into a new experiment folder for another run. Supported optimizers are Adam,
AdamW and SGD. `--device auto` selects CUDA, then MPS, then CPU; pass `--device cpu`
to force CPU. Use `--overwrite` only to explicitly restart an existing run.

```bash
python search_hyperparams.py --data_dir data/toy --parent_dir experiments/learning_rate
python synthesize_results.py --parent_dir experiments
```

Each run saves `history.csv`, `train.log`, resolved settings, environment details,
validation metrics, and best/last checkpoints. Use the CSV for learning curves.
Once choices are fixed, evaluate the selected model on the held-out test split:

```bash
python evaluate.py --data_dir data/toy --model_dir experiments/base_model --split test
```

## Adapt to your project

- `model/data_loader.py`: input format, labels, preprocessing and batching.
- `model/net.py`: architecture and loss. The example returns raw class logits.
- `train.py`: training choices and validation-based checkpoint selection.
- `utils.py`: epoch metrics; the example reports mean loss and image accuracy.
- `evaluate.py` and `predict.py`: task-specific evaluation and inference.

Set `num_classes`, preprocessing and augmentation appropriately. The example
selects checkpoints by validation accuracy. Change that criterion for tasks such
as regression or detection. The CNN supports partial batches, including one image.

## PlantVillage tomato baseline

`params.json` can set `"arch"` to `scratch` (the original small CNN),
`mobilenet_v3_large` or `efficientnet_b0`. The last two start from ImageNet
weights (`"pretrained": true`, downloaded on the first run) and need
`"imagenet_normalize": true`. `"select_metric": "macro_f1"` keeps the epoch with
the best validation macro-F1, since the tomato classes are unbalanced.

```
python train.py --data_dir data/tomato_plantvillage --model_dir experiments/plantvillage_mobilenetv3
python evaluate.py --data_dir data/tomato_plantvillage --model_dir experiments/plantvillage_mobilenetv3 --split val
```

`evaluate.py` prints accuracy, macro-F1 and per-class F1/recall, and saves the
confusion matrix in `metrics_<split>_<restore_file>.json`. Use `--split test`
only for the final numbers.
