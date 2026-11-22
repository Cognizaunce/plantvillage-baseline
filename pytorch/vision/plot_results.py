"""Generate report figures from saved experiment outputs.

Run from pytorch/vision/:
    python plot_results.py

Outputs (saved to plots/):
    fig1_learning_curve.png   -- train/val loss and val macro-F1 over epochs
    fig2_per_class_f1.png     -- per-class F1 before vs after FieldPlant fine-tune
    fig3_ood_gap.png          -- overall accuracy and macro-F1 before vs after
"""
import json
from pathlib import Path
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PLOTS_DIR = Path('plots')
HISTORY_CSV = Path('experiments/fieldplant_finetune_mobilenetv3/history.csv')
BEFORE_JSON = Path('experiments/plantvillage_mobilenetv3/metrics_test_best_fieldplant.json')
AFTER_JSON  = Path('experiments/fieldplant_finetune_mobilenetv3/metrics_test_best.json')
CLASSES_JSON = Path('data/tomato_fieldplant/classes.json')

PLOTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def load_history(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows

def short_name(full):
    """Shorten Tomato___Disease_name -> Disease name."""
    name = full.replace('Tomato___', '').replace('_', ' ')
    replacements = {
        'Tomato Yellow Leaf Curl Virus': 'YL Curl Virus',
        'Spider mites Two-spotted spider mite': 'Spider Mites',
        'Tomato mosaic virus': 'Mosaic Virus',
    }
    return replacements.get(name, name)

# ---------------------------------------------------------------------------
# Figure 1 — Learning curve
# ---------------------------------------------------------------------------
print('Reading history.csv ...')
history = load_history(HISTORY_CSV)
epochs      = [r['epoch'] for r in history]
train_loss  = [r['train_loss'] for r in history]
val_loss    = [r['val_loss'] for r in history]
val_f1      = [r['val_macro_f1'] for r in history]
val_acc     = [r['val_accuracy'] for r in history]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

ax1.plot(epochs, train_loss, 'o-', label='Train loss', color='steelblue')
ax1.plot(epochs, val_loss,   's--', label='Val loss',   color='tomato')
ax1.set_ylabel('Cross-entropy loss')
ax1.legend(framealpha=0.7)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.plot(epochs, val_acc, 'o-',  label='Val accuracy', color='steelblue')
ax2.plot(epochs, val_f1,  's--', label='Val macro-F1', color='darkorange')
ax2.set_ylabel('Score')
ax2.set_xlabel('Epoch')
ax2.set_ylim(0, 1.05)
ax2.legend(framealpha=0.7)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax2.grid(True, linestyle=':', alpha=0.6)

fig.suptitle('FieldPlant fine-tune: MobileNetV3-Large', fontsize=11)
fig.tight_layout()
fig.savefig(PLOTS_DIR / 'fig1_learning_curve.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved fig1_learning_curve.png')

# ---------------------------------------------------------------------------
# Figure 2 — Per-class F1 before vs after
# ---------------------------------------------------------------------------
print('Reading metrics JSON files ...')
before = load_json(BEFORE_JSON)
after  = load_json(AFTER_JSON)

CLASS_NAMES_FULL = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]
classes_file = Path(CLASSES_JSON)
if classes_file.exists():
    class_names = [short_name(load_json(classes_file)['classes'][str(i)]) for i in range(10)]
else:
    class_names = [short_name(n) for n in CLASS_NAMES_FULL]

f1_before_all = before['per_class_f1']
f1_after_all  = after['per_class_f1']

# Only show classes present in FieldPlant (at least one model has non-zero F1)
present = [i for i in range(10) if f1_before_all[i] > 0 or f1_after_all[i] > 0]
f1_before = [f1_before_all[i] for i in present]
f1_after  = [f1_after_all[i]  for i in present]
names_present = [class_names[i] for i in present]

x = np.arange(len(present))
w = 0.38

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(x, f1_after, w * 1.6, color='darkorange', alpha=0.88)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(names_present, rotation=20, ha='right', fontsize=10)
ax.set_ylabel('F1 score')
ax.set_ylim(0, 1.1)
ax.set_title('Fine-tuned MobileNetV3: per-class F1 on FieldPlant test set\n'
             '(6 PlantVillage classes absent from FieldPlant — not shown)', fontsize=10)
ax.grid(axis='y', linestyle=':', alpha=0.6)
fig.tight_layout()
fig.savefig(PLOTS_DIR / 'fig2_per_class_f1.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved fig2_per_class_f1.png')

# ---------------------------------------------------------------------------
# Figure 3 — OOD gap summary
# ---------------------------------------------------------------------------
metrics = {
    'Accuracy': (before['accuracy'], after['accuracy']),
    'Macro-F1': (before['macro_f1'], after['macro_f1']),
}

fig, ax = plt.subplots(figsize=(5, 4))
x = np.arange(len(metrics))
w = 0.3
labels = list(metrics.keys())
vals_before = [metrics[k][0] for k in labels]
vals_after  = [metrics[k][1] for k in labels]

bars1 = ax.bar(x - w/2, vals_before, w, label='PlantVillage only', color='steelblue', alpha=0.85)
bars2 = ax.bar(x + w/2, vals_after,  w, label='+ FieldPlant fine-tune', color='darkorange', alpha=0.85)

for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Score')
ax.set_title('Lab-to-field gap: FieldPlant test set', fontsize=11)
ax.legend(framealpha=0.7)
ax.grid(axis='y', linestyle=':', alpha=0.6)
fig.tight_layout()
fig.savefig(PLOTS_DIR / 'fig3_ood_gap.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved fig3_ood_gap.png')

print(f'\nAll figures saved to {PLOTS_DIR.resolve()}')
