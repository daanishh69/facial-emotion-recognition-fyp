"""
data_prep.py  —  FYP2 Dataset Analysis & Class Imbalance Fix
=============================================================
Run this FIRST before any training script.

What it does:
  1. Scans your train/val/test folders and prints class distribution
  2. Visualizes the imbalance as a bar chart (saved as class_distribution.png)
  3. Computes class weights to handle imbalance during training
  4. Saves class_weights.json → used automatically by all training scripts
  5. Defines the improved ImageDataGenerator with strong augmentation

Usage:
    python data_prep.py

Make sure DATA_DIR below points to your local dataset root folder.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ──────────────────────────────────────────────
# CONFIG — update this path to your local setup
# ──────────────────────────────────────────────
DATA_DIR = r"C:\Users\User\Desktop\SEM5\FYPPROJECT\data\archive"

EMOTIONS = ['angry', 'disgust', 'happy', 'neutral', 'sad', 'surprise']

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "validation")
TEST_DIR  = os.path.join(DATA_DIR, "test")


# ──────────────────────────────────────────────
# 1. COUNT SAMPLES PER CLASS
# ──────────────────────────────────────────────
def count_samples(directory):
    counts = {}
    for emotion in sorted(os.listdir(directory)):
        path = os.path.join(directory, emotion)
        if os.path.isdir(path):
            counts[emotion] = len(os.listdir(path))
    return counts

print("=" * 50)
print("  DATASET CLASS DISTRIBUTION ANALYSIS")
print("=" * 50)

train_counts = count_samples(TRAIN_DIR)
val_counts   = count_samples(VAL_DIR)
test_counts  = count_samples(TEST_DIR)

print(f"\n{'Emotion':<12} {'Train':>8} {'Val':>8} {'Test':>8}  {'Imbalance Ratio':>16}")
print("-" * 56)

train_values = list(train_counts.values())
max_train = max(train_values)

for emotion in train_counts:
    tr = train_counts.get(emotion, 0)
    va = val_counts.get(emotion, 0)
    te = test_counts.get(emotion, 0)
    ratio = max_train / tr if tr > 0 else 999
    flag = "  ⚠ SEVERE" if ratio > 5 else ("  ⚠ HIGH" if ratio > 2 else "")
    print(f"{emotion:<12} {tr:>8} {va:>8} {te:>8}  {ratio:>10.1f}x {flag}")

total_train = sum(train_counts.values())
print("-" * 56)
print(f"{'TOTAL':<12} {total_train:>8}")

print(f"\n→ Most imbalanced class: "
      f"{min(train_counts, key=train_counts.get)} "
      f"({min(train_counts.values())} samples)")
print(f"→ Most common class:     "
      f"{max(train_counts, key=train_counts.get)} "
      f"({max(train_counts.values())} samples)")


# ──────────────────────────────────────────────
# 2. VISUALIZE CLASS DISTRIBUTION
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("FER+ Dataset Class Distribution", fontsize=14, fontweight='bold')

for ax, (counts, split) in zip(axes, [
    (train_counts, "Train"),
    (val_counts,   "Validation"),
    (test_counts,  "Test")
]):
    emotions = list(counts.keys())
    values   = list(counts.values())
    colors   = ['#e74c3c' if v == min(values) else
                '#2ecc71' if v == max(values) else
                '#3498db' for v in values]

    bars = ax.bar(emotions, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_title(f"{split} Set", fontweight='bold')
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Sample Count")
    ax.tick_params(axis='x', rotation=30)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20,
                str(val), ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150, bbox_inches='tight')
print("\n✓ Saved class distribution chart → class_distribution.png")
plt.show()


# ──────────────────────────────────────────────
# 3. COMPUTE CLASS WEIGHTS
# ──────────────────────────────────────────────
# Build label array from train folder
y_train = []
class_to_idx = {e: i for i, e in enumerate(sorted(train_counts.keys()))}

for emotion, idx in class_to_idx.items():
    folder = os.path.join(TRAIN_DIR, emotion)
    if os.path.isdir(folder):
        count = len(os.listdir(folder))
        y_train.extend([idx] * count)

y_train = np.array(y_train)
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

class_weight_dict = {int(k): float(v) for k, v in zip(classes, weights)}

print("\n" + "=" * 50)
print("  COMPUTED CLASS WEIGHTS")
print("=" * 50)
idx_to_emotion = {v: k for k, v in class_to_idx.items()}
for idx, weight in class_weight_dict.items():
    emotion = idx_to_emotion.get(idx, str(idx))
    bar = "█" * int(weight * 4)
    print(f"  [{idx}] {emotion:<10}  weight = {weight:.4f}  {bar}")

print("\n→ Higher weight = model penalized MORE for missing that class")
print("→ This forces the model to pay attention to disgust & sad")

# Save to JSON
with open("class_weights.json", "w") as f:
    json.dump(class_weight_dict, f, indent=2)

print("\n✓ Saved class weights → class_weights.json")
print("  (All training scripts will load this automatically)\n")


# ──────────────────────────────────────────────
# 4. IMPROVED DATA GENERATORS
# ──────────────────────────────────────────────
# These are imported by training scripts — do not run standalone

def get_augmented_datagen():
    """
    Strong augmentation for training.
    Especially helps minority classes (disgust, sad) by creating
    more varied versions of the limited samples.
    """
    return ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,          # rotate ±20°
        width_shift_range=0.15,     # shift horizontally ±15%
        height_shift_range=0.15,    # shift vertically ±15%
        zoom_range=0.15,            # zoom in/out ±15%
        horizontal_flip=True,       # mirror faces (natural variation)
        shear_range=0.1,            # slight shear distortion
        brightness_range=[0.8, 1.2],# lighting variation
        fill_mode='nearest'
    )

def get_val_datagen():
    """
    Validation/test generators — NO augmentation, just rescale.
    """
    return ImageDataGenerator(rescale=1./255)

def get_transfer_augmented_datagen(preprocess_fn):
    """
    Augmented generator for transfer learning models (MobileNetV2, EfficientNetB0).
    Uses model-specific preprocessing instead of simple rescale.
    """
    return ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        shear_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

def get_transfer_val_datagen(preprocess_fn):
    """
    Val/test generator for transfer learning models.
    """
    return ImageDataGenerator(preprocessing_function=preprocess_fn)


# ──────────────────────────────────────────────
# 5. SUMMARY
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  SUMMARY — WHAT TO DO NEXT")
    print("=" * 50)
    print("""
  Files generated:
    ✓ class_distribution.png  — visual chart of your dataset
    ✓ class_weights.json      — weights for training scripts

  Next steps:
    1. Check class_distribution.png to confirm imbalance
    2. Run the improved training scripts:
         python train_baseline_cnn.py
         python train_mobilenet.py
         python train_efficientnet.py
    3. All scripts will automatically load class_weights.json
       and use the augmented data generators from this file.

  Expected improvement over FYP1:
    - Disgust F1-score: 0.407 → ~0.55+
    - Sad F1-score:     0.368 → ~0.50+
    - Overall accuracy: 63%   → 70%+ (before model improvements)
    """)