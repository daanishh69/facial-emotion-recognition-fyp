"""
data_prep.py  —  FYP2 RAF-DB Dataset Preparation
==================================================
For this folder structure:
  FYPPROJECT/
    data/
      DATASET/
        train/
          1/  2/  3/  4/  5/  6/  7/
        test/
          1/  2/  3/  4/  5/  6/  7/
    train_labels.csv
    test_labels.csv

RAF-DB Label Mapping:
  1 = Surprise
  2 = Fear
  3 = Disgust
  4 = Happy
  5 = Sad
  6 = Angry
  7 = Neutral

What this script does:
  1. Renames numbered folders (1-7) to emotion names
  2. Creates val/ split from train/ (20%, stratified)
  3. Visualizes class distribution
  4. Computes & saves class_weights.json

Usage:
    python data_prep.py
"""

import os
import json
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
BASE_DIR    = r"C:\Users\Danish Omar\Desktop\SEM5\FYPPROJECT"
DATASET_DIR = os.path.join(BASE_DIR, "data", "DATASET")
TRAIN_DIR   = os.path.join(DATASET_DIR, "train")
TEST_DIR    = os.path.join(DATASET_DIR, "test")
VAL_DIR     = os.path.join(DATASET_DIR, "val")

VAL_SPLIT   = 0.2
RANDOM_SEED = 42

# RAF-DB number → emotion name
LABEL_MAP = {
    "1": "surprise",
    "2": "fear",
    "3": "disgust",
    "4": "happy",
    "5": "sad",
    "6": "angry",
    "7": "neutral"
}

EMOTIONS = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]

# ──────────────────────────────────────────────
# 1. RENAME NUMBERED FOLDERS TO EMOTION NAMES
# ──────────────────────────────────────────────
print("=" * 55)
print("  RAF-DB DATASET PREPARATION")
print("=" * 55)

def rename_folders(directory):
    """Rename 1/ 2/ 3/... to surprise/ fear/ disgust/..."""
    renamed = 0
    for num, emotion in LABEL_MAP.items():
        src = os.path.join(directory, num)
        dst = os.path.join(directory, emotion)
        if os.path.exists(src) and not os.path.exists(dst):
            os.rename(src, dst)
            print(f"  Renamed: {num}/ → {emotion}/")
            renamed += 1
        elif os.path.exists(dst):
            print(f"  Already exists: {emotion}/  (skipping)")
    return renamed

print("\n→ Renaming train folders...")
rename_folders(TRAIN_DIR)

print("\n→ Renaming test folders...")
rename_folders(TEST_DIR)

print("\n✓ Folder renaming complete!")

# ──────────────────────────────────────────────
# 2. CREATE VALIDATION SPLIT FROM TRAIN
# ──────────────────────────────────────────────
print("\n→ Creating validation split (20% of train)...")

# Create val folder structure
for emotion in EMOTIONS:
    os.makedirs(os.path.join(VAL_DIR, emotion), exist_ok=True)

random.seed(RANDOM_SEED)
total_moved = 0

for emotion in EMOTIONS:
    train_emotion_dir = os.path.join(TRAIN_DIR, emotion)
    val_emotion_dir   = os.path.join(VAL_DIR, emotion)

    if not os.path.exists(train_emotion_dir):
        print(f"  ⚠ Folder not found: {train_emotion_dir}")
        continue

    # Check if val already has images (avoid re-splitting)
    existing_val = os.listdir(val_emotion_dir)
    if len(existing_val) > 0:
        print(f"  {emotion:<10} — val already exists ({len(existing_val)} images), skipping")
        continue

    all_images = [f for f in os.listdir(train_emotion_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    n_val = max(1, int(len(all_images) * VAL_SPLIT))
    val_images = random.sample(all_images, n_val)

    for img in val_images:
        src = os.path.join(train_emotion_dir, img)
        dst = os.path.join(val_emotion_dir, img)
        shutil.move(src, dst)

    total_moved += n_val
    print(f"  {emotion:<10} — moved {n_val} images to val/")

print(f"\n✓ Validation split complete! {total_moved} images moved to val/")

# ──────────────────────────────────────────────
# 3. PRINT CLASS DISTRIBUTION
# ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("  FINAL CLASS DISTRIBUTION")
print("=" * 55)

print(f"\n{'Emotion':<12} {'Train':>8} {'Val':>8} {'Test':>8}  {'Ratio':>8}")
print("-" * 52)

train_counts = {}
for emotion in EMOTIONS:
    folder = os.path.join(TRAIN_DIR, emotion)
    train_counts[emotion] = len(os.listdir(folder)) if os.path.exists(folder) else 0

max_count = max(train_counts.values()) if train_counts else 1

for emotion in EMOTIONS:
    tr_folder  = os.path.join(TRAIN_DIR, emotion)
    val_folder = os.path.join(VAL_DIR,   emotion)
    te_folder  = os.path.join(TEST_DIR,  emotion)

    tr = len(os.listdir(tr_folder))  if os.path.exists(tr_folder)  else 0
    va = len(os.listdir(val_folder)) if os.path.exists(val_folder) else 0
    te = len(os.listdir(te_folder))  if os.path.exists(te_folder)  else 0

    ratio = max_count / tr if tr > 0 else 999
    flag  = "  ⚠ SEVERE" if ratio > 5 else ("  ⚠ HIGH" if ratio > 2 else "")
    print(f"{emotion:<12} {tr:>8} {va:>8} {te:>8}  {ratio:>6.1f}x{flag}")

# ──────────────────────────────────────────────
# 4. VISUALIZE CLASS DISTRIBUTION
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("RAF-DB Class Distribution", fontsize=14, fontweight='bold')

for ax, (split_name, split_dir) in zip(axes, [
    ("Train", TRAIN_DIR),
    ("Val",   VAL_DIR),
    ("Test",  TEST_DIR)
]):
    counts = {}
    for emotion in EMOTIONS:
        folder = os.path.join(split_dir, emotion)
        counts[emotion] = len(os.listdir(folder)) if os.path.exists(folder) else 0

    values = list(counts.values())
    colors = ['#e74c3c' if v == min(values) else
              '#2ecc71' if v == max(values) else
              '#3498db' for v in values]

    bars = ax.bar(counts.keys(), values, color=colors,
                  edgecolor='black', linewidth=0.5)
    ax.set_title(f"{split_name} Set", fontweight='bold')
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=35)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 3,
                str(val), ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150, bbox_inches='tight')
print("\n✓ Saved → class_distribution.png")
plt.show()

# ──────────────────────────────────────────────
# 5. COMPUTE CLASS WEIGHTS
# ──────────────────────────────────────────────
emotion_to_idx = {e: i for i, e in enumerate(EMOTIONS)}
y_train = []

for emotion, idx in emotion_to_idx.items():
    folder = os.path.join(TRAIN_DIR, emotion)
    if os.path.exists(folder):
        count = len(os.listdir(folder))
        y_train.extend([idx] * count)

y_train  = np.array(y_train)
classes  = np.unique(y_train)
weights  = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
cw_dict  = {int(k): float(v) for k, v in zip(classes, weights)}

print("\n" + "=" * 55)
print("  COMPUTED CLASS WEIGHTS")
print("=" * 55)

idx_to_emotion = {v: k for k, v in emotion_to_idx.items()}
for idx, weight in cw_dict.items():
    emotion = idx_to_emotion.get(idx, str(idx))
    bar = "█" * int(weight * 4)
    print(f"  [{idx}] {emotion:<10}  weight = {weight:.4f}  {bar}")

with open("class_weights.json", "w") as f:
    json.dump(cw_dict, f, indent=2)

print("\n✓ Saved → class_weights.json")

# ──────────────────────────────────────────────
# 6. DATA GENERATORS (imported by training scripts)
# ──────────────────────────────────────────────
def get_augmented_datagen():
    return ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        shear_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

def get_val_datagen():
    return ImageDataGenerator(rescale=1./255)

def get_transfer_augmented_datagen(preprocess_fn):
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
    return ImageDataGenerator(preprocessing_function=preprocess_fn)


# ──────────────────────────────────────────────
# 7. SUMMARY
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  DONE — NEXT STEPS")
    print("=" * 55)
    print(f"""
  Update DATA_DIR in all training scripts to:
    {DATASET_DIR}

  Then run:
    python train_baseline_cnn.py
    python train_mobilenet.py
    python train_efficientnet.py

  Also update EMOTIONS list in all training scripts to:
    ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']
    (7 classes instead of 6 — RAF-DB includes Fear)
    """)