"""
augment_balance.py  —  FYP2 Full Dataset Balancing
====================================================
Problem:
  Happy:    3818 images  (dominant)
  Neutral:  2020 images
  Sad:      1586 images
  Surprise: 1032 images
  Disgust:   574 images  ⚠ severe
  Angry:     564 images  ⚠ severe
  Fear:      225 images  ⚠ severe

Solution:
  Generate augmented images for ALL classes below TARGET_COUNT
  until every class has the same number of training images.
  This completely eliminates class imbalance from the data level
  (class weights still applied during training as extra safety).

Target: 2000 images per class (balanced, manageable size)
  Happy:    3818 → keep as is (already above target, we don't remove)
  Neutral:  2020 → keep as is
  Sad:      1586 → generate ~414 new images
  Surprise: 1032 → generate ~968 new images
  Disgust:   574 → generate ~1426 new images
  Angry:     564 → generate ~1436 new images
  Fear:      225 → generate ~1775 new images

Usage:
    python augment_balance.py
"""

import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
TRAIN_DIR    = r"C:\Users\Danish Omar\Desktop\SEM5\FYPPROJECT\data\DATASET\train"
TARGET_COUNT = 2000    # target images per class
RANDOM_SEED  = 42

EMOTIONS = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ──────────────────────────────────────────────
# AUGMENTATION PIPELINE
# ──────────────────────────────────────────────
def augment_image(img):
    """
    Apply random combination of augmentations to a PIL image.
    Aggressive enough to create genuinely different-looking images.
    """
    # 1. Random horizontal flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 2. Random rotation (-25 to +25 degrees)
    angle = random.uniform(-25, 25)
    img = img.rotate(angle, fillcolor=(128, 128, 128))

    # 3. Random brightness
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))

    # 4. Random contrast
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))

    # 5. Random zoom (crop + resize)
    if random.random() > 0.4:
        w, h  = img.size
        zoom  = random.uniform(0.85, 1.0)
        new_w = int(w * zoom)
        new_h = int(h * zoom)
        left  = random.randint(0, w - new_w)
        top   = random.randint(0, h - new_h)
        img   = img.crop((left, top, left + new_w, top + new_h))
        img   = img.resize((w, h), Image.LANCZOS)

    # 6. Random sharpness
    if random.random() > 0.5:
        img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.8, 1.5))

    # 7. Slight blur (simulate camera quality variation)
    if random.random() > 0.7:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

    # 8. Random saturation
    if random.random() > 0.5:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))

    return img


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
print("=" * 55)
print("  FULL DATASET BALANCING — ALL 7 CLASSES")
print("=" * 55)
print(f"  Target per class: {TARGET_COUNT} images\n")

total_generated = 0

for emotion in EMOTIONS:
    emotion_dir = os.path.join(TRAIN_DIR, emotion)

    if not os.path.exists(emotion_dir):
        print(f"⚠ Folder not found: {emotion_dir}")
        continue

    # Count only original + previously generated images
    all_files = [f for f in os.listdir(emotion_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    current_count = len(all_files)

    print(f"→ {emotion.upper():<10}  current: {current_count:>4}", end="")

    if current_count >= TARGET_COUNT:
        print(f"  ✓ Already at/above target ({current_count}) — skipping")
        continue

    needed = TARGET_COUNT - current_count
    print(f"  →  generating {needed} images...")

    # Use only original images as source (not previously augmented ones)
    # to avoid augmenting augmented images repeatedly
    original_files = [f for f in all_files if not f.startswith("aug_")]
    if not original_files:
        original_files = all_files  # fallback if all are augmented

    generated = 0
    attempts  = 0
    max_attempts = needed * 20

    while generated < needed and attempts < max_attempts:
        src_file = random.choice(original_files)
        src_path = os.path.join(emotion_dir, src_file)

        try:
            img      = Image.open(src_path).convert("RGB")
            aug      = augment_image(img)
            base     = os.path.splitext(src_file)[0]
            new_name = f"aug_{base}_{generated:05d}.jpg"
            dst_path = os.path.join(emotion_dir, new_name)

            if os.path.exists(dst_path):
                attempts += 1
                continue

            aug.save(dst_path, "JPEG", quality=90)
            generated += 1

            if generated % 200 == 0:
                print(f"   ... {generated}/{needed}")

        except Exception as e:
            print(f"  ⚠ Error: {e}")

        attempts += 1

    final_count = len([f for f in os.listdir(emotion_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  ✓ {emotion:<10} now has {final_count} images (+{generated} generated)")
    total_generated += generated

# ──────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("  FINAL CLASS DISTRIBUTION")
print("=" * 55)

print(f"\n{'Emotion':<12} {'Count':>8}  {'Status':>10}")
print("-" * 35)

for emotion in EMOTIONS:
    folder = os.path.join(TRAIN_DIR, emotion)
    if os.path.exists(folder):
        count = len([f for f in os.listdir(folder)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        status = "✓ balanced" if count >= TARGET_COUNT else f"⚠ {count}"
        print(f"{emotion:<12} {count:>8}  {status}")

print(f"\n✓ Total new images generated: {total_generated}")
print(f"""
IMPORTANT — Next steps:
  1. Re-run data_prep.py to update class_weights.json:
       python data_prep.py

  2. Retrain both models:
       python train_mobilenet.py
       python train_resnet50.py

Why re-run data_prep.py?
  Class weights are computed from actual file counts.
  Since counts changed, weights must be recalculated.
  Running data_prep.py also skips the val split step
  (val already exists) so it's safe to re-run.
""")