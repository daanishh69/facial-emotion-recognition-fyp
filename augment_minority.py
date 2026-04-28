"""
augment_minority.py  —  FYP2 Targeted Minority Class Augmentation
==================================================================
Problem:
  Fear:    225 training images (17x less than Happy)
  Disgust: 574 training images (6.8x less than Happy)

Solution:
  Generate synthetic images for Fear and Disgust ONLY by applying
  aggressive augmentation to existing samples until each class
  reaches the TARGET_COUNT threshold.

  This is different from runtime augmentation in training scripts —
  these are actual saved image files added to the training folder,
  giving the model more real variety to learn from.

Target:
  Fear:    225 → 1000 images (generate ~775 new images)
  Disgust: 574 → 1000 images (generate ~426 new images)

Usage:
    python augment_minority.py
"""

import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
TRAIN_DIR    = r"C:\Users\Danish Omar\Desktop\SEM5\FYPPROJECT\data\DATASET\train"
TARGET_COUNT = 1000    # generate until each minority class has this many images
RANDOM_SEED  = 42

# Only augment these minority classes
MINORITY_CLASSES = ["fear", "disgust"]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ──────────────────────────────────────────────
# AUGMENTATION FUNCTIONS
# ──────────────────────────────────────────────
def augment_image(img):
    """
    Apply a random combination of augmentations to a PIL image.
    More aggressive than training augmentation since we're
    generating saved files — variety is key.
    """
    augmentations = []

    # 1. Random horizontal flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        augmentations.append("flip")

    # 2. Random rotation (-25 to +25 degrees)
    angle = random.uniform(-25, 25)
    img = img.rotate(angle, fillcolor=(128, 128, 128))
    augmentations.append(f"rot{angle:.0f}")

    # 3. Random brightness (0.7 to 1.3)
    factor = random.uniform(0.7, 1.3)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # 4. Random contrast (0.7 to 1.3)
    factor = random.uniform(0.7, 1.3)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # 5. Random zoom (crop + resize back)
    if random.random() > 0.4:
        w, h = img.size
        zoom = random.uniform(0.85, 1.0)
        new_w, new_h = int(w * zoom), int(h * zoom)
        left   = random.randint(0, w - new_w)
        top    = random.randint(0, h - new_h)
        img    = img.crop((left, top, left + new_w, top + new_h))
        img    = img.resize((w, h), Image.LANCZOS)

    # 6. Slight sharpness variation
    if random.random() > 0.5:
        factor = random.uniform(0.8, 1.5)
        img = ImageEnhance.Sharpness(img).enhance(factor)

    # 7. Very slight blur (simulate different camera quality)
    if random.random() > 0.7:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

    return img


# ──────────────────────────────────────────────
# MAIN — GENERATE IMAGES
# ──────────────────────────────────────────────
print("=" * 55)
print("  TARGETED MINORITY CLASS AUGMENTATION")
print("=" * 55)

for emotion in MINORITY_CLASSES:
    emotion_dir = os.path.join(TRAIN_DIR, emotion)

    if not os.path.exists(emotion_dir):
        print(f"\n⚠ Folder not found: {emotion_dir}")
        continue

    # Get existing images
    existing = [f for f in os.listdir(emotion_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    current_count = len(existing)

    print(f"\n→ {emotion.upper()}")
    print(f"  Current count: {current_count}")
    print(f"  Target count:  {TARGET_COUNT}")

    if current_count >= TARGET_COUNT:
        print(f"  ✓ Already at target — skipping")
        continue

    needed = TARGET_COUNT - current_count
    print(f"  Generating:    {needed} new images...")

    generated = 0
    attempts  = 0
    max_attempts = needed * 10  # safety limit

    while generated < needed and attempts < max_attempts:
        # Pick a random source image
        src_file = random.choice(existing)
        src_path = os.path.join(emotion_dir, src_file)

        try:
            img = Image.open(src_path).convert("RGB")
            aug = augment_image(img)

            # Save with unique name
            base_name = os.path.splitext(src_file)[0]
            new_name  = f"aug_{base_name}_{generated:04d}.jpg"
            save_path = os.path.join(emotion_dir, new_name)

            # Skip if file already exists
            if os.path.exists(save_path):
                attempts += 1
                continue

            aug.save(save_path, "JPEG", quality=90)
            generated += 1

            if generated % 100 == 0:
                print(f"  ... {generated}/{needed} generated")

        except Exception as e:
            print(f"  ⚠ Error processing {src_file}: {e}")

        attempts += 1

    final_count = len([f for f in os.listdir(emotion_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    print(f"  ✓ Done! {emotion} now has {final_count} images (+{generated} generated)")

# ──────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("  SUMMARY")
print("=" * 55)

for emotion in MINORITY_CLASSES:
    folder = os.path.join(TRAIN_DIR, emotion)
    if os.path.exists(folder):
        count = len([f for f in os.listdir(folder)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  {emotion:<10} → {count} images")

print("""
Next steps:
  1. Re-run data_prep.py to update class_weights.json
     (class weights will change since counts changed)
  2. Retrain both models:
       python train_mobilenet.py
       python train_resnet50.py

Note:
  The generated images are saved directly into your train folders.
  They will be automatically picked up by the training scripts.
  You do NOT need to modify any training scripts.
""")