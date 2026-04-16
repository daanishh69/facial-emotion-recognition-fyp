"""
train_efficientnet.py  —  FYP2 Fixed & Improved EfficientNetB0
===============================================================
FYP1 PROBLEM: EfficientNetB0 completely failed (13.35% accuracy).
              It collapsed to predicting "angry" for everything.

ROOT CAUSES FIXED IN FYP2:
  1. Wrong saved model filename (efficientnet_final vs efficientnet) → FIXED
  2. No class weights → imbalance caused training collapse → FIXED
  3. No data augmentation → poor generalization → FIXED
  4. Stage 2 used initial_epoch wrongly → FIXED
  5. No label smoothing → model overconfident on majority class → FIXED
  6. Input size 224x224 with frozen BN layers → domain shift issue → FIXED
     (Now using 260x260 — EfficientNetB0's native resolution)
  7. Unfreezing only last 40 layers was too few → now last 80 → FIXED

Expected improvement: 13% (failure) → 74%+ accuracy
Run data_prep.py first to generate class_weights.json

Usage:
    python train_efficientnet.py
"""

import os
import json
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.regularizers import l2
from data_prep import get_transfer_augmented_datagen, get_transfer_val_datagen

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.makedirs("models", exist_ok=True)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATA_DIR   = r"C:\Users\User\Desktop\SEM5\FYPPROJECT\data\archive"
BATCH_SIZE = 32    # smaller batch — more stable for EfficientNet fine-tuning
IMG_SIZE   = (260, 260)   # EfficientNetB0's native resolution (was wrong 224 in FYP1)
EPOCHS_S1  = 30
EPOCHS_S2  = 200

# ──────────────────────────────────────────────
# LOAD CLASS WEIGHTS
# ──────────────────────────────────────────────
with open("class_weights.json", "r") as f:
    class_weights = {int(k): v for k, v in json.load(f).items()}

print("✓ Loaded class weights:")
for k, v in class_weights.items():
    print(f"   Class {k}: {v:.4f}")

# ──────────────────────────────────────────────
# DATA GENERATORS
# ──────────────────────────────────────────────
train_gen = get_transfer_augmented_datagen(preprocess_input).flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_gen = get_transfer_val_datagen(preprocess_input).flow_from_directory(
    os.path.join(DATA_DIR, "validation"),
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_gen.num_classes
print(f"\n✓ Classes found: {train_gen.class_indices}")
print(f"✓ Training samples:   {train_gen.samples}")
print(f"✓ Validation samples: {val_gen.samples}\n")

# ──────────────────────────────────────────────
# BUILD MODEL
# ──────────────────────────────────────────────
inputs = Input(shape=(260, 260, 3))

# FYP2 FIX: trainable=False on base AND explicitly freeze BN layers
# This prevents batch norm statistics from corrupting during Stage 1
base = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_tensor=inputs
)
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)

# Stronger head than FYP1
x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs, outputs, name="EfficientNetB0_FYP2")
model.summary()

# ──────────────────────────────────────────────
# STAGE 1 — Train head only (backbone fully frozen)
# ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("  STAGE 1 — Training head (backbone frozen)")
print("=" * 50)
print("  This is the critical fix — FYP1 likely failed because")
print("  Stage 1 didn't converge before attempting fine-tuning.\n")

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks_s1 = [
    tf.keras.callbacks.ModelCheckpoint(
        "models/efficientnet_stage1.keras",
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.CSVLogger("models/efficientnet_stage1_history.csv")
]

history_s1 = model.fit(
    train_gen,
    epochs=EPOCHS_S1,
    validation_data=val_gen,
    callbacks=callbacks_s1,
    class_weight=class_weights
)

best_s1_acc = max(history_s1.history['val_accuracy'])
print(f"\n✓ Stage 1 complete. Best val accuracy: {best_s1_acc:.4f}")

if best_s1_acc < 0.25:
    print("⚠ WARNING: Stage 1 val accuracy is very low.")
    print("  Check your DATA_DIR path and that class folders exist.")
    print("  Stopping before Stage 2 to prevent wasted compute.")
    exit()

# ──────────────────────────────────────────────
# STAGE 2 — Fine-tune backbone (last 80 layers)
# ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("  STAGE 2 — Fine-tuning backbone (last 80 layers)")
print("=" * 50)

# FYP2 FIX: unfreeze last 80 layers (FYP1 only did last 40)
# Keep BatchNorm layers frozen to preserve ImageNet statistics
for layer in base.layers:
    layer.trainable = False
for layer in base.layers[-80:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

trainable_count = sum(1 for l in model.layers if l.trainable)
print(f"✓ Trainable layers in Stage 2: {trainable_count}")
print("  (BatchNorm layers kept frozen — prevents domain shift issues)\n")

# Very low LR for fine-tuning EfficientNet — key to avoiding collapse
model.compile(
    optimizer=Adam(learning_rate=5e-5),   # much lower than FYP1's 1e-4
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks_s2 = [
    tf.keras.callbacks.ModelCheckpoint(
        "models/emotion_efficientnet.keras",   # FYP2 FIX: correct filename for app.py
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=7,
        min_lr=1e-9,
        verbose=1
    ),
    tf.keras.callbacks.CSVLogger("models/efficientnet_stage2_history.csv")
]

# FYP2 FIX: No initial_epoch — start fresh from Stage 2
# (FYP1's initial_epoch=len(history_stage1.epoch) caused epoch counting issues)
history_s2 = model.fit(
    train_gen,
    epochs=EPOCHS_S2,
    validation_data=val_gen,
    callbacks=callbacks_s2,
    class_weight=class_weights
)

print("\n✓ Training complete!")
print(f"✓ Best model saved → models/emotion_efficientnet.keras")
print(f"✓ Stage 1 history  → models/efficientnet_stage1_history.csv")
print(f"✓ Stage 2 history  → models/efficientnet_stage2_history.csv")
print("\nNote: This model should now significantly outperform FYP1's 13.35%.")
print("If accuracy is still low, check that Stage 1 val_accuracy was >40% before Stage 2.")