"""
train_baseline_cnn.py  —  FYP2 Baseline CNN (v2 - Overfitting Fix)
===================================================================
Problem found after FYP2 v1 training:
  - Train accuracy: 85%+ but Val accuracy stuck at ~68%
  - Gap of 17% = severe overfitting

Fixes applied in v2:
  1. Increased Dropout rates (0.25 → 0.4 in conv blocks, 0.5 → 0.6 in head)
  2. Stronger L2 regularization (1e-4 → 2e-4)
  3. Reduced model capacity slightly (256 filters → 128 in block 4)
  4. More aggressive ReduceLROnPlateau (factor 0.5 → 0.3)
  5. Increased EarlyStopping patience (15 → 20)

Expected: val_accuracy ~70-73%, smaller train/val gap

Usage:
    python train_baseline_cnn.py
"""

import os
import json
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from data_prep import get_augmented_datagen, get_val_datagen

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.makedirs("models", exist_ok=True)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATA_DIR   = r"C:\Users\Danish Omar\Desktop\SEM5\FYPPROJECT\data\archive"
BATCH_SIZE = 64
IMG_SIZE   = (48, 48)
EPOCHS     = 200

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
train_gen = get_augmented_datagen().flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_gen = get_val_datagen().flow_from_directory(
    os.path.join(DATA_DIR, "validation"),
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_gen.num_classes
print(f"\n✓ Classes: {train_gen.class_indices}")
print(f"✓ Train samples: {train_gen.samples}")
print(f"✓ Val samples:   {val_gen.samples}\n")

# ──────────────────────────────────────────────
# MODEL — stronger regularization vs v1
# ──────────────────────────────────────────────
REG = l2(2e-4)   # stronger than v1's 1e-4

model = Sequential([
    # Block 1
    Conv2D(32, (3,3), padding='same', activation='relu',
           input_shape=(48,48,1), kernel_regularizer=REG),
    BatchNormalization(),
    Conv2D(32, (3,3), padding='same', activation='relu',
           kernel_regularizer=REG),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.4),

    # Block 2
    Conv2D(64, (3,3), padding='same', activation='relu',
           kernel_regularizer=REG),
    BatchNormalization(),
    Conv2D(64, (3,3), padding='same', activation='relu',
           kernel_regularizer=REG),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.4),

    # Block 3
    Conv2D(128, (3,3), padding='same', activation='relu',
           kernel_regularizer=REG),
    BatchNormalization(),
    Conv2D(128, (3,3), padding='same', activation='relu',
           kernel_regularizer=REG),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.4),

    # Block 4 — reduced from 256 to 128 filters (less capacity = less overfit)
    Conv2D(128, (3,3), padding='same', activation='relu',
           kernel_regularizer=REG),
    BatchNormalization(),
    Dropout(0.4),

    # Head
    GlobalAveragePooling2D(),
    Dense(256, activation='relu', kernel_regularizer=REG),
    BatchNormalization(),
    Dropout(0.6),
    Dense(128, activation='relu', kernel_regularizer=REG),
    Dropout(0.4),
    Dense(NUM_CLASSES, activation='softmax')
], name="Baseline_CNN_FYP2_v2")

model.summary()

# ──────────────────────────────────────────────
# COMPILE
# ──────────────────────────────────────────────
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# ──────────────────────────────────────────────
# CALLBACKS
# ──────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "models/baseline_cnn.keras",
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=7,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.CSVLogger("models/baseline_cnn_history.csv")
]

# ──────────────────────────────────────────────
# TRAIN
# ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("  TRAINING BASELINE CNN v2 (Overfitting Fix)")
print("=" * 50)

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    class_weight=class_weights
)

best_val = max(history.history['val_accuracy'])
print(f"\n✓ Training complete!")
print(f"✓ Best val_accuracy: {best_val:.4f} ({best_val*100:.2f}%)")
print(f"✓ Best model saved → models/baseline_cnn.keras")
print(f"✓ Training history → models/baseline_cnn_history.csv")