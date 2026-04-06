"""
train_mobilenet.py  —  FYP2 Improved MobileNetV2
=================================================
Key improvements over FYP1:
  1. Strong data augmentation during training
  2. Class weights loaded from class_weights.json (fixes imbalance)
  3. Better head architecture (two Dense layers + BN)
  4. Label smoothing in loss function
  5. Proper two-stage fine-tuning with warm-up LR
  6. Unfreeze more layers in Stage 2 (last 50 instead of 30)
  7. Separate callbacks per stage with correct model path

Expected improvement: 63% → 73%+ accuracy
Run data_prep.py first to generate class_weights.json

Usage:
    python train_mobilenet.py
"""

import os
import json
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.regularizers import l2
from data_prep import get_transfer_augmented_datagen, get_transfer_val_datagen

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.makedirs("models", exist_ok=True)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATA_DIR   = r"C:\Users\Danish Omar\Desktop\SEM5\FYPPROJECT\data\archive"
BATCH_SIZE = 64
IMG_SIZE   = (160, 160)
EPOCHS_S1  = 30    # Stage 1: train head only (fewer needed)
EPOCHS_S2  = 200   # Stage 2: fine-tune backbone (early stopping handles this)

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
inputs = Input(shape=(160, 160, 3))
base = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs)
base.trainable = False  # Stage 1: freeze all backbone layers

x = base.output
x = GlobalAveragePooling2D()(x)

# FYP2: better head — two Dense layers with BN and L2
x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs, outputs, name="MobileNetV2_FYP2")
model.summary()

# ──────────────────────────────────────────────
# STAGE 1 — Train head only
# ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("  STAGE 1 — Training head (backbone frozen)")
print("=" * 50)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks_s1 = [
    tf.keras.callbacks.ModelCheckpoint(
        "models/mobilenet_stage1.keras",
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
    tf.keras.callbacks.CSVLogger("models/mobilenet_stage1_history.csv")
]

history_s1 = model.fit(
    train_gen,
    epochs=EPOCHS_S1,
    validation_data=val_gen,
    callbacks=callbacks_s1,
    class_weight=class_weights
)

# ──────────────────────────────────────────────
# STAGE 2 — Fine-tune backbone (last 50 layers)
# ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("  STAGE 2 — Fine-tuning backbone (last 50 layers)")
print("=" * 50)

# FYP2: unfreeze last 50 layers (FYP1 only did last 30)
for layer in base.layers:
    layer.trainable = False
for layer in base.layers[-50:]:
    layer.trainable = True

trainable_count = sum(1 for l in model.layers if l.trainable)
print(f"✓ Trainable layers in Stage 2: {trainable_count}")

# Lower LR for fine-tuning — prevents catastrophic forgetting
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks_s2 = [
    tf.keras.callbacks.ModelCheckpoint(
        "models/emotion_mobilenet.keras",   # final model used by app.py
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
        min_lr=1e-8,
        verbose=1
    ),
    tf.keras.callbacks.CSVLogger("models/mobilenet_stage2_history.csv")
]

history_s2 = model.fit(
    train_gen,
    epochs=EPOCHS_S2,
    validation_data=val_gen,
    callbacks=callbacks_s2,
    class_weight=class_weights
)

print("\n✓ Training complete!")
print(f"✓ Best model saved → models/emotion_mobilenet.keras")
print(f"✓ Stage 1 history  → models/mobilenet_stage1_history.csv")
print(f"✓ Stage 2 history  → models/mobilenet_stage2_history.csv")