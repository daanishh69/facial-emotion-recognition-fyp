"""
train_mobilenet_fearfix.py  —  MobileNetV2 with Focal Loss for Fear Fix
========================================================================
Changes from previous version:
  - Uses FocalLoss class instead of focal_loss function
  - FocalLoss is fully serializable — no loading errors

Usage:
    python train_mobilenet_fearfix.py
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
from focal_loss import FocalLoss
from data_prep import get_transfer_augmented_datagen, get_transfer_val_datagen

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.makedirs("models", exist_ok=True)

DATA_DIR   = r"C:\Users\Danish Omar\Desktop\SEM5\FYPPROJECT\data\DATASET"
BATCH_SIZE = 32
IMG_SIZE   = (160, 160)

# Load class weights
with open("class_weights.json", "r") as f:
    class_weights = {int(k): v for k, v in json.load(f).items()}

print("✓ Loaded class weights:")
for k, v in class_weights.items():
    print(f"   Class {k}: {v:.4f}")

# Generators
train_gen = get_transfer_augmented_datagen(preprocess_input).flow_from_directory(
    os.path.join(DATA_DIR, "train"), target_size=IMG_SIZE,
    color_mode="rgb", batch_size=BATCH_SIZE, class_mode="categorical", shuffle=True)

val_gen = get_transfer_val_datagen(preprocess_input).flow_from_directory(
    os.path.join(DATA_DIR, "val"), target_size=IMG_SIZE,
    color_mode="rgb", batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)

NUM_CLASSES = train_gen.num_classes
print(f"\n✓ Classes: {train_gen.class_indices}")
print(f"✓ Training samples:   {train_gen.samples}")
print(f"✓ Validation samples: {val_gen.samples}\n")

# Build model
inputs = Input(shape=(160, 160, 3))
base   = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs)
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)
model   = Model(inputs, outputs, name="MobileNetV2_FearFix")
model.summary()

# ── Stage 1 — head only with Focal Loss ──
print("\n" + "=" * 50)
print("  STAGE 1 — Head training with Focal Loss")
print("=" * 50)

model.compile(
    optimizer=Adam(1e-3),
    loss=FocalLoss(gamma=2.0, alpha=0.25),
    metrics=['accuracy']
)

callbacks_s1 = [
    tf.keras.callbacks.ModelCheckpoint("models/mobilenet_fearfix_s1.keras",
        save_best_only=True, monitor='val_accuracy', verbose=1),
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    tf.keras.callbacks.CSVLogger("models/mobilenet_fearfix_s1_history.csv")
]

history_s1 = model.fit(train_gen, epochs=30, validation_data=val_gen,
                        callbacks=callbacks_s1, class_weight=class_weights)

best_s1 = max(history_s1.history['val_accuracy'])
print(f"\n✓ Stage 1 complete. Best val accuracy: {best_s1:.4f}")

# ── Stage 2 — fine-tune last 80 layers ──
print("\n" + "=" * 50)
print("  STAGE 2 — Fine-tuning last 80 layers")
print("=" * 50)

for layer in base.layers:
    layer.trainable = False
for layer in base.layers[-80:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=Adam(5e-5),
    loss=FocalLoss(gamma=2.0, alpha=0.25),
    metrics=['accuracy']
)

callbacks_s2 = [
    tf.keras.callbacks.ModelCheckpoint("models/emotion_mobilenet.keras",
        save_best_only=True, monitor='val_accuracy', verbose=1),
    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=7, min_lr=1e-9, verbose=1),
    tf.keras.callbacks.CSVLogger("models/mobilenet_fearfix_s2_history.csv")
]

model.fit(train_gen, epochs=200, validation_data=val_gen,
          callbacks=callbacks_s2, class_weight=class_weights)

print("\n✓ Fear-fixed model saved → models/emotion_mobilenet.keras")