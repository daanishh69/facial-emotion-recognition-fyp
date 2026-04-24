"""
train_resnet50.py  —  FYP2 ResNet50 on RAF-DB
==============================================
Why ResNet50 for FYP2:
  - Replaces Baseline CNN which was too shallow for real-world faces
  - Skip connections (residual connections) prevent vanishing gradient
  - Better at learning subtle emotion differences (fear, disgust)
  - Pre-trained on ImageNet — strong feature extractor for faces
  - Higher accuracy than Baseline CNN and comparable to EfficientNetB0

Architecture:
  ResNet50 backbone (ImageNet weights)
  → GlobalAveragePooling2D
  → Dense(512) + BN + Dropout(0.5)
  → Dense(256) + BN + Dropout(0.3)
  → Dense(7, softmax)

Two-stage training:
  Stage 1: Train head only (backbone frozen) — 30 epochs
  Stage 2: Fine-tune last 30 layers — up to 200 epochs

Usage:
    python train_resnet50.py
"""

import os
import json
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.regularizers import l2
from data_prep import get_transfer_augmented_datagen, get_transfer_val_datagen

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.makedirs("models", exist_ok=True)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATA_DIR   = r"C:\Users\Danish Omar\Desktop\SEM5\FYPPROJECT\data\DATASET"
BATCH_SIZE = 32
IMG_SIZE   = (224, 224)   # ResNet50 native resolution
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
    os.path.join(DATA_DIR, "val"),
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_gen.num_classes
print(f"\n✓ Classes found: {train_gen.class_indices}")
print(f"✓ NUM_CLASSES: {NUM_CLASSES}")
print(f"✓ Training samples:   {train_gen.samples}")
print(f"✓ Validation samples: {val_gen.samples}\n")

# ──────────────────────────────────────────────
# BUILD MODEL
# ──────────────────────────────────────────────
inputs = Input(shape=(224, 224, 3))
base = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
base.trainable = False   # Stage 1: freeze backbone

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs, outputs, name="ResNet50_RAFDB")
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
        "models/resnet50_stage1.keras",
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
    tf.keras.callbacks.CSVLogger("models/resnet50_stage1_history.csv")
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
    print("⚠ Stage 1 val accuracy too low — check DATA_DIR and folder structure.")
    exit()

# ──────────────────────────────────────────────
# STAGE 2 — Fine-tune last 30 layers
# ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("  STAGE 2 — Fine-tuning last 30 layers")
print("=" * 50)

# Unfreeze last 30 layers, keep BatchNorm frozen
for layer in base.layers:
    layer.trainable = False
for layer in base.layers[-30:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

trainable_count = sum(1 for l in model.layers if l.trainable)
print(f"✓ Trainable layers in Stage 2: {trainable_count}")
print("  (BatchNorm layers kept frozen)\n")

model.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks_s2 = [
    tf.keras.callbacks.ModelCheckpoint(
        "models/emotion_resnet50.keras",
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
    tf.keras.callbacks.CSVLogger("models/resnet50_stage2_history.csv")
]

history_s2 = model.fit(
    train_gen,
    epochs=EPOCHS_S2,
    validation_data=val_gen,
    callbacks=callbacks_s2,
    class_weight=class_weights
)

best_s2_acc = max(history_s2.history['val_accuracy'])
print(f"\n✓ Training complete!")
print(f"✓ Best val_accuracy: {best_s2_acc:.4f} ({best_s2_acc*100:.2f}%)")
print(f"✓ Best model saved → models/emotion_resnet50.keras")
print(f"✓ Stage 1 history  → models/resnet50_stage1_history.csv")
print(f"✓ Stage 2 history  → models/resnet50_stage2_history.csv")