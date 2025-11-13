# src/train_model.py

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# Paths (Windows safe)
# -----------------------------
train_dir = r"C:\Users\Danish Omar\Desktop\SEM5\FYPPROJECT\data\fer2013\archive\train"
test_dir  = r"C:\Users\Danish Omar\Desktop\SEM5\FYPPROJECT\data\fer2013\archive\test"
model_dir = r"C:\Users\Danish Omar\Desktop\SEM5\FYPPROJECT\models"
os.makedirs(model_dir, exist_ok=True)

# -----------------------------
# Image Data Generator
# -----------------------------
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical"
)

# -----------------------------
# Build CNN Model
# -----------------------------
model = Sequential([
    Input(shape=(48,48,1)),  # Fix input_shape warning
    Conv2D(32, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# Callbacks
# -----------------------------
checkpoint = ModelCheckpoint(
    os.path.join(model_dir, "emotion_model.keras"),  # Keras 3.x format
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    verbose=1
)

# -----------------------------
# Train Model
# -----------------------------
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=test_generator,
    callbacks=[checkpoint, earlystop],
    verbose=1
)
