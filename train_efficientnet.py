import os
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# Disable oneDNN optimizations (optional, for consistent results)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Dataset path
DATA_DIR = r"C:\Users\Danish Omar\Desktop\SEM5\FYPPROJECT\data\archive"

# Data generators
datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=(224,224),
    color_mode="rgb",
    batch_size=64,
    class_mode="categorical"
)

val_gen = datagen.flow_from_directory(
    os.path.join(DATA_DIR, "validation"),
    target_size=(224,224),
    color_mode="rgb",
    batch_size=64,
    class_mode="categorical"
)

# Build model
inputs = Input(shape=(224,224,3))
base = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=inputs)
base.trainable = False  # Stage 1: freeze backbone

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(train_gen.num_classes, activation='softmax')(x)
model = Model(inputs, outputs)

# Common callbacks
callbacks_stage1 = [
    tf.keras.callbacks.ModelCheckpoint(
        "models/emotion_efficientnet_stage1.keras",
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
        verbose=1
    )
]

# Stage 1: train head
model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
history_stage1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=200,
    callbacks=callbacks_stage1
)

# Stage 2: fine-tune backbone
for layer in base.layers[-40:]:
    layer.trainable = True

callbacks_stage2 = [
    tf.keras.callbacks.ModelCheckpoint(
        "models/emotion_efficientnet_stage2.keras",
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
        verbose=1
    )
]

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
history_stage2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=200,
    initial_epoch=len(history_stage1.epoch),  # continue from Stage 1
    callbacks=callbacks_stage2
)

# Final save (best model after Stage 2)
model.save("models/emotion_efficientnet_final.keras")
