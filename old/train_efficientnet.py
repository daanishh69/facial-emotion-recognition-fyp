# train_efficientnet_fixed.py
import os, tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.makedirs("models", exist_ok=True)

IMG_SIZE = (224, 224)
BATCH = 64
TRAIN_DIR = r"C:\Users\Danish Omar\Desktop\SEM5\FYPPROJECT\data\fer2013\archive\train"

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # EfficientNet expects specific normalization
    validation_split=0.2,
    rotation_range=8,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, color_mode="rgb",
    batch_size=BATCH, class_mode="categorical", subset="training"
)
val_gen = datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, color_mode="rgb",
    batch_size=BATCH, class_mode="categorical", subset="validation"
)

inputs = Input(shape=IMG_SIZE + (3,))
base = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=inputs)
base.trainable = False  # stage 1

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(7, activation='softmax')(x)
model = Model(inputs, outputs)

model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

class_weights = {0:1.0, 1:3.0, 2:1.2, 3:0.8, 4:0.9, 5:1.0, 6:1.0}

model.fit(
    train_gen, epochs=10, validation_data=val_gen,
    class_weight=class_weights,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint("models/emotion_efficientnet.keras", save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]
)

# Stage 2: fine-tune
for layer in base.layers[-40:]:
    layer.trainable = True
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train_gen, epochs=6, validation_data=val_gen,
    class_weight=class_weights,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint("models/emotion_efficientnet.keras", save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]
)
print("✅ EfficientNet fine-tuned and saved")
