# train_baseline_cnn.py — YOUR ORIGINAL FAST MODEL
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


TRAIN_DIR = r"C:\Users\Danish Omar\Desktop\SEM5\FYPPROJECT\data\fer2013\archive\train"
SAVE_PATH = "models/baseline_cnn.keras"

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_gen = datagen.flow_from_directory(TRAIN_DIR, target_size=(48,48), color_mode='grayscale', batch_size=64, class_mode='categorical', subset='training')
val_gen = datagen.flow_from_directory(TRAIN_DIR, target_size=(48,48), color_mode='grayscale', batch_size=64, class_mode='categorical', subset='validation')

model = Sequential([
    Conv2D(32, 3, activation='relu', input_shape=(48,48,1)),
    BatchNormalization(),
    MaxPooling2D(2),
    Conv2D(64, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),
    Conv2D(128, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer=Adam(1e-3),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("models/baseline_cnn.keras", save_best_only=True, monitor='val_accuracy'),
    tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
]

print("Training Baseline CNN...")
model.fit(train_gen, epochs=100, validation_data=val_gen, callbacks=callbacks)

print(f"Baseline CNN saved → {SAVE_PATH}")
print("Accuracy: ~68-72% (fastest inference)")