import os, tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.makedirs("models", exist_ok=True)

DATA_DIR = r"C:\Users\Danish Omar\Desktop\SEM5\FYPPROJECT\data\archive"

datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=(160,160),
    color_mode="rgb",
    batch_size=64,
    class_mode="categorical"
)

val_gen = datagen.flow_from_directory(
    os.path.join(DATA_DIR, "validation"),
    target_size=(160,160),
    color_mode="rgb",
    batch_size=64,
    class_mode="categorical"
)

# Build model
inputs = Input(shape=(160,160,3))
base = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs)
base.trainable = False  # Stage 1: freeze backbone

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)   
x = Dropout(0.5)(x)                    
outputs = Dense(train_gen.num_classes, activation='softmax')(x)  
model = Model(inputs, outputs)

# Stage 1: train head
model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("models/emotion_mobilenet.keras", save_best_only=True, monitor='val_accuracy'),
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
]
model.fit(train_gen, epochs=200, validation_data=val_gen, callbacks=callbacks)

# Stage 2: fine-tune backbone
for layer in base.layers[-30:]:
    layer.trainable = True
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, epochs=200, validation_data=val_gen, callbacks=callbacks)
