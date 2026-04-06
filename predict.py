import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

EMOTIONS = ['angry','disgust','happy','neutral','sad','surprise']

def load_model(path):
    return tf.keras.models.load_model(path)

def predict(model, arr, is_baseline=False, model_type=None):
    """
    arr: numpy array of shape (48,48) for baseline CNN (grayscale)
         or (H,W,3) for transfer models (RGB)
    is_baseline: True if baseline CNN, False otherwise
    model_type: "MobileNetV2" or "EfficientNetB0" for correct preprocessing
    """
    if is_baseline:
        x = np.expand_dims(arr / 255.0, axis=(0, -1))
    else:
        arr = arr.astype('float32')
        if model_type == "MobileNetV2":
            arr = mobilenet_preprocess(arr)
        elif model_type == "EfficientNetB0":
            arr = efficientnet_preprocess(arr)
        x = np.expand_dims(arr, 0)  

    probs = model.predict(x, verbose=0)[0]
    idx = np.argmax(probs)
    return probs, EMOTIONS[idx], idx

