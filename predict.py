import numpy as np
import tensorflow as tf

# Emotion labels must match your training order
EMOTIONS = ['angry','disgust','fear','happy','neutral','sad','surprise']

def load_model(model_path: str):
    """Load a saved Keras model."""
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_grayscale_to_rgb48(img_array: np.ndarray) -> np.ndarray:
    """
    For transfer learning models (MobileNetV2, EfficientNetB0).
    Input: 2D grayscale uint8 array (48x48)
    Output: 4D float32 tensor (1,48,48,3) scaled to [0,1]
    """
    rgb = np.repeat(img_array[..., None], 3, axis=-1)  # (48,48,3)
    rgb = (rgb / 255.0).astype('float32')
    return np.expand_dims(rgb, axis=0)  # (1,48,48,3)

def preprocess_grayscale_to_1channel(img_array: np.ndarray) -> np.ndarray:
    """
    For baseline CNN that expects (48,48,1).
    Input: 2D grayscale uint8 array (48x48)
    Output: 4D float32 tensor (1,48,48,1) scaled to [0,1]
    """
    x = (img_array / 255.0).astype('float32')
    return np.expand_dims(x, axis=(0, -1))  # (1,48,48,1)

def predict(model, input_tensor: np.ndarray):
    """
    Run prediction on the input tensor.
    Returns: (probabilities array, top_label string, top_index int)
    """
    probs = model.predict(input_tensor, verbose=0)[0]  # (num_classes,)
    top_idx = int(np.argmax(probs))
    return probs, EMOTIONS[top_idx], top_idx
