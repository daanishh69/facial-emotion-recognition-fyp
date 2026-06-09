"""
predict.py  —  FYP2 Final Prediction Module
============================================
Models:
  - MobileNetV2  → Live Detection (160x160 RGB)
  - DenseNet121  → Image Prediction (224x224 RGB)

Dataset: RAF-DB (7 emotions)
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

# ──────────────────────────────────────────────
# 7 EMOTIONS (RAF-DB)
# ──────────────────────────────────────────────
EMOTIONS = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

EMOTION_COLORS = {
    'surprise': '#FFD700',
    'fear':     '#9B59B6',
    'disgust':  '#2ECC71',
    'happy':    '#00FF88',
    'sad':      '#4B9BFF',
    'angry':    '#FF4B4B',
    'neutral':  '#00CCFF',
}

EMOTION_EMOJIS = {
    'surprise': '😮',
    'fear':     '😨',
    'disgust':  '🤢',
    'happy':    '😊',
    'sad':      '😢',
    'angry':    '😠',
    'neutral':  '😐',
}

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def load_model(path):
    return tf.keras.models.load_model(path)


def detect_face(frame_bgr):
    """
    Detect largest face in a BGR frame using Haar cascade.
    Returns (x, y, w, h) or None if no face found.
    """
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def predict(model, img_rgb, model_type):
    """
    Predict emotion from a PIL RGB image array.
    img_rgb   : numpy array (H, W, 3) RGB format
    model_type: "MobileNetV2" or "DenseNet121"
    Returns   : (probs, label, idx)
    """
    if model_type == "MobileNetV2":
        resized = cv2.resize(img_rgb, (160, 160))
        x = mobilenet_preprocess(resized.astype('float32'))
    elif model_type == "DenseNet121":
        resized = cv2.resize(img_rgb, (224, 224))
        x = densenet_preprocess(resized.astype('float32'))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    x     = np.expand_dims(x, axis=0)
    probs = model.predict(x, verbose=0)[0]
    idx   = np.argmax(probs)
    return probs, EMOTIONS[idx], idx


def predict_frame(model, frame_bgr, model_type):
    """
    Predict emotion from a full video frame.
    Detects face first, then predicts on cropped face.
    Returns: (probs, label, face_coords) or (None, None, None)
    """
    face_coords = detect_face(frame_bgr)
    if face_coords is None:
        return None, None, None

    x, y, w, h = face_coords
    pad = int(0.1 * w)
    x1  = max(0, x - pad)
    y1  = max(0, y - pad)
    x2  = min(frame_bgr.shape[1], x + w + pad)
    y2  = min(frame_bgr.shape[0], y + h + pad)

    face_crop = frame_bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None, None, None

    face_rgb          = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    probs, label, _   = predict(model, face_rgb, model_type)
    return probs, label, (x1, y1, x2 - x1, y2 - y1)