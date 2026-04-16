"""
predict.py  —  FYP2 Improved Prediction Module
================================================
Changes from FYP1:
  1. Added face detection before emotion prediction
     (crops face from frame instead of using full image)
  2. Added predict_frame() for real-time video stream processing
  3. Cleaner preprocessing pipeline per model type
  4. Returns face bounding box coordinates for UI overlay
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

EMOTIONS = ['angry', 'disgust', 'happy', 'neutral', 'sad', 'surprise']

# Emotion to color mapping for UI
EMOTION_COLORS = {
    'angry':    '#FF4B4B',
    'disgust':  '#9B59B6',
    'happy':    '#00FF88',
    'neutral':  '#00CCFF',
    'sad':      '#4B9BFF',
    'surprise': '#FFD700'
}

# Emotion to emoji mapping
EMOTION_EMOJIS = {
    'angry':    '😠',
    'disgust':  '🤢',
    'happy':    '😊',
    'neutral':  '😐',
    'sad':      '😢',
    'surprise': '😮'
}

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def load_model(path):
    return tf.keras.models.load_model(path)


def detect_face(frame_bgr):
    """
    Detect face in a BGR frame using Haar cascade.
    Returns (x, y, w, h) of the largest face, or None if no face found.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    if len(faces) == 0:
        return None
    # Return the largest face
    return max(faces, key=lambda f: f[2] * f[3])


def preprocess_for_model(face_img_bgr, model_type):
    """
    Preprocess a cropped face image for the specified model.
    face_img_bgr: BGR numpy array (cropped face region)
    Returns: preprocessed numpy array ready for model.predict()
    """
    if model_type == "Baseline CNN":
        gray = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        x = resized.astype('float32') / 255.0
        x = np.expand_dims(x, axis=(0, -1))  # (1, 48, 48, 1)

    elif model_type == "MobileNetV2":
        rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (160, 160))
        x = resized.astype('float32')
        x = mobilenet_preprocess(x)
        x = np.expand_dims(x, axis=0)  # (1, 160, 160, 3)

    elif model_type == "EfficientNetB0":
        rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (260, 260))
        x = resized.astype('float32')
        x = efficientnet_preprocess(x)
        x = np.expand_dims(x, axis=0)  # (1, 260, 260, 3)

    return x


def predict(model, arr, is_baseline=False, model_type=None):
    """
    Original predict function — kept for compatibility with image upload.
    arr: numpy array (preprocessed image)
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


def predict_frame(model, frame_bgr, model_type):
    """
    FYP2 NEW — Predict emotion from a full video frame.
    Detects face first, then runs emotion prediction on cropped face.

    Returns:
        probs      : probability array for all 6 emotions
        label      : predicted emotion string
        face_coords: (x, y, w, h) of detected face, or None
    """
    face_coords = detect_face(frame_bgr)

    if face_coords is None:
        return None, None, None

    x, y, w, h = face_coords

    # Add padding around face (10%)
    pad = int(0.1 * w)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame_bgr.shape[1], x + w + pad)
    y2 = min(frame_bgr.shape[0], y + h + pad)

    face_crop = frame_bgr[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None, None, None

    preprocessed = preprocess_for_model(face_crop, model_type)
    probs = model.predict(preprocessed, verbose=0)[0]
    idx = np.argmax(probs)

    return probs, EMOTIONS[idx], (x1, y1, x2 - x1, y2 - y1)