import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Import from predict.py
from predict import load_model, predict

# Emotion classes (update if dataset changes)
EMOTIONS = ['angry', 'disgust', 'happy', 'neutral', 'sad', 'surprise']

# -----------------------------
# Page config + Cyberpunk style
# -----------------------------
st.set_page_config(page_title="Facial Emotion Recognition", layout="centered")
st.markdown("""
<style>
    .main {background-color: #0a0a0a; color: #00ffcc;}
    .stButton>button {background: #ff00ff; color: white; border-radius: 10px; font-weight: bold;}
    .stButton>button:hover {background: #00ffcc; color: black;}
    h1, h2, h3 {color: #ff00ff; text-shadow: 0 0 10px #00ffcc;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar navigation
# -----------------------------
page = st.sidebar.radio("Navigation", ["Prediction", "Evaluation"])

# -----------------------------
# MODEL PATHS
# -----------------------------
MODEL_PATHS = {
    "MobileNetV2": "models/emotion_mobilenet.keras",
    "EfficientNetB0": "models/emotion_efficientnet.keras",
    "Baseline CNN": "models/baseline_cnn.keras",
}

# -----------------------------
# Prediction Page
# -----------------------------
if page == "Prediction":
    st.title("Facial Emotion Recognition")
    st.write("Upload a photo or use webcam → AI detects your emotion in real-time.")

    # Model selection
    model_choice = st.sidebar.selectbox(
        "Choose Model",
        ["MobileNetV2", "EfficientNetB0", "Baseline CNN"],
        index=0
    )

    path = MODEL_PATHS[model_choice]

    def get_model(p):
        return load_model(p)

    if not os.path.exists(path):
        st.error(f"Model not found: {path}\n\n"
                 "→ Run the training script for this model first!\n"
                 "   python train_mobilenet.py   (or train_efficientnet.py / train_baseline_cnn.py)")
        st.stop()

    model = get_model(path)
    st.success(f"Loaded: {model_choice}")

    # Input: Upload or Webcam
    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    with col2:
        camera = st.camera_input("Webcam Capture")

    # Prediction flow
    if uploaded or camera:
        img = Image.open(uploaded or camera)

        if model_choice == "Baseline CNN":
            gray = img.convert('L').resize((48, 48))
            arr = np.array(gray)
            probs, label, idx = predict(model, arr, is_baseline=True)

        elif model_choice == "MobileNetV2":
            rgb = img.convert('RGB').resize((160, 160))
            arr = np.array(rgb)
            probs, label, idx = predict(model, arr, is_baseline=False, model_type="MobileNetV2")

        elif model_choice == "EfficientNetB0":
            rgb = img.convert('RGB').resize((224, 224))
            arr = np.array(rgb)
            probs, label, idx = predict(model, arr, is_baseline=False, model_type="EfficientNetB0")

        # Display results
        st.image(img, caption="Your Photo", use_column_width=True)
        st.markdown(f"## Predicted: **{label.upper()}**")

        # Bar chart
        chart_data = {e.capitalize(): float(p) for e, p in zip(EMOTIONS, probs)}
        st.bar_chart(chart_data)

        # Top-3 predictions
        st.write("### Top 3 Predictions")
        top3_idx = np.argsort(probs)[-3:][::-1]
        for i in top3_idx:
            st.write(f"- **{EMOTIONS[i].capitalize()}**: {probs[i]:.3f}")

    else:
        st.info("Upload an image or use the webcam to see the magic!")

# -----------------------------
# Evaluation Page
# -----------------------------
elif page == "Evaluation":
    st.title("Model Evaluation Dashboard")

    model_choice = st.selectbox("Choose Model to Evaluate", ["Baseline CNN", "MobileNetV2", "EfficientNetB0"])
    path = MODEL_PATHS[model_choice]

    if not os.path.exists(path):
        st.error(f"Model not found: {path}\n\nTrain the model first!")
        st.stop()

    model = tf.keras.models.load_model(path)
    st.success(f"✅ Loaded model: {model_choice}")

    # Pick correct input size
    if model_choice == "Baseline CNN":
        target_size, color_mode = (48,48), "grayscale"
    elif model_choice == "MobileNetV2":
        target_size, color_mode = (160,160), "rgb"
    else:
        target_size, color_mode = (224,224), "rgb"

    datagen = ImageDataGenerator(rescale=1./255)
    test_gen = datagen.flow_from_directory(
        "data/archive/test",
        target_size=target_size,
        color_mode=color_mode,
        batch_size=64,
        class_mode="categorical",
        shuffle=False
    )

    # Evaluate
    loss, acc = model.evaluate(test_gen, verbose=0)
    st.markdown(f"### 📊 Test Accuracy: **{acc:.4f}**")
    st.markdown(f"### 📉 Test Loss: **{loss:.4f}**")

    # Confusion matrix
    y_true = test_gen.classes
    y_pred = np.argmax(model.predict(test_gen), axis=1)

    cm = confusion_matrix(y_true, y_pred)
    labels = list(test_gen.class_indices.keys())

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm",
                xticklabels=labels, yticklabels=labels, ax=ax)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(f"Confusion Matrix - {model_choice}")
    st.pyplot(fig)

    # Classification report
    report = classification_report(y_true, y_pred, target_names=labels)
    st.text("Classification Report:\n" + report)
