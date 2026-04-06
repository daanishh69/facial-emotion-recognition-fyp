import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd  

# Import from predict.py
from predict import load_model, predict

# Emotion classes
EMOTIONS = ['angry', 'disgust', 'happy', 'neutral', 'sad', 'surprise']

# -----------------------------
# Page config 
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

    if not os.path.exists(path):
        st.error(f"Model not found: {path}\n\n"
                 "→ Run the training script for this model first!\n"
                 "   example: python train_mobilenet.py")
        st.stop()

    model = load_model(path)
    st.success(f"Loaded: {model_choice}")

    # Input: Upload or Webcam
    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    with col2:
        camera = st.camera_input("Live Webcam Capture")

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

        st.image(img, caption="Input Image", width="stretch")

        st.markdown(f"## Predicted Emotion: **{label.upper()}**")

        chart_data = {e.capitalize(): float(p) for e, p in zip(EMOTIONS, probs)}
        st.bar_chart(chart_data)

        st.write("### Top 3 Predictions")
        top3_idx = np.argsort(probs)[-3:][::-1]
        for i in top3_idx:
            st.write(f"- **{EMOTIONS[i].capitalize()}**: {probs[i]:.3f}")

    else:
        st.info("Upload an image or use the webcam to get started!")

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
    st.success(f"Loaded model: {model_choice}")

    # Target size & color mode
    if model_choice == "Baseline CNN":
        target_size, color_mode = (48, 48), "grayscale"
    elif model_choice == "MobileNetV2":
        target_size, color_mode = (160, 160), "rgb"
    else:
        target_size, color_mode = (224, 224), "rgb"

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

    # 1. Test Accuracy
    st.markdown("### 1. Test Accuracy")
    st.markdown(f"**{acc:.4f} ({acc*100:.2f}%)**")
    st.divider()

    # 2. Test Loss
    st.markdown("### 2. Test Loss")
    st.markdown(f"**{loss:.4f}**")
    st.divider()

    # Predictions
    y_true = test_gen.classes
    y_pred = np.argmax(model.predict(test_gen, verbose=0), axis=1)

    # 3. Confusion Matrix
    st.markdown("### 3. Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)
    labels = list(test_gen.class_indices.keys())

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(f"Confusion Matrix - {model_choice}")
    st.pyplot(fig)
    st.divider()

    # 4. Classification Report
    st.markdown("### 4. Classification Report")

    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    df_report = df_report.round(3)
    df_report['support'] = df_report['support'].astype(int)

    styled_df = df_report.style \
        .background_gradient(cmap='coolwarm', subset=['precision', 'recall', 'f1-score']) \
        .format("{:.3f}", subset=['precision', 'recall', 'f1-score']) \
        .format("{:d}", subset=['support'])

    st.dataframe(styled_df, use_container_width=True)

    # Optional final recap
    st.markdown("---")
    st.markdown(f"**Final Overall Accuracy**: {acc:.4f} ({acc*100:.2f}%)")