import os
import numpy as np
import streamlit as st
from PIL import Image

from predict import (
    load_model,
    preprocess_grayscale_to_rgb48,
    preprocess_grayscale_to_1channel,
    predict,
)

# -----------------------------
# ✅ Page config (must be first Streamlit command)
# -----------------------------
st.set_page_config(page_title="Facial Emotion Recognition", page_icon="😎", layout="centered")

# -----------------------------
# Cyberpunk / Anime UI Styling
# -----------------------------
st.markdown(
    """
    <style>
    /* Global background and font */
    body {
        background-color: #0d0d0d;
        color: #00ffcc;
        font-family: 'Orbitron', sans-serif;
    }

    /* Sidebar styling */
    .stSidebar {
        background-color: #1a1a1a;
        color: #00ffcc;
    }

    /* Buttons */
    .stButton>button {
        background-color: #ff00ff;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #00ffcc;
        color: black;
    }

    /* Headers */
    h1, h2, h3 {
        color: #ff00ff;
        text-shadow: 0px 0px 8px #00ffcc;
    }

    /* Sidebar title */
    .css-1d391kg {
        color: #ff00ff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# UI: Title and intro
# -----------------------------
st.title("Facial Emotion Recognition Demo")
st.write("Select a model, upload an image or use your webcam, and see the predicted emotion.")

# -----------------------------
# Sidebar: Model selection
# -----------------------------
model_choice = st.sidebar.selectbox(
    "Choose a model",
    ["MobileNetV2 (transfer)", "EfficientNetB0 (transfer)", "Baseline CNN"],
    index=0,
)

# Set model paths. Adjust names to match your actual files.
MODEL_PATHS = {
    "MobileNetV2 (transfer)": "models/emotion_mobilenet.keras",
    "EfficientNetB0 (transfer)": "models/emotion_efficientnet.keras",
    "Baseline CNN": "models/emotion_model.keras",
}

selected_model_path = MODEL_PATHS[model_choice]

# -----------------------------
# Cache: Model loader
# -----------------------------
@st.cache_resource(show_spinner=True)
def get_model(path: str):
    return load_model(path)

# Try to load model, handle missing file gracefully
model = None
if not os.path.exists(selected_model_path):
    st.warning(
        f"Model file not found: {selected_model_path}\n\n"
        "Place the correct .keras file in the 'models/' folder OR update MODEL_PATHS accordingly.\n"
        "For now, you can still explore the UI, but predictions will be disabled."
    )
else:
    try:
        model = get_model(selected_model_path)
        st.success(f"Loaded model: {model_choice}")
    except Exception as e:
        st.error(f"Failed to load model from {selected_model_path}.\n\nError: {e}")
        st.stop()

# -----------------------------
# Inputs: File upload or webcam
# -----------------------------
st.subheader("Input image")
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader(
        "Upload a face image (will be converted to 48×48 grayscale)",
        type=["png", "jpg", "jpeg"],
        help="Clear, front-facing face works best."
    )
with col2:
    camera_image = st.camera_input("Or capture from webcam")

# -----------------------------
# Helper: Prepare image
# -----------------------------
def prepare_image_to_48x48_grayscale(img: Image.Image) -> np.ndarray:
    # Convert to grayscale and resize to 48x48
    img_gray = img.convert("L").resize((48, 48))
    return np.array(img_gray)

# -----------------------------
# Predict flow
# -----------------------------
if uploaded_file or camera_image:
    if model is None:
        st.error("Model not loaded. Please add the correct .keras file and reload the app.")
        st.stop()

    # Load the input image
    pil_img = Image.open(uploaded_file or camera_image)
    img_array = prepare_image_to_48x48_grayscale(pil_img)

    # Choose preprocessing based on model type
    if model_choice == "Baseline CNN":
        input_tensor = preprocess_grayscale_to_1channel(img_array)
    else:
        input_tensor = preprocess_grayscale_to_rgb48(img_array)

    # Run prediction
    probs, label, idx = predict(model, input_tensor)

    # -------------------------
    # Display results
    # -------------------------
    st.subheader("Result")
    c_img, c_pred = st.columns([1, 1])

    with c_img:
        st.image(pil_img, caption="Input image", use_column_width=True)

    with c_pred:
        st.markdown(f"**Predicted emotion:** {label}")
        st.bar_chart(probs)

    # Top-3 breakdown for clarity
    st.subheader("Top-3 probabilities")
    EMOTIONS = ['angry','disgust','fear','happy','neutral','sad','surprise']
    top_indices = np.argsort(probs)[::-1][:3]
    for rank, i in enumerate(top_indices, 1):
        st.write(f"{rank}. {EMOTIONS[i]} — {probs[i]:.3f}")

else:
    st.info("Upload an image or use the webcam to run a prediction.")

# -----------------------------
# Sidebar: About section
# -----------------------------
st.sidebar.markdown("## About this demo")
st.sidebar.info(
    "This app demonstrates **Facial Emotion Recognition** using three models:\n\n"
    "- MobileNetV2 (transfer learning)\n"
    "- EfficientNetB0 (transfer learning)\n"
    "- Baseline CNN\n\n"
    "Upload a face image or use your webcam to see predictions.\n\n"
    "Cyberpunk UI inspired by anime aesthetics ✨"
)
