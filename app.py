import io
import numpy as np
from PIL import Image, ImageOps
import streamlit as st

from predict import (
    load_model,
    preprocess_grayscale_to_rgb48,
    preprocess_grayscale_to_1channel,
    predict,
    EMOTIONS
)

# Sidebar: model selection
st.sidebar.title("Settings")
model_choice = st.sidebar.selectbox(
    "Select model",
    ["MobileNetV2 (transfer)", "EfficientNetB0 (transfer)", "Baseline CNN"]
)
model_path = {
    "MobileNetV2 (transfer)": "emotion_mobilenet.keras",
    "EfficientNetB0 (transfer)": "emotion_efficientnet.keras",
    "Baseline CNN": "emotion_model.keras",
}[model_choice]

# Load model (cache to avoid reloading every interaction)
@st.cache_resource
def get_model(path):
    return load_model(path)

model = get_model(model_path)

st.title("Emotion Recognition Demo")
st.write("Upload a face image. The system predicts the emotion using your trained models.")

uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

def to_grayscale_48(image: Image.Image) -> np.ndarray:
    """
    Convert PIL image to 48x48 grayscale numpy array (uint8).
    """
    image = image.convert("L")
    image = ImageOps.fit(image, (48, 48), method=Image.Resampling.LANCZOS)
    return np.array(image)

if uploaded is not None:
    # Read image with PIL
    img_bytes = uploaded.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Show original image
    st.image(pil_img, caption="Uploaded image", use_column_width=True)

    try:
        # Preprocess: grayscale 48x48
        gray48 = to_grayscale_48(pil_img)

        # Choose preprocessing based on model
        if model_choice == "Baseline CNN":
            input_tensor = preprocess_grayscale_to_1channel(gray48)
        else:
            input_tensor = preprocess_grayscale_to_rgb48(gray48)

        # Predict
        probs, top_label, top_idx = predict(model, input_tensor)

        # Display result
        st.subheader(f"Predicted emotion: {top_label}")
        st.bar_chart({label: float(probs[i]) for i, label in enumerate(EMOTIONS)})

        with st.expander("Show preprocessing preview"):
            st.image(Image.fromarray(gray48), caption="Grayscale 48x48", width=200)
            st.write(f"Input tensor shape: {input_tensor.shape}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.markdown("---")
st.caption("Tip: Use close-up face images for more reliable predictions.")
