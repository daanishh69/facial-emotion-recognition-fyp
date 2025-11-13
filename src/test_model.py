# src/test_model.py
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# -----------------------------
# Load Model
# -----------------------------
model = load_model(r"models\emotion_model.keras")

# -----------------------------
# Test a single image
# -----------------------------
# Replace 'img_123.png' with an actual image from your test folder
img_path = r"C:\Users\Danish Omar\Desktop\SEM5\FYPPROJECT\data\fer2013\archive\test\happy\PrivateTest_928647.jpg"  

img = image.load_img(img_path, target_size=(48,48), color_mode="grayscale")
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
print("Predicted emotion:", emotions[predicted_class])
