import json
import numpy as np
import streamlit as st
import tensorflow as tf
import gdown
import os
from PIL import Image


MODEL_PATH = "model.keras"

if not os.path.exists(MODEL_PATH):
    url = ("https://drive.google.com/uc?id=1tkxy4bDMTY2im9wlae6aTqebzDIEePdo&confirm=t")  # replace your ID
    gdown.download(url, MODEL_PATH, quiet=False)
# Load model

model = tf.keras.models.load_model(MODEL_PATH)
# Load class labels
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

cure_info = {
    "Black_Spot": "Remove infected leaves and Spray Fungicide such as Mancozeb or Chlorothalonil.",
    "Downy_Mildew": "Improve air circulation and apply Copper-Based Fungicide.",
    "Gray_Mold": "Remove infected buds, reduce humidity, and use Botrytis Fungicide.",
    "Rose_Mosaic": "No chemical cure available. Remove infected plants and use virus-free stock.",
    "Rose_Rust": "Remove affected leaves and spray Sulfur or Copper fungicide.",
    "Powdery_Mildew": "Prune affected parts, improve air circulation and apply Sulfur or Potassium Bicarbonate Fungicide.",
    "Healthy_Leaf": "Plant is healthy. Maintain proper watering, sunlight and regular monitoring.",
    "Healthy_Flower":"Plant is healthy. Maintain proper watering, sunlight and regular monitoring."
}

# Page config
st.set_page_config(
    page_title="Rose Plant Disease Classifier",
    page_icon="🌿",
    layout="centered"
)

# Clean CSS
st.markdown("""
<style>
html, body, [class*="stApp"] {
    background-color: #e8f5e9 !important;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>🌿 Rose Plant Disease Classifier</h1>", unsafe_allow_html=True)

st.write("Upload an image...")

# Upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="file1")

# Preprocess
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, width=200)

    with col2:
        if st.button("Classify"):
            img = preprocess_image(image)
            pred = model.predict(img)
            class_id = np.argmax(pred)
            disease = class_indices[str(class_id)]

            st.success(f"Prediction: {disease}")
            st.info(cure_info.get(disease, "No cure available"))