import json
import numpy as np
import streamlit as st
import os
import urllib.request
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models

# ------------------ DOWNLOAD WEIGHTS ------------------
MODEL_PATH = "weights.weights.h5"

if not os.path.exists(MODEL_PATH):
    st.write("Downloading model weights...")
    url = "https://huggingface.co/sabatamboli/rose-disease-model1/resolve/main/weights.weights.h5"
    urllib.request.urlretrieve(url, MODEL_PATH)

# ------------------ BUILD MODEL ------------------
NUM_CLASSES = 8

model = models.Sequential([
    layers.Input(shape=(224,224,3)),

    layers.Conv2D(32,(3,3),activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(NUM_CLASSES,activation='softmax')
])

# ------------------ LOAD WEIGHTS ------------------
model.load_weights(MODEL_PATH)

# ------------------ LOAD CLASS LABELS ------------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# ------------------ CURE INFO ------------------
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

# ------------------ UI ------------------
st.set_page_config(page_title="Rose Disease Detector", page_icon="🌿")

st.markdown("<h1 style='text-align:center;'>🌿 Rose Disease Classifier</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# ------------------ PREPROCESS ------------------
def preprocess(image):
    image = image.convert("RGB")
    image = image.resize((224,224))
    img = np.array(image)/255.0
    img = np.expand_dims(img, axis=0)
    return img

# ------------------ PREDICTION ------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, width=250)

    if st.button("Classify"):
        img = preprocess(image)
        pred = model.predict(img)
        class_id = np.argmax(pred)

        disease = class_indices[str(class_id)]

        st.success(f"Prediction: {disease}")
        st.info(cure_info.get(disease, "No cure available"))