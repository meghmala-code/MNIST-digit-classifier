import streamlit as st
import numpy as np
import joblib
import gdown
import os
from PIL import Image, ImageOps

# --------------------------
# Download model if not found
# --------------------------
MODEL_PATH = "svm_model.pkl"


FILE_ID = "1G3l_a-QAbWyRi3GaYLGsgWGe8JoZ2asV"
URL = f"https://drive.google.com/file/d/1G3l_a-QAbWyRi3GaYLGsgWGe8JoZ2asV/view?usp=sharing={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(URL, MODEL_PATH, quiet=False)

# Load the model
model = joblib.load(MODEL_PATH)

# --------------------------
# Streamlit App
# --------------------------
st.title("✏️ MNIST Digit Classifier")
st.write("Upload a 28×28 grayscale image of a handwritten digit (0–9)")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

def preprocess_image(image):
    """Convert to 28×28 grayscale and flatten to 1D"""
    image = image.convert("L")
    image = image.resize((28, 28))
    image_array = np.array(image)
    flat = image_array.reshape(1, -1)
    return flat

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Digit", use_container_width=False, width=150)

    processed = preprocess_image(image)

    try:
        prediction = model.predict(processed)
        st.success(f"🧠 Predicted Digit: **{prediction[0]}**")
    except Exception as e:
        st.error(f"Error while predicting: {e}")
