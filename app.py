import os
import numpy as np
import pickle
import tensorflow as tf
import streamlit as st
from PIL import Image
from io import BytesIO
import requests

st.set_page_config(layout="wide", page_title="Insurance", page_icon="🚓")

def download_file(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure we notice bad responses
        with open(filename, 'wb') as file:
            file.write(response.content)
    except requests.RequestException as e:
        st.error(f"Error downloading file: {e}")

def load_model():
    model_file = "model.pkl"
    if not os.path.isfile(model_file):
        download_url = "https://github.com/vishnudathan/insurance/raw/main/models/model.pkl"
        download_file(download_url, model_file)
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
    return model

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        try:
            image_data = uploaded_file.read()
            if not image_data:
                st.error("Uploaded file is empty")
                raise ValueError("Uploaded file is empty")
            img = Image.open(BytesIO(image_data)).convert("RGB")
            img = tf.image.convert_image_dtype(np.array(img), dtype=tf.float32)
            img = tf.image.resize(img, (256, 256))
            return np.expand_dims(img, 0)  # Add batch dimension
        except Exception as e:
            st.error(f"Error processing image: {e}")
            raise
    else:
        st.error("No file uploaded")
        raise FileNotFoundError("No file uploaded")

def predict_img(model, final_data):
    try:
        yhat = model.predict(final_data)
        return 'Glass is not broken. Estimated price = 0' if yhat[0] > 0.5 else 'Glass is broken. Estimated price = 5000'
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        raise

def main():
    st.header("Insurance Price Estimator")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "jfif"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", width=400)

    if st.button("Check here"):
        if uploaded_file is None:
            st.warning("Please upload an image.")
        else:
            final_data = input_image_setup(uploaded_file)
            prediction = predict_img(trained_model, final_data)
            st.subheader("Result")
            st.write(prediction)

if __name__ == '__main__':
    trained_model = load_model()
    main()
