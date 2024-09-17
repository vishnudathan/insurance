# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 18:37:53 2024

@author: Nithin
"""
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.models as keras_models
import streamlit as st
import requests
from PIL import Image

st.set_page_config(layout="wide", page_title="insurance", page_icon="🚓")

def download_file(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as file:
        file.write(response.content)

def load_model():
    model = "image_classifier_models.h5"
    if not os.path.isfile(model):
        download_url = "https://github.com/vishnudathan/insurance/raw/main/models/insurance.h5"
        download_file(download_url, model)
    return keras_models.load_model(model)

trained_model = load_model()

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue() 
        img = tf.image.decode_image(image_data, channels=3) 
        resize = tf.image.resize(img, (256, 256))
        final_data = np.expand_dims(resize / 255, 0)
        return final_data
    else:
        raise FileNotFoundError("No file uploaded")
        
def predict_img(final_data):
    yhat = trained_model.predict(final_data)
    if yhat > 0.5: 
        prediction = 'Glass is not broken estimated price = 0'
    else:
        prediction ='Glass is borken estimated price = 5000'
    return prediction

def main():
    st.header("insurance price")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image = ""   

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", width = 400)

    submit=st.button("Check here")
    if submit:
        if uploaded_file is None:
            st.warning("Upload!!!")
        else:
            final_data = input_image_setup(uploaded_file)
            prediction = predict_img(final_data)
            st.subheader("result")
            st.write(prediction)

if __name__ == '__main__':
    main()