import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load model
model = tf.keras.models.load_model("fruit_quality_model.h5")

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

st.title("🍎 Fruit Quality Live Detection")
st.write("Use webcam to detect fruit quality")

camera = st.camera_input("Take a picture")

if camera:
    image = Image.open(camera)
    img_array = np.array(image)

    img = cv2.resize(img_array, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    index = np.argmax(prediction)
    confidence = float(prediction[0][index]) * 100
    label = labels[index]

    st.image(image, caption="Captured Image")
    st.success(f"Prediction: **{label}**")
    st.info(f"Confidence: **{confidence:.2f}%**")
