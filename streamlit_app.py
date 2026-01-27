import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import csv
from datetime import datetime
import os

# Load model
model = tf.keras.models.load_model("fruit_quality_model.h5")

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

st.set_page_config(page_title="Fruit Quality Detection", layout="centered")

st.title("🍎 Fruit Quality Detection App")
st.write("AI-powered fruit quality prediction using Image Upload or Camera")

option = st.radio("Choose Input Method", ["Upload Image", "Use Camera"])

CSV_FILE = "predictions.csv"

# Create CSV file if not exists
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Date & Time", "Prediction", "Confidence (%)"])

def save_prediction(label, confidence):
    with open(CSV_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            label,
            round(confidence, 2)
        ])

def predict_image(image):
    img_array = np.array(image)
    img = cv2.resize(img_array, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    index = np.argmax(prediction)
    confidence = float(prediction[0][index]) * 100
    label = labels[index]

    st.success(f"Prediction: **{label}**")
    st.info(f"Confidence: **{confidence:.2f}%**")

    save_prediction(label, confidence)

# -------- Upload Mode --------
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload Fruit Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")
        predict_image(image)

# -------- Camera Mode --------
if option == "Use Camera":
    camera = st.camera_input("Capture Fruit Image")

    if camera:
        image = Image.open(camera)
        st.image(image, caption="Captured Image")
        predict_image(image)

# -------- Show Prediction History --------
st.subheader("📜 Prediction History")

if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "r") as file:
        st.text(file.read())
