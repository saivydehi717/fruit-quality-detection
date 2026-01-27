import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import csv
from datetime import datetime
import os

# Page Config
st.set_page_config(page_title="Fruit AI", layout="wide")

# Load model
model = tf.keras.models.load_model("fruit_quality_model.h5")

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# UI Design
st.markdown("""
<style>
body { background-color: #fff8f0; }
h1 { color: orange; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🍎 Fruit Quality Live Detection</h1>", unsafe_allow_html=True)
st.write("Real-time AI fruit freshness detector")

run = st.checkbox("🎥 Start Live Camera")

FRAME_WINDOW = st.image([])

CSV_FILE = "live_predictions.csv"

# Create CSV if not exists
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Date & Time", "Prediction", "Confidence"])

camera = cv2.VideoCapture(0)

while run:
    success, frame = camera.read()
    if not success:
        st.warning("Camera not detected")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = cv2.resize(frame_rgb, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    index = np.argmax(prediction)
    confidence = float(prediction[0][index]) * 100
    label = labels[index]

    # Save prediction
    with open(CSV_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, round(confidence, 2)])

    # Display label
    cv2.putText(frame_rgb, f"{label} ({confidence:.2f}%)",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 3)

    FRAME_WINDOW.image(frame_rgb)

camera.release()
