from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import base64
import csv
from datetime import datetime

import os

app = Flask(__name__, template_folder="templates", static_folder="static")


# Load model
model = tf.keras.models.load_model("fruit_quality_model.h5")

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# ---------------- HOME ROUTE ----------------
@app.route("/")
def home():
    return render_template("camera.html")

# ---------------- PREDICT ROUTE ----------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["image"]

    image_data = base64.b64decode(data.split(",")[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (150, 150))   # MUST MATCH TRAINING
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    index = np.argmax(preds)
    confidence = float(preds[0][index]) * 100
    label = labels[index]

    # 📁 SAVE TO CSV
    with open("predictions.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            label,
            round(confidence, 2)
        ])

    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 2)
    })

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

