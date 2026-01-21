import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fruit_quality_model.h5")

model = tf.keras.models.load_model(MODEL_PATH)

# ---------------- LABELS ----------------
CLASSES = ["Fresh", "Medium", "Rotten"]

# ---------------- PREDICTION FUNCTION ----------------
def predict_fruit(image):
    if image is None:
        return "No image", "0%"

    # Convert PIL to OpenCV
    img = np.array(image)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    index = np.argmax(preds)
    confidence = preds[0][index] * 100

    return CLASSES[index], f"{confidence:.2f}%"

# ---------------- GRADIO UI ----------------
app = gr.Interface(
    fn=predict_fruit,
    inputs=gr.Image(
        sources=["webcam", "upload"],
        type="pil",
        label="📷 Capture or Upload Fruit Image"
    ),
    outputs=[
        gr.Label(label="🍎 Fruit Quality"),
        gr.Textbox(label="📊 Confidence")
    ],
    title="🍍 Fruit Quality Detection using AI",
    description="Detect whether fruit is **Fresh, Medium, or Rotten** using Deep Learning",
    theme="soft",
)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.launch()
