import os
import tensorflow as tf
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fruit_quality_model.h5")

print("Loading model from:", MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

CLASSES = ["Fresh", "Medium", "Rotten"]

def predict_fruit(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    confidence = float(np.max(predictions))

    return CLASSES[class_index], confidence
