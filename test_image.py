import sys
import numpy as np
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model("model/fruit_model.h5")

def predict(img_path):
    img = Image.open(img_path).resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, 0)

    pred = model.predict(img)[0][0]
    print(f"Prediction Score: {pred}")

    if pred > 0.5:
        print("🍏 Fresh Fruit")
    else:
        print("🍂 Rotten Fruit")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_image.py path/to/image")
    else:
        predict(sys.argv[1])
