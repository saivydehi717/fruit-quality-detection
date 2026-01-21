import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load and preprocess image
def preprocess_image(image_path, img_size=(150, 150)):
    """
    Loads an image from the given path, resizes to model requirement,
    converts to array and normalizes for prediction.
    Returns image as a batch of size 1.
    """
    try:
        img = load_img(image_path, target_size=img_size)
        img_array = img_to_array(img)
        img_array = img_array / 255.0   # Normalize to 0-1
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


# Convert predictions to label
def decode_prediction(pred_index):
    """
    Takes predicted class index from model
    Returns a readable fruit label.
    Update this as per your model.
    """
    labels = ["Fresh Apple", "Rotten Apple", "Fresh Banana", "Rotten Banana"]
    try:
        return labels[pred_index]
    except:
        return "Unknown Fruit"
