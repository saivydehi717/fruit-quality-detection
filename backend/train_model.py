import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    Flatten, Dense, Dropout
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ===============================
# PATH SETUP (IMPORTANT)
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_dir = os.path.join(BASE_DIR, "dataset", "train")
val_dir = os.path.join(BASE_DIR, "dataset", "val")

print("Train path:", train_dir)
print("Validation path:", val_dir)

# ===============================
# IMAGE PARAMETERS
# ===============================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# ===============================
# DATA GENERATORS
# ===============================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

print("Classes:", train_generator.class_indices)

# ===============================
# MODEL ARCHITECTURE
# ===============================
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(3, activation="softmax")   # 3 classes
])

# ===============================
# COMPILE MODEL
# ===============================
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================
# TRAIN MODEL
# ===============================
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# ===============================
# SAVE MODEL
# ===============================
model_path = os.path.join(BASE_DIR, "backend", "fruit_quality_model.h5")
model.save(model_path)

print("✅ Model saved at:", model_path)
