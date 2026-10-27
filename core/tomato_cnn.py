from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "tomato_full_mobilenetv2.keras"
)


# ============================================================
# Load trained model
# ============================================================

model = tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# Class names
# IMPORTANT: This order must match training output
# ============================================================

CLASS_NAMES = [
    "Bacterial_Spot",
    "Early_Blight",
    "Late_Blight",
    "Healthy"
]


# ============================================================
# Label mapping used by the application
# ============================================================

LABEL_MAP = {
    "Bacterial_Spot": "bacterial_spot",
    "Early_Blight": "early_blight",
    "Late_Blight": "late_blight",
    "Healthy": "healthy",
}


# ============================================================
# Prediction function
# ============================================================

def predict_tomato_disease(image_path):

    image = (
        Image.open(image_path)
        .convert("RGB")
        .resize((224, 224))
    )

    arr = np.asarray(
        image,
        dtype=np.float32
    )

    arr = np.expand_dims(
        arr,
        axis=0
    )

    # IMPORTANT:
    # MobileNetV2 preprocessing is already INSIDE
    # the trained model.
    predictions = model.predict(
        arr,
        verbose=0
    )[0]

    idx = int(
        np.argmax(predictions)
    )

    raw_class = CLASS_NAMES[idx]

    confidence = float(
        predictions[idx]
    )

    label = LABEL_MAP[raw_class]

    return {
        "label": label,
        "confidence": confidence,
        "method": "full-image-mobilenetv2-cnn",
        "raw_class": raw_class,
        "model_file": "tomato_full_mobilenetv2.keras",
    }