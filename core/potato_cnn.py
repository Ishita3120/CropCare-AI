import tensorflow as tf
import numpy as np
from PIL import Image


MODEL_PATH = "models/potato_mobilenetv2.keras"

# Load model once when the application starts
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]


def predict_potato_disease(image_path):
    """
    Predict Potato disease using the trained MobileNetV2 model.

    Returns:
        disease_label: predicted class
        confidence: model confidence
    """

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Resize to MobileNetV2 input size
    image = image.resize((224, 224))

    # Convert to numpy array
    image_array = np.array(image, dtype=np.float32)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # MobileNetV2 preprocessing
   

    # Prediction
    predictions = model.predict(image_array, verbose=0)

    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index])

    disease_label = CLASS_NAMES[predicted_index]

    print("DEBUG CNN predicted index:", predicted_index)
    print("DEBUG CNN predicted label:", disease_label)
    print("DEBUG CNN confidence:", confidence)

    return disease_label, confidence
