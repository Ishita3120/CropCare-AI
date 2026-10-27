
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf

from PIL import Image

MODEL_PATH = Path("models/leaf_validator.keras")
IMG_SIZE = (224, 224)

if not MODEL_PATH.exists():
    sys.exit(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(image_path):
    image_path = Path(image_path)

    if not image_path.is_file():
        print(f"File not found: {image_path}")
        return

    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize(IMG_SIZE)

        array = np.asarray(image, dtype=np.float32)
        array = np.expand_dims(array, axis=0)

        probability = float(model.predict(array, verbose=0)[0][0])

        print(f"\nImage: {image_path.name}")
        print(f"Leaf probability: {(1 - probability) * 100:.2f}%")
        print(f"Non-leaf probability: {probability * 100:.2f}%")

        if probability >= 0.5:
            print("Prediction: NON-LEAF")
        else:
            print("Prediction: LEAF")

    except Exception as error:
        print(f"Could not process image: {error}")

if len(sys.argv) < 2:
    print(
        'Usage: python test_leaf_validator.py "path/to/image.jpg"'
    )
else:
    predict_image(sys.argv[1])
