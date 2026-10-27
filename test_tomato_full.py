import numpy as np
import tensorflow as tf
from PIL import Image
import os

MODEL_PATH = "models/tomato_full_mobilenetv2.keras"

model = tf.keras.models.load_model(MODEL_PATH)

classes = [
    "Black_Spot",
    "Early_Blight",
    "Healthy",
    "Late_Blight"
]
tests = [
    ("Late_Blight", "tomato_0070.jpg"),
    ("Late_Blight", "tomato_0128.jpg"),
    ("Late_Blight", "tomato_0160.jpg"),
    ("Late_Blight", "tomato_0281.jpg"),
    ("Late_Blight", "tomato_0326.jpg"),
    ("Late_Blight", "tomato_0331.jpg"),
    ("Late_Blight", "tomato_0340.jpg"),
    ("Late_Blight", "tomato_0392.jpg"),
    ("Late_Blight", "tomato_0479.jpg"),
    ("Late_Blight", "tomato_0482.jpg"),
    ("Late_Blight", "tomato_0571.jpg"),
    ("Late_Blight", "tomato_0575.jpg"),
    ("Late_Blight", "tomato_0652.jpg")
]

for actual, filename in tests:

    # Find image inside test folders
    image_path = None

    for folder in classes:
        possible = os.path.join(
            "datasets/Tomato/FullImageCNN/test",
            folder,
            filename
        )

        if os.path.exists(possible):
            image_path = possible
            break

    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    arr = np.asarray(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    prediction = model.predict(arr, verbose=0)[0]

    index = int(np.argmax(prediction))
    predicted = classes[index]
    confidence = float(prediction[index]) * 100

    print("--------------------------------")
    print("Actual     :", actual)
    print("Predicted  :", predicted)
    print("Confidence :", round(confidence, 2), "%")

print("--------------------------------")
