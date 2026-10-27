import os
import glob
import pickle
import numpy as np
from PIL import Image


IMAGE_SIZE = (64, 64)

MODEL_PATH = "models/crop_classifier.pkl"


def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    array = np.array(image, dtype=np.float32) / 255.0

    return array.flatten()


def predict(image_path, model):

    features = extract_features(image_path)

    centroids = model["centroids"]

    distances = {}

    for crop, centroid in centroids.items():
        distance = np.linalg.norm(features - centroid)
        distances[crop] = distance

    predicted_crop = min(
        distances,
        key=distances.get
    )

    return predicted_crop, distances


# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


print("Crop classifier loaded.")
print("Classes:", model["classes"])


# Test Potato images
potato_folder = "datasets/Potato/Test"
potato_images = []

for extension in ["*.jpg", "*.jpeg", "*.png"]:
    potato_images.extend(
        glob.glob(
            os.path.join(potato_folder, "**", extension),
            recursive=True
        )
    )


print("\n--- POTATO TEST ---")

for image_path in potato_images[:5]:

    prediction, distances = predict(image_path, model)

    print(
        os.path.basename(image_path),
        "->",
        prediction,
        "| distances:",
        {k: round(v, 2) for k, v in distances.items()}
    )


# Test Tomato images
tomato_folder = "datasets/Tomato/CropDataset/test/Tomato"
tomato_images = []

for extension in ["*.jpg", "*.jpeg", "*.png"]:
    tomato_images.extend(
        glob.glob(
            os.path.join(tomato_folder, "**", extension),
            recursive=True
        )
    )


print("\n--- TOMATO TEST ---")

for image_path in tomato_images[:5]:

    prediction, distances = predict(image_path, model)

    print(
        os.path.basename(image_path),
        "->",
        prediction,
        "| distances:",
        {k: round(v, 2) for k, v in distances.items()}
    )