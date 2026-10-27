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


def predict_crop(image_path, model):
    features = extract_features(image_path)

    distances = {}

    for crop, centroid in model["centroids"].items():
        distances[crop] = np.linalg.norm(features - centroid)

    detected_crop = min(
        distances,
        key=distances.get
    )

    return detected_crop, distances


# Load crop classifier
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# Pick one Tomato test image
tomato_folder = "datasets/Tomato/CropDataset/test/Tomato"

tomato_images = glob.glob(
    os.path.join(tomato_folder, "*.jpg")
)

test_image = tomato_images[0]

# Pretend user selected Potato
selected_crop = "Potato"

detected_crop, distances = predict_crop(
    test_image,
    model
)


print("Image:", test_image)
print("Selected crop:", selected_crop)
print("Detected crop:", detected_crop)

print(
    "Distances:",
    {k: round(float(v), 2) for k, v in distances.items()}
)


if detected_crop != selected_crop:
    print("\n❌ CROP MISMATCH")
    print("Analysis should be rejected.")
else:
    print("\n✅ CROP MATCH")
    print("Disease analysis can continue.")