
from pathlib import Path
import pickle
import shutil
import numpy as np
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "crop_classifier.pkl"

IMAGE_SIZE = (64, 64)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Use the crop-specific dataset folders.
DATASETS = {
    "train": {
        "Potato": BASE_DIR / "datasets" / "Potato" / "Train",
        "Tomato": BASE_DIR / "datasets" / "Tomato" / "TomatoCNN4000" / "train",
    },
    "valid": {
        "Potato": BASE_DIR / "datasets" / "Potato" / "Valid",
        "Tomato": BASE_DIR / "datasets" / "Tomato" / "TomatoCNN4000" / "valid",
    },
    "test": {
        "Potato": BASE_DIR / "datasets" / "Potato" / "Test",
        "Tomato": BASE_DIR / "datasets" / "Tomato" / "TomatoCNN4000" / "test",
    },
}


def extract_features(image_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize(IMAGE_SIZE)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array.flatten()


def load_split(split):
    features, labels = [], []

    for crop, folder in DATASETS[split].items():
        if not folder.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {folder}"
            )

        paths = [
            path for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        ]

        print(f"{split} - {crop}: {len(paths)} images")

        if not paths:
            raise ValueError(
                f"No images found for {crop} in {folder}"
            )

        for path in paths:
            try:
                features.append(extract_features(path))
                labels.append(crop)
            except Exception as exc:
                print(f"Skipping {path}: {exc}")

    if not features:
        raise ValueError(f"No usable images in {split} dataset.")

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)

    return shuffle(X, y, random_state=42)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading datasets...")
    X_train, y_train = load_split("train")
    X_valid, y_valid = load_split("valid")
    X_test, y_test = load_split("test")

    print("\nTraining SVM crop classifier...")
    classifier = make_pipeline(
        StandardScaler(),
        LinearSVC(class_weight="balanced", max_iter=5000, random_state=42)
    )
    classifier.fit(X_train, y_train)

    valid_pred = classifier.predict(X_valid)
    test_pred = classifier.predict(X_test)

    valid_accuracy = accuracy_score(y_valid, valid_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    print(f"\nValidation accuracy: {valid_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print("\nTest classification report:")
    print(classification_report(y_test, test_pred, zero_division=0))

    # Do not overwrite the old model unless the test result is usable.
    if test_accuracy < 0.70:
        print(
            "\nTest accuracy is below 70%. "
            "Existing crop_classifier.pkl has NOT been changed."
        )
        return

    if MODEL_PATH.exists():
        backup_path = MODEL_DIR / "crop_classifier_backup.pkl"
        if not backup_path.exists():
            shutil.copy2(MODEL_PATH, backup_path)
            print(f"Old model backed up to: {backup_path}")

    payload = {
        "classifier": classifier,
        "image_size": IMAGE_SIZE,
        "classes": ["Potato", "Tomato"],
        "method": "linear-svm-raw-pixels",
    }

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(payload, file)

    print(f"\nNew crop classifier saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
