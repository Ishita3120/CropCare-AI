
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# CONFIGURATION
# =========================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "crop_classifier_cnn.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 8
SEED = 42

# Label mapping:
# Potato = 0
# Tomato = 1

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

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================
# LOAD IMAGE PATHS
# =========================

def collect_images(split):
    paths = []
    labels = []

    for crop, folder in DATASETS[split].items():
        if not folder.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {folder}"
            )

        files = sorted(
            p for p in folder.rglob("*")
            if p.is_file()
            and p.suffix.lower() in EXTENSIONS
        )

        print(f"{split} - {crop}: {len(files)} images")

        if not files:
            raise ValueError(f"No images found in {folder}")

        label = 0 if crop == "Potato" else 1

        paths.extend(str(p) for p in files)
        labels.extend([label] * len(files))

    return np.array(paths), np.array(labels, dtype=np.float32)


# =========================
# IMAGE PREPROCESSING
# =========================

def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    def load_image(path, label):
        image_bytes = tf.io.read_file(path)

        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False
        )

        image.set_shape([None, None, 3])

        image = tf.image.resize(image, IMAGE_SIZE)

        image = keras.applications.mobilenet_v2.preprocess_input(
            image
        )

        return image, label

    ds = ds.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# =========================
# BUILD CNN MODEL
# =========================

def build_model():
    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    # Initially freeze pretrained layers
    base_model.trainable = False

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))

    features = base_model(inputs, training=False)

    features = layers.GlobalAveragePooling2D()(features)
    features = layers.Dropout(0.3)(features)

    # Binary classification: Potato or Tomato
    outputs = layers.Dense(
        1,
        activation="sigmoid"
    )(features)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =========================
# TRAIN AND EVALUATE
# =========================

def main():
    print("TensorFlow version:", tf.__version__)
    print("\nLoading dataset paths...")

    train_paths, train_labels = collect_images("train")
    valid_paths, valid_labels = collect_images("valid")
    test_paths, test_labels = collect_images("test")

    if len(np.unique(train_labels)) != 2:
        raise ValueError(
            "Training dataset must contain Potato and Tomato."
        )

    train_ds = make_dataset(
        train_paths, train_labels, training=True
    )

    valid_ds = make_dataset(
        valid_paths, valid_labels
    )

    test_ds = make_dataset(
        test_paths, test_labels
    )

    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),

        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
    ]

    print("\nStarting crop classification CNN training...")

    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("\nLoading best saved model...")

    model = keras.models.load_model(MODEL_PATH)

    print("\nEvaluating on test dataset...")

    loss, accuracy = model.evaluate(test_ds, verbose=1)

    print(f"\nTest accuracy: {accuracy * 100:.2f}%")

    print("\nGenerating predictions...")

    probabilities = model.predict(test_ds, verbose=1).ravel()

    predictions = (probabilities >= 0.5).astype(int)

    print("\nClass mapping: Potato = 0, Tomato = 1")

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            test_labels.astype(int),
            predictions,
            labels=[0, 1]
        )
    )

    print("\nClassification report:")

    print(
        classification_report(
            test_labels.astype(int),
            predictions,
            labels=[0, 1],
            target_names=["Potato", "Tomato"],
            zero_division=0
        )
    )

    print("\nCNN model saved successfully:")
    print(MODEL_PATH)


if __name__ == "__main__":
    main()
