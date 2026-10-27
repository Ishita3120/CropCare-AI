import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import numpy as np

# =========================
# 1. Paths
# =========================

BASE_DIR = Path("datasets/Potato")

TRAIN_DIR = BASE_DIR / "Train"
VALID_DIR = BASE_DIR / "Valid"
TEST_DIR = BASE_DIR / "Test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 3
EPOCHS = 8

# =========================
# 2. Load datasets
# =========================

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

valid_ds = keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names

print("\nClasses:", class_names)

# =========================
# 3. Performance optimization
# =========================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# =========================
# 4. Data augmentation
# =========================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# =========================
# 5. MobileNetV2
# =========================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Initially freeze pretrained layers
base_model.trainable = False

# =========================
# 6. Build model
# =========================

inputs = keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.2)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = keras.Model(inputs, outputs)

# =========================
# 7. Compile
# =========================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# 8. Train
# =========================

print("\nStarting training...\n")

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS
)

# =========================
# 9. Test
# =========================

print("\nEvaluating on TEST dataset...\n")

test_loss, test_accuracy = model.evaluate(test_ds)

print(f"\nCNN Test Accuracy: {test_accuracy * 100:.2f}%")

# =========================
# 10. Confusion Matrix
# =========================

from sklearn.metrics import confusion_matrix, classification_report

y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)

    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

# =========================
# 11. Save model
# =========================

MODEL_PATH = "models/potato_mobilenetv2.keras"

model.save(MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")