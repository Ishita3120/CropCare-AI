import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = "datasets/Tomato/TomatoCNN"
MODEL_PATH = "models/tomato_mobilenetv2.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# -----------------------------
# Load Test Dataset
# -----------------------------
test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, "->", name)

# -----------------------------
# Load Saved Model
# -----------------------------
print("\nLoading saved model...")

model = keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# -----------------------------
# Test Accuracy
# -----------------------------
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

print("\n==============================")
print("Tomato Test Accuracy:", test_accuracy)
print("==============================")

# -----------------------------
# Predictions
# -----------------------------
y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(images, verbose=0)

    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# Classification Report
# -----------------------------
print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)