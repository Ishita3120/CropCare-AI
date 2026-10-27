# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import confusion_matrix, classification_report


# # ============================================================
# # Paths
# # ============================================================

# DATA_DIR = "datasets/Tomato/TomatoCNN4000"
# MODEL_PATH = "models/tomato_full_mobilenetv2.keras"

# IMG_SIZE = (224, 224)
# BATCH_SIZE = 16

# INITIAL_EPOCHS = 10
# FINE_TUNE_EPOCHS = 8


# # ============================================================
# # Load datasets
# # ============================================================

# train_ds = tf.keras.utils.image_dataset_from_directory(
#     os.path.join(DATA_DIR, "train"),
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     shuffle=True
# )

# valid_ds = tf.keras.utils.image_dataset_from_directory(
#     os.path.join(DATA_DIR, "valid"),
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     shuffle=False
# )

# test_ds = tf.keras.utils.image_dataset_from_directory(
#     os.path.join(DATA_DIR, "test"),
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     shuffle=False
# )


# # ============================================================
# # Class names
# # ============================================================

# class_names = train_ds.class_names

# print("\nClasses:")

# for i, name in enumerate(class_names):
#     print(i, "->", name)


# # ============================================================
# # Class weights
# # ============================================================

# y_train = []

# for _, labels in train_ds:
#     y_train.extend(labels.numpy())

# y_train = np.array(y_train)

# weights = compute_class_weight(
#     class_weight="balanced",
#     classes=np.unique(y_train),
#     y=y_train
# )

# class_weights = {
#     i: float(weight)
#     for i, weight in enumerate(weights)
# }

# print("\nClass weights:")
# print(class_weights)


# # ============================================================
# # Performance
# # ============================================================

# AUTOTUNE = tf.data.AUTOTUNE

# train_ds = train_ds.prefetch(AUTOTUNE)
# valid_ds = valid_ds.prefetch(AUTOTUNE)
# test_ds = test_ds.prefetch(AUTOTUNE)


# # ============================================================
# # Data augmentation
# # ============================================================

# data_augmentation = keras.Sequential([
#     layers.RandomFlip("horizontal"),
#     layers.RandomRotation(0.1),
#     layers.RandomZoom(0.1),
# ])


# # ============================================================
# # MobileNetV2
# # ============================================================

# base_model = tf.keras.applications.MobileNetV2(
#     input_shape=(224, 224, 3),
#     include_top=False,
#     weights="imagenet"
# )

# base_model.trainable = False


# # ============================================================
# # Model
# # ============================================================

# inputs = keras.Input(shape=(224, 224, 3))

# x = data_augmentation(inputs)

# # Preprocessing INSIDE model
# x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

# x = base_model(x, training=False)

# x = layers.GlobalAveragePooling2D()(x)

# x = layers.Dropout(0.3)(x)

# outputs = layers.Dense(
#     len(class_names),
#     activation="softmax"
# )(x)

# model = keras.Model(inputs, outputs)


# # ============================================================
# # Initial compile
# # ============================================================

# model.compile(
#     optimizer=keras.optimizers.Adam(
#         learning_rate=0.001
#     ),
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )


# model.summary()


# # ============================================================
# # Initial training
# # ============================================================

# print("\n========================================")
# print("Starting Initial Training")
# print("========================================")

# history = model.fit(
#     train_ds,
#     validation_data=valid_ds,
#     epochs=INITIAL_EPOCHS,
#     class_weight=class_weights
# )


# # ============================================================
# # Fine-tuning
# # ============================================================

# print("\n========================================")
# print("Starting Fine-Tuning")
# print("========================================")

# base_model.trainable = True

# # Freeze all except last 30 layers
# for layer in base_model.layers[:-30]:
#     layer.trainable = False

# # Keep BatchNormalization frozen
# for layer in base_model.layers:
#     if isinstance(layer, layers.BatchNormalization):
#         layer.trainable = False


# trainable_count = sum(
#     1 for layer in base_model.layers
#     if layer.trainable
# )

# print("Trainable MobileNetV2 layers:", trainable_count)


# # ============================================================
# # Recompile
# # ============================================================

# model.compile(
#     optimizer=keras.optimizers.Adam(
#         learning_rate=0.00001
#     ),
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )


# # ============================================================
# # Fine-tune
# # ============================================================

# fine_tune_history = model.fit(
#     train_ds,
#     validation_data=valid_ds,
#     epochs=FINE_TUNE_EPOCHS,
#     class_weight=class_weights
# )


# # ============================================================
# # Final test
# # ============================================================

# print("\n========================================")
# print("Final Test Evaluation")
# print("========================================")

# test_loss, test_accuracy = model.evaluate(
#     test_ds,
#     verbose=1
# )

# print("\n========================================")
# print("Tomato Full-Image Test Accuracy:",
#       test_accuracy)
# print("========================================")


# # ============================================================
# # Predictions
# # ============================================================

# y_true = []
# y_pred = []

# for images, labels in test_ds:

#     predictions = model.predict(
#         images,
#         verbose=0
#     )

#     predicted_classes = np.argmax(
#         predictions,
#         axis=1
#     )

#     y_true.extend(labels.numpy())
#     y_pred.extend(predicted_classes)


# y_true = np.array(y_true)
# y_pred = np.array(y_pred)


# # ============================================================
# # Confusion Matrix
# # ============================================================

# cm = confusion_matrix(
#     y_true,
#     y_pred
# )

# print("\nConfusion Matrix:")
# print(cm)


# # ============================================================
# # Classification Report
# # ============================================================

# print("\nClassification Report:")

# print(
#     classification_report(
#         y_true,
#         y_pred,
#         target_names=class_names,
#         zero_division=0
#     )
# )


# # ============================================================
# # Save model
# # ============================================================

# os.makedirs(
#     "models",
#     exist_ok=True
# )

# model.save(MODEL_PATH)

# print("\n========================================")
# print("Final model saved to:")
# print(MODEL_PATH)
# print("========================================")



import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
# Paths
# ============================================================

DATA_DIR = "datasets/Tomato/TomatoCNN4000"
MODEL_PATH = "models/tomato_full_mobilenetv2.keras"
BEST_MODEL_PATH = "models/tomato_best.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 8


# ============================================================
# Load datasets
# ============================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "valid"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# Class names
# ============================================================

class_names = train_ds.class_names

print("\nClasses:")

for i, name in enumerate(class_names):
    print(i, "->", name)


# ============================================================
# Class weights
# ============================================================

y_train = []

for _, labels in train_ds:
    y_train.extend(labels.numpy())

y_train = np.array(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(weights)
}

print("\nClass weights:")
print(class_weights)


# ============================================================
# Performance
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ============================================================
# Data augmentation
# ============================================================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])


# ============================================================
# MobileNetV2
# ============================================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


# ============================================================
# Model
# ============================================================

inputs = keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

# Preprocessing INSIDE model
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = keras.Model(inputs, outputs)


# ============================================================
# Initial compile
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

os.makedirs("models", exist_ok=True)

checkpoint = ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    mode="max",
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# Initial training
# ============================================================

print("\n========================================")
print("Starting Initial Training")
print("========================================")

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)


# ============================================================
# Fine-tuning
# ============================================================

print("\n========================================")
print("Starting Fine-Tuning")
print("========================================")

base_model.trainable = True

# Freeze all except last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization frozen
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False


trainable_count = sum(
    1 for layer in base_model.layers
    if layer.trainable
)

print(
    "Trainable MobileNetV2 layers:",
    trainable_count
)


# ============================================================
# Recompile
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# Fine-tune
# ============================================================

fine_tune_history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)


# ============================================================
# Load BEST MODEL
# ============================================================

print("\n========================================")
print("Loading Best Validation Model")
print("========================================")

model = tf.keras.models.load_model(
    BEST_MODEL_PATH
)

print("Best model loaded from:")
print(BEST_MODEL_PATH)


# ============================================================
# Final test
# ============================================================

print("\n========================================")
print("Final Test Evaluation")
print("========================================")

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print("\n========================================")
print(
    "Tomato Full-Image Test Accuracy:",
    test_accuracy
)
print("========================================")


# ============================================================
# Predictions
# ============================================================

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# Classification Report
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# Save FINAL model
# ============================================================

model.save(MODEL_PATH)

print("\n========================================")
print("Final model saved to:")
print(MODEL_PATH)
print("========================================")