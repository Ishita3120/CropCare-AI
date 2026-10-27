from pathlib import Path
import shutil
import random

# Original downloaded dataset
SOURCE_DIR = Path("datasets/tomato1/train")

# New dataset for CNN
OUTPUT_DIR = Path("datasets/Tomato/TomatoCNN4000")

CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
]

# 80% train, 10% validation, 10% test
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10

random.seed(42)

for class_name in CLASSES:

    source_class = SOURCE_DIR / class_name

    images = [
        p for p in source_class.iterdir()
        if p.is_file()
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    valid_end = train_end + int(total * VALID_RATIO)

    train_images = images[:train_end]
    valid_images = images[train_end:valid_end]
    test_images = images[valid_end:]

    for split, split_images in [
        ("train", train_images),
        ("valid", valid_images),
        ("test", test_images),
    ]:

        output_class = OUTPUT_DIR / split / class_name
        output_class.mkdir(parents=True, exist_ok=True)

        for image_path in split_images:
            shutil.copy2(
                image_path,
                output_class / image_path.name
            )

    print(
        f"{class_name}: "
        f"train={len(train_images)}, "
        f"valid={len(valid_images)}, "
        f"test={len(test_images)}"
    )

print("\nDataset preparation completed!")
print(f"Output: {OUTPUT_DIR}")
