
from pathlib import Path
import random
import shutil

random.seed(42)

base = Path("datasets/LeafValidation")
source = base / "train" / "non_leaf"
train_dir = base / "train" / "non_leaf"
val_dir = base / "val" / "non_leaf"

val_dir.mkdir(parents=True, exist_ok=True)

# Keep 20% aside for validation
images = [
    p for p in source.iterdir()
    if p.is_file() and p.suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".webp"
    }
]

random.shuffle(images)
val_count = max(1, int(len(images) * 0.2))

for image in images[:val_count]:
    destination = val_dir / image.name
    if image.resolve() != destination.resolve():
        shutil.move(str(image), str(destination))

print("Non-leaf images remaining in train:", len(list(train_dir.iterdir())))
print("Non-leaf images in val:", len(list(val_dir.iterdir())))
print("Dataset folders prepared.")
