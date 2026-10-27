
from pathlib import Path
import random
import shutil

ROOT = Path("datasets")
OUTPUT = ROOT / "LeafValidation"

SOURCES = [
    ROOT / "Potato" / "Train",
    ROOT / "Tomato" / "TomatoCNN4000" / "train",
]

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
random.seed(42)

images = []
for source in SOURCES:
    if not source.exists():
        print("Folder not found:", source)
        continue

    for folder in source.iterdir():
        if folder.is_dir():
            for img in folder.rglob("*"):
                if img.is_file() and img.suffix.lower() in EXTENSIONS:
                    images.append(img)

random.shuffle(images)
split = int(len(images) * 0.8)

for part, batch in [
    ("train", images[:split]),
    ("val", images[split:]),
]:
    dest = OUTPUT / part / "leaf"
    dest.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(batch):
        shutil.copy2(img, dest / f"leaf_{i:05d}{img.suffix.lower()}")

print("Total leaf images found:", len(images))
print("Training leaf images:", split)
print("Validation leaf images:", len(images) - split)
print("Copied leaf images to:", OUTPUT)
