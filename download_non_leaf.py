
import os
import time
import requests
from io import BytesIO
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(
    BASE_DIR, "datasets", "LeafValidation", "train", "non_leaf"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

API = "https://commons.wikimedia.org/w/api.php"
session = requests.Session()
session.headers.update({
    "User-Agent": "CropDiseaseStudentProject/1.0 (educational use)"
})

# Search broad non-leaf categories. Images still need manual checking.
SEARCH_TERMS = [
    "person portrait photograph",
    "family people photograph",
    "building street photograph",
    "car vehicle photograph",
    "food plate photograph",
    "computer laptop photograph",
    "furniture chair photograph",
    "indoor room photograph",
]

TARGET = 150
MAX_BYTES = 2 * 1024 * 1024  # Maximum 2 MB per image
saved = 0
seen = set()

for term in SEARCH_TERMS:
    if saved >= TARGET:
        break

    print(f"\nSearching: {term}")

    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {term}",
        "gsrnamespace": 6,
        "gsrlimit": 30,
        "prop": "imageinfo",
        "iiprop": "url|mime|size",
        "iiurlwidth": 320,
    }

    try:
        response = session.get(API, params=params, timeout=30)
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})

        for page in pages.values():
            if saved >= TARGET:
                break

            info_list = page.get("imageinfo", [])
            if not info_list:
                continue

            info = info_list[0]
            url = info.get("thumburl") or info.get("url")
            mime = info.get("thumbmime") or info.get("mime", "")

            if not url or not mime.startswith("image/") or url in seen:
                continue
            seen.add(url)

            try:
                img_response = session.get(url, timeout=20)
                img_response.raise_for_status()

                if len(img_response.content) > MAX_BYTES:
                    continue

                image = Image.open(BytesIO(img_response.content)).convert("RGB")

                if image.width < 100 or image.height < 100:
                    continue

                filename = os.path.join(
                    OUTPUT_DIR, f"non_leaf_{saved + 1:04d}.jpg"
                )
                image.save(filename, "JPEG", quality=85, optimize=True)
                saved += 1
                print(f"Saved {saved}/{TARGET}: {page.get('title', 'image')}")

            except Exception as exc:
                print("Skipped image:", str(exc)[:100])

            time.sleep(0.2)

    except Exception as exc:
        print("Search failed:", exc)

print(f"\nFinished. Downloaded {saved} images.")
print("Folder:", OUTPUT_DIR)
print("Please manually inspect images before using them for training.")
