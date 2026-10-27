# """
# core/image_analyzer.py

# IMPORTANT, READ FIRST:
# This module performs REAL image analysis (it actually inspects the uploaded
# photo's pixels), but it is a color-based computer-vision HEURISTIC, not a
# trained deep learning model. It is a deliberately honest placeholder so the
# web app works end to end today, with the exact seam where you plug in a real
# trained classifier (EfficientNet/YOLO/etc.) marked below.

# How the heuristic works:
# 1. Excludes background pixels (low-saturation areas -- table, sky, hand, etc.)
#    to isolate leaf material.
# 2. Within leaf pixels, separates "healthy green" hue range from
#    "diseased" hues (brown/yellow/near-black lesions).
# 3. affected_area_ratio = diseased leaf pixels / total leaf pixels.
# 4. A simple rule maps color signature (hue + brightness) + selected crop to a
#    plausible disease label, matching the crop/disease pairs in data/treatments.json.

# Replace `analyze_image` with a call to your trained model once you have one --
# keep the same return shape (see AnalysisResult) so nothing downstream changes.
# """
# from __future__ import annotations

# from dataclasses import dataclass
# import numpy as np
# from PIL import Image


# @dataclass
# class AnalysisResult:
#     disease_label: str          # 'healthy' | 'early_blight' | 'late_blight' | 'leaf_rust'
#     confidence: float           # 0-1, pseudo-confidence derived from signal strength
#     affected_area_ratio: float  # 0-1
#     leaf_area_ratio: float      # fraction of the frame identified as leaf (QC signal)
#     method: str = "cv-heuristic-v1"  # bump this string once a real model replaces the heuristic


# # Disease labels the knowledge base actually has entries for, per crop.
# _CROP_DISEASE_OPTIONS = {
#     "tomato": ["early_blight", "late_blight"],
#     "potato": ["early_blight", "late_blight"],
#     "wheat": ["leaf_rust"],
# }


# def analyze_image(image: Image.Image, crop: str) -> AnalysisResult:
#     """
#     TODO(real model): replace the body of this function with inference from a
#     trained CNN/ViT classifier + segmentation model. Keep returning an
#     AnalysisResult with the same fields so app.py needs no changes.
#     """
#     img = image.convert("RGB")
#     img.thumbnail((320, 320))
#     hsv = np.array(img.convert("HSV"), dtype=np.float32)
#     h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
#     hue_deg = h / 255.0 * 360.0

#     # 1. Isolate leaf material from background (low-saturation / near-white / near-black areas)
#     leaf_mask = (s > 35) & (v > 25) & (v < 250)
#     leaf_pixels = int(leaf_mask.sum())
#     total_pixels = h.size
#     leaf_area_ratio = round(leaf_pixels / total_pixels, 4) if total_pixels else 0.0

#     if leaf_pixels < 200:
#         # Not enough leaf material detected to say anything meaningful.
#         return AnalysisResult(
#             disease_label="healthy",
#             confidence=0.5,
#             affected_area_ratio=0.0,
#             leaf_area_ratio=leaf_area_ratio,
#         )

#     # 2. Healthy green range vs. diseased (brown/yellow/dark) range, within leaf pixels
#     healthy_green = leaf_mask & (hue_deg >= 65) & (hue_deg <= 170) & (v > 60)
#     diseased = leaf_mask & ~healthy_green

#     diseased_count = int(diseased.sum())
#     affected_area_ratio = round(diseased_count / leaf_pixels, 4)

#     if affected_area_ratio < 0.03:
#         confidence = round(min(0.97, 0.78 + (1 - affected_area_ratio) * 0.2), 3)
#         return AnalysisResult(
#             disease_label="healthy",
#             confidence=confidence,
#             affected_area_ratio=affected_area_ratio,
#             leaf_area_ratio=leaf_area_ratio,
#         )

#     # 3. Characterize the diseased pixels to guess a label (brightness -> lesion darkness)
#     diseased_v = v[diseased]
#     diseased_hue = hue_deg[diseased]
#     avg_v = float(np.mean(diseased_v)) if diseased_v.size else 128.0
#     avg_hue = float(np.mean(diseased_hue)) if diseased_hue.size else 40.0

#     options = _CROP_DISEASE_OPTIONS.get(crop, ["early_blight"])
#     if len(options) == 1:
#         label = options[0]
#     else:
#         # dark, low-brightness lesions read as late blight; lighter brown/tan as early blight
#         label = "late_blight" if avg_v < 95 else "early_blight"
#         if label not in options:
#             label = options[0]

#     signal_strength = min(affected_area_ratio / 0.35, 1.0)  # saturates by 35% coverage
#     confidence = round(min(0.95, 0.6 + signal_strength * 0.32), 3)

#     return AnalysisResult(
#         disease_label=label,
#         confidence=confidence,
#         affected_area_ratio=affected_area_ratio,
#         leaf_area_ratio=leaf_area_ratio,
#     )

"""
core/image_analyzer.py

Image analysis using a lightweight HSV-based computer-vision heuristic.

This module performs two tasks:
1. Basic crop/leaf image quality validation.
2. Disease/affected-area estimation for valid leaf images.

This is NOT a deep-learning disease model.
The trained Potato disease model is integrated separately in app.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from PIL import Image


@dataclass
class AnalysisResult:
    disease_label: str
    confidence: float
    affected_area_ratio: float
    leaf_area_ratio: float

    # New validation fields
    status: str = "ok"
    green_area_ratio: float = 0.0

    method: str = "cv-heuristic-v2"


_CROP_DISEASE_OPTIONS = {
    "tomato": ["early_blight", "late_blight"],
    "potato": ["early_blight", "late_blight"],
    "wheat": ["leaf_rust"],
    "cucumber": ["early_blight", "late_blight"],
}


def analyze_image(image: Image.Image, crop: str) -> AnalysisResult:

    img = image.convert("RGB")
    img.thumbnail((320, 320))

    hsv = np.array(img.convert("HSV"), dtype=np.float32)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    hue_deg = h / 255.0 * 360.0

    total_pixels = h.size

    if total_pixels == 0:
        return AnalysisResult(
            disease_label="not_crop",
            confidence=0.0,
            affected_area_ratio=0.0,
            leaf_area_ratio=0.0,
            status="not_crop",
            green_area_ratio=0.0,
        )

    # ---------------------------------------------------------
    # 1. Detect pixels that could belong to plant/leaf material
    # ---------------------------------------------------------

    leaf_mask = (
        (s > 35)
        & (v > 25)
        & (v < 250)
    )

    leaf_pixels = int(leaf_mask.sum())

    leaf_area_ratio = round(
        leaf_pixels / total_pixels,
        4
    )

    # ---------------------------------------------------------
    # 2. Detect green plant-like pixels
    # ---------------------------------------------------------

    green_mask = (
        (hue_deg >= 45)
        & (hue_deg <= 180)
        & (s > 40)
        & (v > 35)
    )

    green_pixels = int(green_mask.sum())

    green_area_ratio = round(
        green_pixels / total_pixels,
        4
    )

    # ---------------------------------------------------------
    # 3. Basic crop/leaf validation
    # ---------------------------------------------------------

    # If almost no saturated/plant-like material exists,
    # this is very unlikely to be a leaf image.

    if leaf_area_ratio < 0.05:
        return AnalysisResult(
            disease_label="not_crop",
            confidence=0.0,
            affected_area_ratio=0.0,
            leaf_area_ratio=leaf_area_ratio,
            status="not_crop",
            green_area_ratio=green_area_ratio,
        )

    # We also require some meaningful green plant area.
    # This prevents many random objects/images from being
    # accepted as crops.

    if green_area_ratio < 0.02:
        return AnalysisResult(
            disease_label="not_crop",
            confidence=0.0,
            affected_area_ratio=0.0,
            leaf_area_ratio=leaf_area_ratio,
            status="not_crop",
            green_area_ratio=green_area_ratio,
        )

    # ---------------------------------------------------------
    # 4. Calculate healthy green pixels within leaf region
    # ---------------------------------------------------------

    healthy_green = (
        leaf_mask
        & (hue_deg >= 65)
        & (hue_deg <= 170)
        & (v > 60)
    )

    healthy_green_pixels = int(
        healthy_green.sum()
    )

    # If there is enough leaf-like material but almost no
    # healthy/plant green signal, we still allow the image
    # because a heavily diseased leaf can lose its green color.

    # ---------------------------------------------------------
    # 5. Disease affected-area calculation
    # ---------------------------------------------------------

    diseased = leaf_mask & ~healthy_green

    diseased_count = int(diseased.sum())

    affected_area_ratio = round(
        diseased_count / leaf_pixels,
        4
    ) if leaf_pixels else 0.0

    # ---------------------------------------------------------
    # 6. Healthy leaf
    # ---------------------------------------------------------

    if affected_area_ratio < 0.03:

        confidence = round(
            min(
                0.97,
                0.78 + (1 - affected_area_ratio) * 0.2
            ),
            3
        )

        return AnalysisResult(
            disease_label="healthy",
            confidence=confidence,
            affected_area_ratio=affected_area_ratio,
            leaf_area_ratio=leaf_area_ratio,
            status="ok",
            green_area_ratio=green_area_ratio,
        )

    # ---------------------------------------------------------
    # 7. Characterize diseased pixels
    # ---------------------------------------------------------

    diseased_v = v[diseased]
    diseased_hue = hue_deg[diseased]

    avg_v = (
        float(np.mean(diseased_v))
        if diseased_v.size
        else 128.0
    )

    avg_hue = (
        float(np.mean(diseased_hue))
        if diseased_hue.size
        else 40.0
    )

    # ---------------------------------------------------------
    # 8. Disease label heuristic
    # ---------------------------------------------------------

    options = _CROP_DISEASE_OPTIONS.get(
        crop,
        ["early_blight"]
    )

    if len(options) == 1:

        label = options[0]

    else:

        # Dark lesions → late blight
        # Lighter brown/tan lesions → early blight

        label = (
            "late_blight"
            if avg_v < 95
            else "early_blight"
        )

        if label not in options:
            label = options[0]

    # ---------------------------------------------------------
    # 9. Pseudo-confidence
    # ---------------------------------------------------------

    signal_strength = min(
        affected_area_ratio / 0.35,
        1.0
    )

    confidence = round(
        min(
            0.95,
            0.6 + signal_strength * 0.32
        ),
        3
    )

    # ---------------------------------------------------------
    # 10. Final valid analysis
    # ---------------------------------------------------------

    return AnalysisResult(
        disease_label=label,
        confidence=confidence,
        affected_area_ratio=affected_area_ratio,
        leaf_area_ratio=leaf_area_ratio,
        status="ok",
        green_area_ratio=green_area_ratio,
    )
