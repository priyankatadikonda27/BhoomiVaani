"""
Generates two synthetic 'scanned' land record document images for the demo:
  1. sample_docs/document_good.png   -> all fields present, clean scan
  2. sample_docs/document_bad.png    -> missing fields + slight noise (simulates a poor/old scan)

These stand in for real scanned land registers so the OCR -> extraction -> validation
-> confidence pipeline has guaranteed, repeatable input during the judging demo.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")
os.makedirs(OUT_DIR, exist_ok=True)

W, H = 1000, 700

def get_font(size=28):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()

def make_document(lines, path, noisy=False, skew=False):
    img = Image.new("RGB", (W, H), color=(255, 255, 250))
    draw = ImageDraw.Draw(img)

    title_font = get_font(34)
    body_font = get_font(26)

    draw.text((40, 30), "GOVERNMENT LAND RECORD - RECORD OF RIGHTS (ROR)", fill=(20, 20, 20), font=title_font)
    draw.line([(40, 80), (960, 80)], fill=(0, 0, 0), width=2)

    y = 130
    for line in lines:
        draw.text((60, y), line, fill=(10, 10, 10), font=body_font)
        y += 55

    draw.line([(40, y + 10), (960, y + 10)], fill=(0, 0, 0), width=1)
    draw.text((40, y + 30), "Office of the Tehsildar - Seal & Signature", fill=(60, 60, 60), font=get_font(20))

    if skew:
        img = img.rotate(1.2, expand=True, fillcolor=(255, 255, 250))

    if noisy:
        arr = np.array(img).astype(np.int16)
        noise = np.random.normal(0, 14, arr.shape).astype(np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        # simulate a faded/water-stained patch over part of the text
        overlay = Image.new("RGB", img.size, (255, 255, 250))
        odraw = ImageDraw.Draw(overlay)
        odraw.ellipse([550, 300, 950, 550], fill=(235, 225, 205))
        img = Image.blend(img, overlay, alpha=0.35)

    img.save(path)
    print("Saved:", path)


good_lines = [
    "Owner Name: Ramesh Kumar",
    "Survey No: 123/4",
    "Khata No: 5678",
    "Area: 2.35 acres",
    "Village: Pedakakani",
    "Tehsil: Guntur",
    "District: Guntur",
]

bad_lines = [
    "Owner Name: Ramesh Kumar",
    "Survey No: 123/4",
    "Area: 2.35 acres",
    "Village: Pedakakani",
    # Khata No, Tehsil, District intentionally omitted / illegible
]

make_document(good_lines, os.path.join(OUT_DIR, "document_good.png"), noisy=False, skew=False)
make_document(bad_lines, os.path.join(OUT_DIR, "document_bad.png"), noisy=True, skew=True)
