"""
Generates a more realistic-looking 'scanned' land record document —
table-format Record of Rights (ROR) with a government header, official
seal, signature line, and aged/scanned paper texture — for use in the
BhoomiVaani OCR demo.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUT_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")
os.makedirs(OUT_DIR, exist_ok=True)

W, H = 1240, 1600  # portrait, like a scanned A4 page


def font(size, bold=False):
    path = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    return ImageFont.truetype(path, size) if os.path.exists(path) else ImageFont.load_default()


def aged_paper(w, h):
    """Off-white paper with subtle grain and vignette to mimic a scan."""
    base = np.full((h, w, 3), 247, dtype=np.uint8)
    noise = np.random.normal(0, 4, (h, w, 3))
    base = np.clip(base + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(base)

    # subtle vignette
    vignette = Image.new("L", (w, h), 0)
    vd = ImageDraw.Draw(vignette)
    vd.ellipse([-w * 0.3, -h * 0.3, w * 1.3, h * 1.3], fill=255)
    vignette = vignette.filter(ImageFilter.GaussianBlur(120))
    dark_overlay = Image.new("RGB", (w, h), (225, 220, 205))
    img = Image.composite(img, dark_overlay, vignette)
    return img


def draw_seal(draw, cx, cy, r=70):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(120, 20, 20), width=3)
    draw.ellipse([cx - r + 8, cy - r + 8, cx + r - 8, cy + r - 8], outline=(120, 20, 20), width=1)
    f = font(13, bold=True)
    draw.text((cx, cy - 30), "GOVT. OF", fill=(120, 20, 20), font=f, anchor="mm")
    draw.text((cx, cy - 10), "ANDHRA PRADESH", fill=(120, 20, 20), font=f, anchor="mm")
    draw.text((cx, cy + 10), "REVENUE DEPT.", fill=(120, 20, 20), font=f, anchor="mm")
    draw.text((cx, cy + 30), "TEHSIL OFFICE", fill=(120, 20, 20), font=f, anchor="mm")


def make_realistic_record(path):
    img = aged_paper(W, H)
    draw = ImageDraw.Draw(img)

    margin = 60
    draw.rectangle([margin, margin, W - margin, H - margin], outline=(40, 40, 40), width=3)
    draw.rectangle([margin + 8, margin + 8, W - margin - 8, H - margin - 8], outline=(40, 40, 40), width=1)

    y = margin + 30
    draw.text((W / 2, y), "GOVERNMENT OF ANDHRA PRADESH", fill=(20, 20, 20), font=font(30, True), anchor="mm")
    y += 40
    draw.text((W / 2, y), "REVENUE DEPARTMENT", fill=(20, 20, 20), font=font(22, True), anchor="mm")
    y += 34
    draw.text((W / 2, y), "RECORD OF RIGHTS, TENANCY AND CROPS (ROR)", fill=(20, 20, 20), font=font(24, True), anchor="mm")
    y += 30
    draw.line([(margin + 20, y), (W - margin - 20, y)], fill=(20, 20, 20), width=2)
    y += 20
    draw.text((W / 2, y), "( PAHANI / RECORD OF RIGHTS EXTRACT )", fill=(60, 60, 60), font=font(16), anchor="mm")
    y += 40

    # Reference numbers
    draw.text((margin + 30, y), "ROR No: AP/GNT/2026/00417", fill=(20, 20, 20), font=font(18))
    draw.text((W - margin - 30, y), "Date: 03-09-2026", fill=(20, 20, 20), font=font(18), anchor="ra")
    y += 50

    # ---- Field table ----
    rows = [
        ("Owner Name", "Ramesh Kumar S/o Venkaiah"),
        ("Survey No", "123/4"),
        ("Khata No", "5678"),
        ("Area", "2.35 acres"),
        ("Village", "Pedakakani"),
        ("Tehsil", "Guntur"),
        ("District", "Guntur"),
        ("Classification", "Dry Land - Patta"),
        ("Nature of Right", "Owner Possession"),
    ]

    table_left = margin + 30
    table_right = W - margin - 30
    col_split = table_left + 320
    row_h = 56
    table_top = y

    for i, (label, value) in enumerate(rows):
        ry = table_top + i * row_h
        draw.rectangle([table_left, ry, table_right, ry + row_h], outline=(80, 80, 80), width=1)
        draw.line([(col_split, ry), (col_split, ry + row_h)], fill=(80, 80, 80), width=1)
        draw.text((table_left + 15, ry + row_h / 2), label, fill=(20, 20, 20), font=font(19, True), anchor="lm")
        draw.text((col_split + 15, ry + row_h / 2), value, fill=(20, 20, 20), font=font(19), anchor="lm")

    y = table_top + len(rows) * row_h + 50

    draw.text((margin + 30, y), "Remarks:", fill=(20, 20, 20), font=font(18, True))
    y += 30
    draw.text((margin + 30, y), "No encumbrance recorded as on date of issue. Mutation entries updated.",
               fill=(50, 50, 50), font=font(16))
    y += 70

    # Signature / seal block
    draw_seal(draw, W - margin - 130, y + 60)
    draw.line([(margin + 30, y + 120), (margin + 330, y + 120)], fill=(20, 20, 20), width=1)
    draw.text((margin + 30, y + 128), "Signature of Tehsildar", fill=(20, 20, 20), font=font(16))
    draw.text((margin + 30, y + 6), "Issued by:", fill=(20, 20, 20), font=font(16, True))
    draw.text((margin + 30, y + 34), "Office of the Tehsildar, Guntur Mandal", fill=(50, 50, 50), font=font(16))

    y += 170
    draw.line([(margin + 20, y), (W - margin - 20, y)], fill=(20, 20, 20), width=1)
    draw.text((W / 2, H - margin - 25), "This is a computer-generated demo document for prototype testing purposes only.",
               fill=(120, 120, 120), font=font(14), anchor="mm")

    # slight scan artifacts: rotation + blur + jpeg-like noise
    img = img.rotate(0.6, expand=True, fillcolor=(247, 244, 235))
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    arr = np.array(img).astype(np.int16)
    noise = np.random.normal(0, 6, arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    img.save(path, quality=92)
    print("Saved:", path)


make_realistic_record(os.path.join(OUT_DIR, "sample_real_land_record.png"))
