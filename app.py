"""
BhoomiVaani — AI-Powered Land Record Digitization & Validation Platform
SIH 2026 | Problem Statement 26018 | Team Catalyst

2-hour hackathon prototype demonstrating the core pipeline as a step-by-step flow:
Upload -> Processing (OCR + AI extraction happens silently in the background)
       -> Confidence & Human Verification -> Record Log

Note on language support: Tesseract's Hindi ('hin') and Telugu ('tel') language
packs are installed alongside English ('eng'). The document-language selector on
the Upload step controls which OCR language model is used. Mixing all languages
at once ("eng+hin+tel") on an English-only document was tested and found to
reduce accuracy (Tesseract starts misreading Latin table lines as Telugu/Hindi
glyphs), so the prototype asks the user (or an operator) to pick the dominant
script instead of always guessing all three. Field *labels* now have Telugu
alternatives too (e.g. "సర్వే నంబరు" as well as "Survey No"), so a native-Telugu
document — labels and values both in Telugu script — is extracted correctly, not
just documents with English labels and Telugu names/values. Hindi labels are not
yet covered the same way; only English and Telugu label patterns are defined
below, so a Hindi-labelled document still needs English labels to extract
correctly (its OCR text will read fine, it's the field regexes that are
English/Telugu-only for now).
"""

import os
import re
import time
import unicodedata
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import pytesseract

# --------------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="BhoomiVaani",
    page_icon="🌱",
    layout="wide",
)

BASE_DIR = os.path.dirname(__file__)
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_docs")
LOG_PATH = os.path.join(BASE_DIR, "verified_records.csv")

REQUIRED_FIELDS = ["Owner", "Survey No", "Khata No", "Area", "Village", "Tehsil", "District"]

FIELD_PATTERNS = {
    # Separator ([:\-]) is optional so this matches both "Label: value" (line-format
    # documents) and "Label   value" (table/column-format documents, common in real
    # scanned ROR extracts where OCR doesn't preserve a colon between cells).
    # \b word boundaries after each label stop false matches inside longer words
    # (e.g. "Tehsil" inside "Tehsildar") now that the separator itself is optional.
    #
    # Each field now has a list of patterns tried in order — an English one and
    # a Telugu one — so a document with Telugu field labels (not just Telugu
    # names/values under English labels) still gets extracted. \w and \b are
    # Unicode-aware in Python 3, so they work the same way against Telugu script.
    "Owner":     [r"Owner\b\s*(?:Name)?\s*[:\-]?\s*(.+)",
                  r"(?:యజమాని|పట్టాదారు(?:ని)?|రైతు)\s*(?:పేరు)?\s*[:\-]?\s*(.+)"],
    "Survey No": [r"Survey\s*No\b\.?\s*[:\-]?\s*([\w\/\-]+)",
                  r"సర్వే\s*నంబ(?:రు|ర్)\s*[:\-]?\s*([\w\/\-]+)"],
    "Khata No":  [r"Khata\s*No\b\.?\s*[:\-]?\s*([\w\/\-]+)",
                  r"ఖాతా\s*నంబ(?:రు|ర్)\s*[:\-]?\s*([\w\/\-]+)"],
    "Area":      [r"Area\b\s*[:\-]?\s*([\d.]+\s*\w*)",
                  r"విస్తీర్ణం\s*[:\-]?\s*([\d.]+\s*\w*)"],
    "Village":   [r"Village\b\s*[:\-]?\s*(.+)",
                  r"గ్రామం\s*[:\-]?\s*(.+)"],
    "Tehsil":    [r"Tehsil\b\s*[:\-]?\s*(.+)",
                  r"మండలం\s*[:\-]?\s*(.+)"],
    "District":  [r"District\b\s*[:\-]?\s*(.+)",
                  r"జిల్లా\s*[:\-]?\s*(.+)"],
}

STEPS = ["Upload", "Processing", "Verify", "Record Log"]

LANGUAGE_OPTIONS = {
    "English": "eng",
    "Hindi (हिन्दी)": "hin",
    "Telugu (తెలుగు)": "tel",
    "Mixed / Auto (English + Hindi + Telugu)": "eng+hin+tel",
}


# --------------------------------------------------------------------------------------
# CORE PIPELINE FUNCTIONS (run silently — not shown directly in the UI)
# --------------------------------------------------------------------------------------
def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 arbitrary corner points as [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_document_contour(image_bgr: np.ndarray):
    """Look for a large 4-cornered contour (the page) in a photographed document.
    Returns the 4 corner points, or None if nothing convincing is found — callers
    should fall back to using the original image untouched."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    edged = cv2.dilate(edged, None, iterations=1)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # Require the candidate to be a quadrilateral covering a large share of the
        # frame — otherwise it's more likely a stamp/table cell than the page edge.
        if len(approx) == 4 and cv2.contourArea(approx) > 0.2 * image_area:
            return approx.reshape(4, 2).astype("float32")
    return None


def warp_to_document(image_bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Perspective-warp a photographed document to a flat top-down view, like a
    scanner app would, given its 4 detected corners."""
    rect = _order_corners(corners)
    (tl, tr, br, bl) = rect
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 10 or height < 10:
        return image_bgr  # degenerate quad; not safe to warp
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image_bgr, matrix, (width, height))


def deskew(gray: np.ndarray) -> np.ndarray:
    """Rotate the image so text lines run horizontally, based on the minimum-area
    bounding box of the dark (text/ink) pixels. Skips rotation when there isn't
    enough foreground to estimate an angle safely, or when it's already straight."""
    inverted = cv2.bitwise_not(gray)
    _, bw = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw > 0))
    if coords.shape[0] < 20:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.5:
        return gray
    (h, w) = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess_and_ocr(pil_image: Image.Image, lang: str = "eng") -> tuple[str, Image.Image]:
    """Clean up the scan, then run Tesseract OCR.

    The original version only did grayscale + blur + a single global Otsu
    threshold, which is a reasonable start for a clean flatbed scan but breaks
    down for the phone photos most users will actually upload: pages shot at an
    angle, tilted text, low resolution, and uneven lighting/shadows all cause
    Tesseract to misread or drop text. This pipeline handles those cases, with
    every added stage wrapped so a failed/absent detection just falls back to
    passing the image through unchanged rather than crashing the run.
    """
    img_rgb = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # 1. Flatten perspective if the page was photographed at an angle rather
    #    than scanned flat.
    try:
        corners = find_document_contour(img_bgr)
        if corners is not None:
            img_bgr = warp_to_document(img_bgr, corners)
    except cv2.error:
        pass  # keep the original framing if edge detection misbehaves

    # 2. Upscale small images — OCR accuracy drops sharply below roughly
    #    150-200 DPI equivalent, and phone photos are often shrunk on upload.
    h, w = img_bgr.shape[:2]
    if w < 1500:
        scale = 1500 / w
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # 3. Grayscale + denoise (phone camera sensor noise / JPEG artifacts).
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # 4. Straighten tilted text.
    try:
        gray = deskew(gray)
    except cv2.error:
        pass

    # 5. Normalize uneven lighting/shadows before binarizing, rather than
    #    relying on a single global Otsu threshold across the whole page.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 6. Binarize.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 7. OCR. psm 6 ("assume a single uniform block of text") suits the dense
    #    field-per-line/table layout of land records better than the default
    #    fully-automatic segmentation (psm 3), which tends to fragment tables
    #    into disconnected blocks and drop lines.
    config = "--oem 3 --psm 6"
    try:
        text = pytesseract.image_to_string(thresh, lang=lang, config=config)
    except pytesseract.TesseractError:
        # Language pack not installed on this machine — fall back to English.
        text = pytesseract.image_to_string(thresh, lang="eng", config=config)
    return text, Image.fromarray(thresh)


def extract_fields(text: str) -> dict:
    """Rule-based extraction of the 7 required land-record fields from OCR text.
    Each field has an English and a Telugu label pattern; the first one that
    matches wins, so this works on English-labelled and Telugu-labelled
    documents alike."""
    data = {}
    for field, patterns in FIELD_PATTERNS.items():
        value = ""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                break
        data[field] = value
    return data


# Format checks applied only where we have a reliable expected shape. Anything
# not listed here is judged purely by _looks_like_garbage (below) since free-text
# fields like Owner/Village/Tehsil/District don't have one fixed format.
FIELD_FORMAT_CHECKS = {
    "Survey No": (r"^[\w]+(?:[\/\-][\w]+)*$", "expected e.g. 123/4"),
    "Khata No":  (r"^[\w]+(?:[\/\-][\w]+)*$", "expected e.g. 12 or 12/3"),
    "Area":      (r"\d", "expected a number, e.g. 2.5 Acres"),
}


def _looks_like_garbage(value: str) -> bool:
    """Heuristic OCR-noise detector: flags values that are too short to be real,
    or where more than 40% of the (non-space) characters are punctuation/symbol
    junk (stray @#~%$| characters, repeated punctuation, box-drawing artifacts)
    — a common signature of a misread region rather than a genuine short value.

    Deliberately checks Unicode category rather than str.isalnum(): Telugu (and
    Hindi) spelling normally includes combining vowel signs and virama marks
    (Unicode category "Mn"/"Mc") that isalnum() does NOT count as alphanumeric,
    which would otherwise make correctly-read Telugu text look "garbled" just
    for being in that script — the opposite of what this check is for.
    """
    stripped = value.strip()
    if len(stripped) < 2:
        return True
    non_space = [ch for ch in stripped if not ch.isspace()]
    if not non_space:
        return True
    # L* = letter, N* = number, M* = combining mark (accents/vowel signs/virama)
    junk = sum(1 for ch in non_space if unicodedata.category(ch)[0] not in ("L", "N", "M"))
    return (junk / len(non_space)) > 0.4


def _field_problem(field: str, value: str) -> str:
    """Returns a human-readable problem description for this field's value, or
    "" if it looks fine. Used by both validate_record (to show the message) and
    confidence_score (to decide whether the field counts toward confidence) —
    so a field that drags down confidence always has a matching explanation."""
    if not value.strip():
        return f"{field} missing"
    if _looks_like_garbage(value):
        return f'{field} looks garbled, likely an OCR misread: "{value}"'
    check = FIELD_FORMAT_CHECKS.get(field)
    if check:
        pattern, hint = check
        if not re.search(pattern, value):
            return f"{field} format looks unusual ({hint})"
    return ""


def validate_record(data: dict) -> list[str]:
    """Rule-based validation engine — flags missing, garbled, or oddly-formatted
    fields (see _field_problem)."""
    return [problem for field in REQUIRED_FIELDS
            if (problem := _field_problem(field, data.get(field, "")))]


def confidence_score(data: dict) -> float:
    """Confidence = share of required fields that are both present AND pass the
    same content-quality check validate_record uses — not just 'is this field
    non-empty'. A field filled with garbled OCR output (symbols, a one-character
    fragment, a Survey No with no digits) now counts against confidence instead
    of padding it the way a merely-blank field would have before."""
    if not data:
        return 0.0
    good = sum(1 for field in REQUIRED_FIELDS if not _field_problem(field, data.get(field, "")))
    return round((good / len(REQUIRED_FIELDS)) * 100, 2)


def confidence_bucket(score: float) -> tuple[str, str]:
    if score >= 80:
        return "🟢 High Confidence — Auto Verification", "success"
    elif score >= 60:
        return "🟡 Medium Confidence — Review Recommended", "warning"
    else:
        return "🔴 Low Confidence — Human Review Required", "error"


def log_verified_record(data: dict, confidence: float, status: str):
    row = {**data, "Confidence": confidence, "Status": status,
           "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    df_row = pd.DataFrame([row])
    if os.path.exists(LOG_PATH):
        df_row.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(LOG_PATH, mode="w", header=True, index=False)


# --------------------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------------------
defaults = {
    "step": 0,               # index into STEPS
    "image": None,
    "source_label": None,
    "ocr_lang_name": "English",
    "processed": False,      # whether OCR + extraction has run for the current image
    "preprocessed_image": None,  # cleaned-up (perspective-corrected, deskewed, thresholded) scan
    "fields": None,
    "errors": None,
    "confidence": None,
    "final_status": None,
    "final_data": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def go_to(step_index: int):
    st.session_state.step = step_index


def reset_flow():
    for key, value in defaults.items():
        st.session_state[key] = value


# --------------------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌱 BhoomiVaani")
    st.caption("SIH 2026 · PS 26018 · Team Catalyst")
    st.markdown("---")
    st.markdown("### 🔄 Pipeline")
    st.markdown(
        """
        1. **Upload** land record
        2. **Processing** (AI reads & extracts fields)
        3. **Confidence & Human Verification**
        4. **Record Log**
        """
    )
    st.markdown("---")
    if st.button("🔁 Start Over", use_container_width=True):
        reset_flow()
        st.rerun()

# --------------------------------------------------------------------------------------
# HEADER + STEP TRACKER
# --------------------------------------------------------------------------------------
st.title("🌱 BhoomiVaani")
st.subheader("AI-Powered Land Record Digitization & Validation")

step_cols = st.columns(len(STEPS))
for i, (col, name) in enumerate(zip(step_cols, STEPS)):
    marker = "🟢" if i < st.session_state.step else ("🔵" if i == st.session_state.step else "⚪")
    col.markdown(f"**{marker} {i + 1}. {name}**")

st.divider()

current = st.session_state.step

# ========================================================================================
# STEP 0 — UPLOAD
# ========================================================================================
if current == 0:
    st.markdown("### Step 1 — Upload the Land Record")
    st.write("Upload a scanned image of a land record, or load one of the demo samples below.")

    st.session_state.ocr_lang_name = st.selectbox(
        "Document language",
        options=list(LANGUAGE_OPTIONS.keys()),
        index=list(LANGUAGE_OPTIONS.keys()).index(st.session_state.ocr_lang_name),
        help="Choose the language the document is mostly written in, for the most accurate reading.",
    )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        uploaded_file = st.file_uploader(
            "Upload Land Record (JPG / PNG)", type=["jpg", "jpeg", "png"]
        )
    with col_b:
        st.markdown("**Or use a demo sample:**")
        use_good = st.button("📄 Load Clean Sample Record", use_container_width=True)
        use_bad = st.button("📄 Load Damaged/Incomplete Sample", use_container_width=True)

    if uploaded_file is not None:
        st.session_state.image = Image.open(uploaded_file)
        st.session_state.source_label = uploaded_file.name
        st.session_state.processed = False
    elif use_good:
        st.session_state.image = Image.open(os.path.join(SAMPLE_DIR, "document_good.png"))
        st.session_state.source_label = "document_good.png (demo sample — clean scan)"
        st.session_state.processed = False
    elif use_bad:
        st.session_state.image = Image.open(os.path.join(SAMPLE_DIR, "document_bad.png"))
        st.session_state.source_label = "document_bad.png (demo sample — poor/incomplete scan)"
        st.session_state.processed = False

    if st.session_state.image is not None:
        st.success(f"Document loaded: **{st.session_state.source_label}**")
        st.image(st.session_state.image, caption="Preview", width=450)
        if st.button("Next → Process Document", type="primary"):
            go_to(1)
            st.rerun()
    else:
        st.info("Upload a document or click a demo sample button above to continue.")

# ========================================================================================
# STEP 1 — PROCESSING (OCR + extraction happen silently, nothing technical shown)
# ========================================================================================
elif current == 1:
    st.markdown("### Step 2 — Processing Your Document")
    st.write("The system is reading the document and identifying land-record details. This only takes a moment.")

    st.image(st.session_state.image, caption=st.session_state.source_label, width=450)

    if not st.session_state.processed:
        progress = st.progress(0, text="Scanning document...")
        time.sleep(0.4)
        progress.progress(35, text="Reading text...")

        lang_code = LANGUAGE_OPTIONS[st.session_state.ocr_lang_name]
        ocr_text, cleaned_image = preprocess_and_ocr(st.session_state.image, lang=lang_code)

        progress.progress(70, text="Identifying land-record fields...")
        fields = extract_fields(ocr_text)
        time.sleep(0.3)
        progress.progress(100, text="Done.")

        # Raw extraction JSON is intentionally NOT displayed — only the structured
        # result moves forward to the Verify step. The cleaned-up scan image IS kept
        # (shown later behind an expander) so a reviewer can sanity-check what the
        # OCR engine actually saw when a field comes back wrong or blank.
        st.session_state.fields = fields
        st.session_state.preprocessed_image = cleaned_image
        st.session_state.processed = True
        st.rerun()
    else:
        st.success("✅ Document processed successfully.")

    b1, b2 = st.columns(2)
    with b1:
        if st.button("← Back"):
            go_to(0)
            st.rerun()
    with b2:
        if st.session_state.processed:
            if st.button("Next → Confidence & Verification", type="primary"):
                go_to(2)
                st.rerun()

# ========================================================================================
# STEP 2 — CONFIDENCE + HUMAN VERIFICATION
# ========================================================================================
elif current == 2:
    st.markdown("### Step 3 — Confidence Score & Human Verification")

    fields = st.session_state.fields
    errors = validate_record(fields)
    confidence = confidence_score(fields)
    label, level = confidence_bucket(confidence)
    st.session_state.errors = errors
    st.session_state.confidence = confidence

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("AI Confidence", f"{confidence}%")
        st.progress(confidence / 100)
    with c2:
        getattr(st, level)(label)
        if errors:
            st.markdown("**Validation issues found:**")
            for e in errors:
                st.write("•", e)
        else:
            st.write("All required fields present and well-formed.")

    with st.expander("🔍 View cleaned-up scan (what the OCR engine actually read)"):
        if st.session_state.preprocessed_image is not None:
            st.image(st.session_state.preprocessed_image, caption="After perspective/deskew/threshold cleanup", width=450)
        else:
            st.caption("No cleaned-up scan available for this record.")

    st.markdown("#### 👤 Human Verification")
    st.caption(
        "Review and correct the extracted fields below before approving. "
        "Every correction here becomes feedback data for improving the model over time."
    )

    edited = {}
    cols = st.columns(2)
    for i, field in enumerate(REQUIRED_FIELDS):
        with cols[i % 2]:
            edited[field] = st.text_input(field, value=fields.get(field, ""), key=f"edit_{field}")

    st.markdown("#### ✅ Decision")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("← Back"):
            go_to(1)
            st.rerun()
    with b2:
        if st.button("✅ Approve & Digitize", type="primary", use_container_width=True):
            final_confidence = confidence_score(edited)
            log_verified_record(edited, final_confidence, "Verified")
            st.session_state.final_status = "Verified"
            st.session_state.final_data = {**edited, "Confidence": final_confidence}
            go_to(3)
            st.rerun()
    with b3:
        if st.button("⚠️ Send for Human Review", use_container_width=True):
            log_verified_record(edited, confidence, "Pending Review")
            st.session_state.final_status = "Pending Review"
            st.session_state.final_data = {**edited, "Confidence": confidence}
            go_to(3)
            st.rerun()

# ========================================================================================
# STEP 3 — RECORD LOG
# ========================================================================================
elif current == 3:
    st.markdown("### Step 4 — Record Log & Details")

    if st.session_state.final_status == "Verified":
        st.success("Record verified and added to the digital land database!")
        st.balloons()
    else:
        st.warning("Record moved to the Revenue Officer review queue.")

    st.markdown("#### 📋 This Record's Details")
    st.json(st.session_state.final_data)

    st.markdown("#### 🗂️ All Records Logged This Session")
    if os.path.exists(LOG_PATH):
        log_df = pd.read_csv(LOG_PATH)
        st.dataframe(log_df, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Logged", len(log_df))
        c2.metric("Verified", int((log_df["Status"] == "Verified").sum()))
        c3.metric("Pending Review", int((log_df["Status"] == "Pending Review").sum()))
    else:
        st.caption("No records logged yet.")

    if st.button("🔁 Digitize Another Record", type="primary"):
        reset_flow()
        st.rerun()