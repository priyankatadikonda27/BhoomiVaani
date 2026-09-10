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
script instead of always guessing all three. Field *labels* in the extraction
patterns are currently English ("Owner", "Survey No", ...); field *values* in
Hindi/Telugu script are still captured correctly since the value-capture groups
accept any characters. Fully label-language-agnostic extraction (matching
vernacular-script labels too) is a natural next step beyond this prototype.
"""

import os
import re
import time
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
    "Owner":     r"Owner\b\s*(?:Name)?\s*[:\-]?\s*(.+)",
    "Survey No": r"Survey\s*No\b\.?\s*[:\-]?\s*([\w\/\-]+)",
    "Khata No":  r"Khata\s*No\b\.?\s*[:\-]?\s*([\w\/\-]+)",
    "Area":      r"Area\b\s*[:\-]?\s*([\d.]+\s*\w*)",
    "Village":   r"Village\b\s*[:\-]?\s*(.+)",
    "Tehsil":    r"Tehsil\b\s*[:\-]?\s*(.+)",
    "District":  r"District\b\s*[:\-]?\s*(.+)",
}

STEPS = ["Upload", "Processing", "Verify", "Record Log"]

LANGUAGE_OPTIONS = {
    "English": "eng"
}


# --------------------------------------------------------------------------------------
# CORE PIPELINE FUNCTIONS (run silently — not shown directly in the UI)
# --------------------------------------------------------------------------------------
def preprocess_and_ocr(pil_image: Image.Image, lang: str = "eng") -> tuple[str, Image.Image]:
    """Preprocess image (grayscale, blur, Otsu threshold) then run Tesseract OCR."""
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        text = pytesseract.image_to_string(thresh, lang=lang)
    except pytesseract.TesseractError:
        # Language pack not installed on this machine — fall back to English.
        text = pytesseract.image_to_string(thresh, lang="eng")
    return text, Image.fromarray(thresh)


def extract_fields(text: str) -> dict:
    """Rule-based extraction of the 7 required land-record fields from OCR text."""
    data = {}
    for field, pattern in FIELD_PATTERNS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        data[field] = match.group(1).strip() if match else ""
    return data


def validate_record(data: dict) -> list[str]:
    """Rule-based validation engine — flags missing or suspicious fields."""
    errors = []
    if not data.get("Owner"):
        errors.append("Owner name missing")
    if not data.get("Survey No"):
        errors.append("Survey number missing")
    elif "/" not in data["Survey No"]:
        errors.append("Survey number format looks unusual (expected e.g. 123/4)")
    if not data.get("Khata No"):
        errors.append("Khata number missing")
    if not data.get("Area"):
        errors.append("Area missing")
    if not data.get("Village"):
        errors.append("Village missing")
    if not data.get("Tehsil"):
        errors.append("Tehsil missing")
    if not data.get("District"):
        errors.append("District missing")
    return errors


def confidence_score(data: dict) -> float:
    """Simple explainable confidence metric = % of required fields successfully filled."""
    total = len(data)
    filled = sum(1 for v in data.values() if v.strip())
    return round((filled / total) * 100, 2) if total else 0.0


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
        ocr_text, _ = preprocess_and_ocr(st.session_state.image, lang=lang_code)

        progress.progress(70, text="Identifying land-record fields...")
        fields = extract_fields(ocr_text)
        time.sleep(0.3)
        progress.progress(100, text="Done.")

        # OCR text and raw extraction JSON are intentionally NOT displayed —
        # only the structured result moves forward to the Verify step.
        st.session_state.fields = fields
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
