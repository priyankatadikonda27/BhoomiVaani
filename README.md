# 🌱 BhoomiVaani — Prototype

**SIH 2026 · Problem Statement 26018 · Team Catalyst**
AI-Powered Land Record Digitization & Validation Platform

This is the **2-hour hackathon prototype**, built to demonstrate the core pipeline
described in the team's proposal — without building out the full production stack
(React/FastAPI/PostgreSQL/Docker/ML training etc.). It proves the *concept*, not the
final architecture.

## What it demonstrates

```
Upload Land Record → [Processing: OCR + Field Extraction happen silently] →
      Confidence Score → Auto-Verify / Human Review → Record Log
```

- **OCR**: OpenCV preprocessing (grayscale, blur, Otsu threshold) + Tesseract OCR,
  with a document-language selector (English / Hindi / Telugu / Mixed)
- **Field extraction**: rule-based regex extraction of the 7 core fields
  (Owner, Survey No, Khata No, Area, Village, Tehsil, District) — this runs
  in the background; the raw OCR text and extraction internals are **not**
  shown in the interface, only the final structured result
- **Validation engine**: flags missing/malformed fields
- **Confidence score**: % of required fields successfully filled, with
  green/yellow/red routing (auto-verify / review recommended / human review required)
- **Human review UI**: officer can edit fields, then Approve & Digitize or send
  for review — every correction is logged (stand-in for the active-learning loop)
- **Record log**: this record's final details plus a table of everything
  logged in the session

## Multi-language OCR (Hindi / Telugu)

Tesseract's Hindi (`hin`) and Telugu (`tel`) language packs are supported
alongside English (`eng`) — pick the document's language on the Upload step.

Two things worth knowing:
- **Mixing all three languages at once on an English document reduces accuracy**
  (tested: Tesseract starts misreading Latin table lines as Telugu/Hindi
  glyphs), so the app asks you to pick the dominant script rather than always
  guessing all three. A "Mixed / Auto" option is available for genuinely
  multilingual documents.
- **Field labels in the extraction patterns are currently English** ("Owner",
  "Survey No", ...). If a document's field *values* are in Hindi/Telugu script
  (e.g. an owner's name), they're still captured correctly — the value-capture
  patterns accept any characters. But if the field *labels themselves* are
  written in Hindi/Telugu script rather than English, extraction won't find
  them yet. Matching vernacular-script labels too is a natural next step
  beyond this prototype (the production system's IndicBERT/Bhashini-based
  NLP classification in the full proposal is designed to handle exactly this).

## Files

```
bhoomivaani/
├── app.py                          # Main Streamlit application (the full pipeline)
├── generate_samples.py             # Generates the two simple demo document images
├── generate_realistic_sample.py    # Generates a realistic table-format ROR document
├── sample_docs/
│   ├── document_good.png           # Clean scan — all 7 fields present
│   ├── document_bad.png            # Damaged/incomplete scan — 3 fields missing
│   └── sample_real_land_record.png # Realistic government-style ROR document
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt

# Tesseract binary is required (the Python package is just a wrapper):
# Ubuntu/Debian:
sudo apt-get install -y tesseract-ocr tesseract-ocr-hin tesseract-ocr-tel
# Mac:
brew install tesseract
brew install tesseract-lang   # for Hindi/Telugu
# Windows: install from https://github.com/UB-Mannheim/tesseract/wiki
#          (select additional language data during setup for Hindi/Telugu)
```

## Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Recommended judge demo flow (2 minutes)

1. On the **Upload** step, leave language as "English" and click
   **"Load Clean Sample Record"**, then click Next.
2. The **Processing** step shows a short progress animation — OCR and field
   extraction are running behind the scenes, not exposed on screen.
3. On **Verify**, you'll see **100% confidence → auto-verified (green)**.
   Click Approve & Digitize.
4. Go back and repeat with **"Load Damaged/Incomplete Sample"** →
   **57% confidence → human review (red)** — the moment that shows the system
   isn't "just OCR", it knows when to trust itself and when to ask a human.
5. On **Record Log**, show the record's final details and the session log table.

### Suggested spoken explanation

> "BhoomiVaani is not just an OCR system. We convert legacy land documents into
> structured digital records. The document is preprocessed and passed through OCR.
> We extract key land fields — owner, survey number, khata number, area, and location.
> A validation engine checks the extracted data and generates a confidence score.
> High-confidence records are verified automatically, while low-confidence or
> inconsistent records are sent to a revenue official for human review. Every
> correction becomes feedback for improving the model over time."

## What this prototype intentionally does NOT include

These belong to the full production architecture in the proposal, not the 2-hour demo:
React frontend, FastAPI backend, PostgreSQL/PostGIS, MongoDB, AWS, Docker,
IndicBERT/TrOCR fine-tuning, Bhashini API integration, real LRMS/DILRMP sync,
vernacular-script field-label recognition, ML-based duplicate detection.

The prototype's rule-based extraction/validation stands in for the future
ML-based NLP classification, duplicate detection, and active-learning retraining
loop described in the proposal — the same pipeline shape, at hackathon scale.
