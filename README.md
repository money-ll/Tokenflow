# TokenFlow

A multimodal token optimization middleware for reducing LLM input tokens while preserving important semantic information.

TokenFlow accepts TXT files, PDFs, and images, extracts their content using modality-appropriate recognition pipelines, and then applies information-aware token optimization rather than blindly deleting words.

> **Merged multimodal build.** TokenFlow combines text/PDF optimization with image understanding, including automatic image classification, printed-text OCR, handwriting recognition, OCR reliability filtering, and lazy-loaded BLIP image captioning for photographs and non-text visual content.

---

# Key Features

## Multimodal Input Processing

- TXT extraction with UTF-8 replacement handling
- Native PDF text extraction with PyMuPDF
- Per-page scanned PDF OCR fallback
- PNG/JPG/JPEG processing
- Lightweight image classification
- Printed-text recognition with EasyOCR
- Handwriting recognition with TrOCR
- Photograph/non-text image description with BLIP
- Lazy loading of heavy recognition models
- Deskewing and conservative image preprocessing
- OCR garbage-output filtering
- Page-aware PDF extraction
- Mixed PDFs containing typed, scanned, and blank pages

## Token Optimization

- Unicode and whitespace normalization
- Repeated boilerplate removal
- Conservative phrase compaction
- Negation-safe word reduction
- TF-IDF near-duplicate detection
- Information-importance ranking
- Redundancy-aware sentence selection
- Token-budget enforcement
- Optional BART abstractive compression
- Token estimation with `tiktoken`
- Compression metrics
- Stage-by-stage processing analytics

## Application

- FastAPI backend
- React/Vite frontend
- TokenFlow UI
- Image recognition metadata/badges
- Processing metadata
- Health and history endpoints

---

# Architecture

```text
                              UPLOAD
                                │
                                ▼
                         INPUT TYPE ROUTING
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
             TXT               PDF              IMAGE
              │                 │                 │
              │          ┌──────┴──────┐           │
              │          │             │           │
              │        TYPED        SCANNED       │
              │          │             │           │
              │       PyMuPDF        Render        │
              │          │             │           │
              │          │          EasyOCR        │
              │          │             │           │
              │          │             │      ImageProcessor
              │          │             │           │
              │          │             │    ┌──────┼─────────────┐
              │          │             │    │      │      │      │
              │          │             │ Printed Handwritten Photo/Context
              │          │             │    │      │      │
              │          │             │ EasyOCR TrOCR  EasyOCR
              │          │             │             │      │
              │          │             │             │     BLIP
              └──────────┴─────────────┴─────────────┴──────┘
                                │
                                ▼
                         EXTRACTED TEXT /
                       IMAGE DESCRIPTION
                                │
                                ▼
                         NORMALIZATION
                                │
                                ▼
                       PHRASE COMPACTION
                                │
                                ▼
                    NEGATION-SAFE REDUCTION
                                │
                                ▼
                    TF-IDF SIMILARITY ANALYSIS
                                │
                                ▼
                       IMPORTANCE RANKING
                                │
                                ▼
                  REDUNDANCY-AWARE SELECTION
                                │
                                ▼
                        PROMPT ASSEMBLY
                                │
                                ▼
                      TOKEN + METRIC ANALYSIS
```

---

# Input Extraction

TokenFlow deliberately separates **input extraction** from **text optimization**.

The extraction layer converts different input modalities into a common textual representation. The optimization layer then operates on that representation.

```text
Input
  │
  ├── TXT
  │    └── Direct UTF-8 extraction
  │
  ├── PDF
  │    ├── Native text → PyMuPDF
  │    └── Scanned page → EasyOCR
  │
  └── Image
       └── ImageProcessor
            ├── Printed → EasyOCR
            ├── Handwritten → TrOCR → EasyOCR fallback
            ├── Mixed/Contextual → EasyOCR → TrOCR where appropriate
            ├── Photo → EasyOCR attempt → BLIP
            └── Blank → no expensive recognition
```

---

# TXT Processing

Plain-text files are decoded using UTF-8 with replacement handling for invalid byte sequences.

```python
content.decode(
    "utf-8",
    errors="replace",
)
```

The resulting text enters the same downstream normalization and optimization pipeline used by PDF and image-derived content.

---

# PDF Processing

PDF files are processed page by page.

TokenFlow first attempts to use the native PDF text layer:

```python
page.get_text("text")
```

If a page contains enough usable native text, OCR is avoided.

If the page does not contain enough native text but contains an embedded image, the page is rendered and passed through printed-text OCR.

## PDF Decision Flow

```text
PDF Page
   │
   ▼
Native PyMuPDF Text Extraction
   │
   ├── Sufficient text
   │       └── Use native text
   │
   └── Insufficient text
           │
           ▼
      Embedded image?
           │
        ┌──┴──┐
       Yes    No
        │      │
        ▼      ▼
     Render   Blank /
      page    unusable
        │
        ▼
     EasyOCR
        │
        ▼
     OCR text
```

The decision is made independently for each page, so a single PDF may contain typed, scanned, and blank pages without forcing every page through OCR.

## Typed Page Threshold

The default threshold is:

```text
MIN_TYPED_CHARS_PER_PAGE = 20
```

A page meeting this threshold uses its native text layer.

## Scanned PDF OCR

```text
Scanned PDF Page
      ↓
PyMuPDF Rendering
      ↓
PIL Image
      ↓
PrintedTextRecognizer
      ↓
EasyOCR
      ↓
OCR Text
```

The default rendering resolution is **150 DPI**.

It can be changed with:

```env
OCR_RENDER_DPI=175
```

or:

```env
OCR_RENDER_DPI=200
```

Higher DPI may improve recognition of small text but increases image size and OCR processing time.

## Page Preservation

Extracted pages retain their original page number:

```text
[Page 1]
Native PDF text...

[Page 2 - scanned, OCR]
Recognized OCR text...

[Page 3]
Native PDF text...
```

PDF metadata records:

- `page_count`
- `processed_pages`
- `typed_pages`
- `ocr_pages`
- `blank_pages`

---

# Image Processing

PNG, JPG, and JPEG inputs are processed through `ImageProcessor`.

The processor performs lightweight image analysis before selecting an expensive recognition model.

```text
Image
  ↓
ImageProcessor
  ↓
Lightweight Classification
  ↓
Recognition Strategy
```

Classification uses inexpensive visual statistics such as:

- ink ratio
- local variance
- edge density
- visual structure

No separate machine-learning classifier is required for this stage.

Possible classifications include:

```text
printed
handwritten
mixed
contextual
photo
blank
```

---

# Printed Images

Printed/document-like images are routed to EasyOCR.

```text
Printed Image
      ↓
Deskew / preprocessing
      ↓
EasyOCR
      ↓
Extracted Text
```

Conservative preprocessing can include:

- deskewing
- grayscale conversion
- autocontrast
- contrast enhancement
- sharpening

The preprocessing is designed to improve recognition without aggressively modifying document content.

---

# Handwritten Images

Images classified as strong handwriting candidates use TrOCR first.

```text
Handwritten Image
      ↓
TrOCR
      ↓
Recognized Text
```

The intended handwriting model is:

```text
microsoft/trocr-large-handwritten
```

If TrOCR does not produce reliable output, EasyOCR is attempted as a fallback.

```text
Handwritten
    │
    ▼
  TrOCR
    │
    ├── Reliable → return text
    │
    └── Unreliable
          │
          ▼
       EasyOCR
          │
          ├── Reliable → return text
          │
          └── Unreliable → BLIP description fallback
```

TrOCR remains separate from printed OCR because printed and handwritten recognition have different characteristics.

---

# Photograph and Non-Text Image Understanding

A photograph is not considered a failed input simply because OCR cannot find readable text.

For photographs and visual scenes, TokenFlow uses a separate image-captioning stage.

```text
Photo
  │
  ▼
EasyOCR attempt
  │
  ├── Reliable text → return OCR text
  │
  └── No reliable text
          │
          ▼
        BLIP
          │
          ▼
   Image Description
```

This allows images such as:

- animals
- landscapes
- buildings
- vehicles
- people in scenes
- objects
- rooms
- everyday photographs

to become usable textual representations for the optimization pipeline.

For example:

```text
Image
  ↓
BLIP
  ↓
"a cat sitting on a chair"
  ↓
TokenFlow optimization
```

The description is passed downstream as text, so the normal TokenFlow optimization stages can process it.

---

# Deep Image Description

The captioning architecture is intentionally isolated in:

```text
app/services/image_captioner.py
```

The dedicated component is `PhotoCaptioner`.

Its responsibility is only visual description.

```text
OCR
  └── What text is visible?

BLIP
  └── What is visually present?
```

This separation prevents image captioning from being unnecessarily invoked for normal text documents.

## BLIP Model

The current captioning model is:

```text
Salesforce/blip-image-captioning-base
```

The model is loaded lazily.

It is **not loaded during application startup**.

Instead:

```text
Application Startup
       │
       ▼
PhotoCaptioner object only
       │
       ▼
No BLIP model loaded
       │
       ▼
Captioning becomes necessary
       │
       ▼
Load BLIP
       │
       ▼
Cache model
       │
       ▼
Generate description
```

After the first captioning request, the loaded processor/model is reused for subsequent images within the same process.

This is important because BLIP can be expensive to initialize and should not increase startup time for applications that never process photographs.

## CPU / GPU Selection

The captioner automatically selects:

```text
CUDA → when PyTorch detects a compatible GPU
CPU  → otherwise
```

The model is placed on the selected device and switched to evaluation mode.

Inference uses:

```python
torch.inference_mode()
```

to avoid unnecessary training-related overhead.

## Captioning Output

The caption is cleaned only with safe whitespace normalization.

The caption is then returned as the textual representation of the image.

---

# Image Processing Fallback Strategy

The complete image routing strategy is:

```text
                         IMAGE
                           │
                           ▼
                    Classification
                           │
       ┌───────────┬───────┼──────────┬──────────┐
       │           │       │          │          │
     Printed   Handwritten Mixed   Contextual   Photo
       │           │       │          │          │
       ▼           ▼       ▼          ▼          ▼
    EasyOCR      TrOCR  EasyOCR    EasyOCR    EasyOCR
       │           │       │          │          │
       │           └───┐   │          │          │
       │               │   └────┬─────┘          │
       │               │        │                │
       │               ▼        ▼                ▼
       │           EasyOCR   TrOCR             BLIP
       │               │        │                │
       └───────────────┴────────┴────────────────┘
                           │
                           ▼
                 Reliable text / description
                           │
                           ▼
                   TokenFlow optimizer
```

If OCR is unreliable and captioning is unavailable, TokenFlow returns a structured failure rather than silently treating meaningless OCR output as valid text.

---

# Lazy Model Loading

Heavy models are deliberately separated from lightweight service construction.

The application can construct:

```python
ImageProcessor(...)
PhotoCaptioner(...)
```

without loading BLIP.

Similarly, OCR recognizers can defer their heavy model initialization until recognition is actually required.

The principle is:

```text
Startup
  ↓
Lightweight objects only
  ↓
Request arrives
  ↓
Required recognizer selected
  ↓
Model loaded once
  ↓
Model reused
```

This keeps startup fast while preserving warm-process performance.

Note that **lazy loading improves startup time, not the actual neural inference cost**. A first BLIP/TrOCR/EasyOCR request can still be slower because the corresponding model must be initialized.

---

# OCR Reliability and Preservation

TokenFlow deliberately avoids aggressive OCR correction.

OCR may produce errors involving:

- spelling
- punctuation
- characters
- numbers
- technical terminology
- abbreviations
- legal terminology

Automatically "correcting" these values can introduce new errors.

Therefore, TokenFlow performs conservative cleanup such as:

- line-ending normalization
- whitespace normalization
- excessive blank-line removal
- formatting cleanup
- page-boundary preservation

It does not perform aggressive spelling correction.

---

# OCR Garbage Detection

Before accepting OCR output, TokenFlow checks whether the result is plausibly useful.

The reliability layer considers:

- minimum useful character count
- alphanumeric/whitespace ratio
- repeated-character patterns
- suspicious symbol sequences
- estimated confidence

This prevents obviously corrupted OCR output from entering the optimization pipeline.

---

# OCR Model Reuse

Heavy OCR models are reused after initialization.

```text
First OCR request
      ↓
Initialize recognizer
      ↓
Store model/reader
      ↓
Process input
```

Subsequent requests:

```text
Next OCR request
      ↓
Existing model/reader
      ↓
Process input
```

This is particularly important for multi-page scanned PDFs.

---

# GPU Acceleration

EasyOCR and transformer-based recognition can use GPU acceleration when a compatible PyTorch CUDA environment is installed.

The system should be benchmarked on the actual hardware because OCR performance depends on:

- image resolution
- number of text regions
- document layout
- model
- GPU memory
- CUDA/PyTorch configuration

GPU availability does not eliminate neural inference time; it mainly reduces that cost compared with CPU execution.

---

# Common Text Optimization Pipeline

Once extraction is complete, all modalities enter the common optimization pipeline:

```text
Extracted Text / Image Description
              ↓
Unicode / Whitespace Normalization
              ↓
Repeated Boilerplate Removal
              ↓
Conservative Phrase Compaction
              ↓
Negation-Safe Word Reduction
              ↓
Duplicate Detection
              ↓
TF-IDF Near-Duplicate Detection
              ↓
Information Importance Ranking
              ↓
Redundancy-Aware Selection
              ↓
Optional BART Compression
              ↓
Final Prompt Assembly
              ↓
Token Budget Enforcement
              ↓
Token + Compression Metrics
```

---

# Compression Philosophy

TokenFlow does **not** blindly remove words.

Naive stopword removal can destroy meaning.

For example:

```text
The system should not remove this.
```

must not become:

```text
system remove
```

Likewise:

```text
The model must never ignore safety constraints.
```

must not become:

```text
model ignore safety constraints
```

TokenFlow protects semantic-critical information including:

- `not`
- `no`
- `never`
- `neither`
- `nor`
- `without`
- `must`
- `should`
- `cannot`
- `can't`
- `may`
- `might`
- `required`

It also protects information such as:

- numbers
- dates
- percentages
- units
- URLs
- code-like strings
- quoted phrases
- technical terminology
- named entities
- requirements
- constraints

The objective is:

```text
        Information Retained
       ───────────────────────
          Token Budget Used
```

rather than simply maximizing deleted words.

---

# Phrase Compaction

TokenFlow identifies repetitive or unnecessarily verbose phrases and attempts to compact them while preserving meaning.

The operation is conservative and is part of a broader information-preservation strategy.

---

# TF-IDF Similarity

TokenFlow uses TF-IDF representations and cosine similarity to identify duplicate or near-duplicate sentences.

```text
Sentence A ──→ TF-IDF Vector ──┐
                               ├──→ Cosine Similarity
Sentence B ──→ TF-IDF Vector ──┘
                                      │
                                      ▼
                              Redundancy Decision
```

This allows semantically similar text to be handled even when the strings are not exactly identical.

---

# Information Importance Ranking

TokenFlow assigns higher importance to content likely to be semantically significant.

Examples include:

- technical terms
- numerical values
- dates
- requirements
- constraints
- named entities
- domain-specific terminology
- important statements

The ranking is used during redundancy-aware selection.

---

# Redundancy-Aware Selection

The optimizer selects the most valuable information under the requested token budget.

The objective is not:

```text
Delete as many words as possible.
```

Instead:

```text
Retain the most useful information
while respecting the token budget.
```

This is especially important for technical, instructional, legal, and structured inputs.

---

# Optional BART Compression

TokenFlow can optionally use BART-based abstractive compression.

The core optimizer can operate without loading a large generative model.

When enabled:

```text
Extracted Text
      ↓
Token Optimization
      ↓
BART Compression
      ↓
Final Prompt
```

Install the optional dependencies with:

```bash
pip install transformers torch
```

---

# Token Measurement

TokenFlow can use `tiktoken` when available.

Metrics can include:

- original token count
- optimized token count
- token reduction
- compression ratio
- processing time
- stage-level processing information

Example:

```text
Original Tokens:    2500
Optimized Tokens:   1450
Tokens Reduced:     1050
Compression:        42%
```

Exact values depend on the configured tokenizer and optimization request.

---

# API

## Optimize

```http
POST /api/v1/optimize
```

Typical multipart fields:

```text
file
query
target_reduction
```

### Supported Files

```text
.txt
.pdf
.png
.jpg
.jpeg
```

### `query`

Optional user task or query used by the optimization pipeline.

### `target_reduction`

Optional target token reduction, typically between:

```text
0.0 – 0.9
```

---

## Health

```http
GET /api/v1/health
```

Used to verify backend availability.

---

## History

```http
GET /api/v1/history
```

Returns available optimization history information.

---

# Backend Setup

The backend is implemented using FastAPI.

```bash
cd backend

python -m venv .venv
```

## Windows

```bash
.venv\Scripts\activate
```

## macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn app.main:app --reload --port 8000
```

The backend is then available at:

```text
http://localhost:8000
```

---

# Frontend Setup

The frontend uses React and Vite.

```bash
cd frontend
npm install
npm run dev
```

The frontend expects the backend API at:

```text
http://localhost:8000
```

---

# Dependencies

## Core

```bash
pip install fastapi uvicorn python-multipart pydantic
```

## PDF Processing

```bash
pip install pymupdf
```

## Image Processing

```bash
pip install pillow numpy
```

## Printed OCR

```bash
pip install easyocr
```

## Handwriting / Image Captioning

```bash
pip install transformers torch
```

## Token Measurement

```bash
pip install tiktoken
```

The project requirements file should be treated as the authoritative dependency list for the current build.

---

# Environment Configuration

OCR rendering resolution:

```env
OCR_RENDER_DPI=150
```

Example:

```env
OCR_RENDER_DPI=200
```

Higher values can improve recognition of small text but increase processing time.

---

# Project Structure

A simplified structure is:

```text
TokenFlow/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── routes.py
│   │   ├── schemas.py
│   │   │
│   │   └── services/
│   │       ├── extractor.py
│   │       ├── printed_ocr.py
│   │       ├── handwriting.py
│   │       ├── imageprocessing.py
│   │       ├── image_captioner.py
│   │       ├── optimizer.py
│   │       ├── pipeline.py
│   │       └── history.py
│   │
│   └── requirements.txt
│
└── frontend/
    ├── src/
    ├── package.json
    └── vite.config.*
```

---

# Responsibility of the Three Multimodal Components

## `image_captioner.py`

Responsible only for visual descriptions.

```text
Image
  ↓
PhotoCaptioner
  ↓
BLIP
  ↓
Visual description
```

BLIP is lazy-loaded and cached after first use.

## `imageprocessing.py`

Responsible for deciding what kind of image it is and which recognizer should handle it.

```text
Image
  ↓
Classification
  ↓
EasyOCR / TrOCR / BLIP
  ↓
ImageProcessResult
```

It also handles reliability checks and fallbacks.

## `extractor.py`

Responsible for file-level input routing.

```text
TXT  → direct extraction
PDF  → native text / scanned OCR
IMG  → ImageProcessor
```

It converts the resulting image description into normal textual input when OCR text is unavailable.

This means the downstream TokenFlow optimizer does not need separate optimization logic for photographs.

---

# End-to-End Multimodal Flow

```text
                    USER UPLOAD
                         │
          ┌──────────────┼──────────────┐
          │              │              │
         TXT            PDF           IMAGE
          │              │              │
          │       ┌──────┴──────┐       │
          │       │             │       │
          │     Typed        Scanned    │
          │       │             │       │
          │    PyMuPDF        EasyOCR   │
          │       │             │       │
          │       └──────┬──────┘       │
          │              │              │
          │              │         ImageProcessor
          │              │              │
          │              │      ┌───────┼────────┐
          │              │      │       │        │
          │              │   Printed  Handwritten Photo
          │              │      │       │        │
          │              │   EasyOCR   TrOCR    BLIP
          │              │      │       │        │
          └──────────────┴──────┴───────┴────────┘
                         │
                         ▼
                COMMON TEXT REPRESENTATION
                         │
                         ▼
                    NORMALIZATION
                         │
                         ▼
                 PHRASE COMPACTION
                         │
                         ▼
              NEGATION-SAFE REDUCTION
                         │
                         ▼
                 TF-IDF ANALYSIS
                         │
                         ▼
                IMPORTANCE RANKING
                         │
                         ▼
             REDUNDANCY-AWARE SELECTION
                         │
                         ▼
                  PROMPT ASSEMBLY
                         │
                         ▼
                  TOKEN METRICS
```

---

# Performance Considerations

Native PDF text extraction is substantially cheaper than OCR.

Scanned pages require neural OCR inference and can therefore take several seconds per page even when the OCR model is already warm.

Similarly, the first request requiring BLIP or TrOCR may include model initialization overhead.

TokenFlow therefore prioritizes:

- native PDF extraction whenever possible
- lazy loading of heavy models
- persistent model reuse
- GPU acceleration when available
- controlled rendering resolution
- avoiding unnecessary OCR
- avoiding BLIP unless visual description is actually required
- lightweight preprocessing
- conservative post-processing

A warm model removes model-loading overhead, but it does **not** eliminate the cost of neural inference itself.

For production deployments, benchmark the actual workload and hardware rather than assuming that a GPU or larger batch is always faster.

---

# Design Principles

### 1. Preserve Meaning

Optimization must not destroy important semantic information.

### 2. Avoid Unnecessary OCR

Native PDF text is preferred whenever usable text already exists.

### 3. Use the Right Recognition Model

```text
Printed / Scanned → EasyOCR
Handwritten       → TrOCR
Photographs       → BLIP when OCR is not useful
```

### 4. Load Heavy Models Only When Needed

BLIP is not initialized just because the application starts.

### 5. Reuse Heavy Models

Once initialized, recognition models are reused by the running process.

### 6. Preserve Document Structure

PDF page boundaries and page numbers are retained during extraction.

### 7. Optimize Information, Not Just Words

The optimizer considers importance and redundancy rather than relying exclusively on word deletion.

### 8. Keep Extraction Separate From Optimization

Recognition components produce a textual representation. The optimizer decides how that representation should be compressed.

---

# Current Limitations

- BLIP base produces general-purpose captions rather than guaranteed detailed scene understanding.
- OCR accuracy depends on image quality, text size, orientation, language, and document layout.
- TrOCR is optimized for handwriting and should not replace printed-document OCR.
- Image classification is intentionally lightweight and heuristic rather than a dedicated trained classifier.
- Native PDF text extraction may still contain layout artifacts.
- Neural inference remains computationally expensive even after model warm-up.
- Captioning should not be treated as exact visual fact extraction for specialized domains.

---

# Future Extensions

Potential extensions include:

- stronger BLIP/BLIP-2 style image understanding
- richer multi-stage image descriptions
- OCR + caption fusion
- document layout analysis
- table and form extraction
- mathematical expression recognition
- multilingual OCR
- specialized OCR models
- embedding-based semantic similarity
- model-specific tokenizers
- LLMLingua-style prompt compression
- BART-based abstractive compression
- structure-aware document compression
- image question answering
- vision-language models for deeper scene reasoning

The modular architecture allows OCR, handwriting recognition, image captioning, extraction, and optimization components to evolve independently.

---

# Architecture Summary

TokenFlow separates the system into three major layers:

```text
┌───────────────────────────────────────────────┐
│              INPUT / RECOGNITION              │
│                                               │
│ TXT | PDF | EasyOCR | TrOCR | BLIP            │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│                  EXTRACTION                   │
│                                               │
│ File routing + page handling + result fusion  │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│                 OPTIMIZATION                 │
│                                               │
│ Normalize → Compact → Protect → Rank → Select │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│                  LLM INPUT                    │
│                                               │
│ Optimized prompt + token/compression metrics  │
└───────────────────────────────────────────────┘
```

The central design goal is simple:

> **Reduce token usage while preserving the information that actually matters.**
