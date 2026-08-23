# TokenFlow Architecture

## Overview

TokenFlow separates **input extraction**, **multimodal recognition**, and **text optimization** into independent stages.

The key architectural principle is:

> Expensive models are loaded only when their processing path is actually required.

BLIP image captioning is not initialized during application startup. It is invoked only when image description is actually necessary.

---

## 1. End-to-End Pipeline

```text
Upload
  ↓
Input Type Routing
  ├── TXT
  │    └── Direct UTF-8 text extraction
  │
  ├── PDF
  │    ├── Usable embedded text
  │    │    └── Native PyMuPDF extraction
  │    │
  │    └── Insufficient native text + embedded image
  │         ├── Render page with PyMuPDF
  │         └── PrintedTextRecognizer → EasyOCR
  │
  └── PNG / JPG / JPEG
       ↓
       ImageProcessor
       ↓
       Lightweight visual classification
       ├── Blank
       │    └── Structured blank result
       ├── Handwritten
       │    ├── TrOCR
       │    ├── EasyOCR fallback
       │    └── BLIP fallback
       ├── Printed
       │    └── EasyOCR → BLIP fallback
       ├── Mixed / Contextual
       │    ├── EasyOCR
       │    ├── TrOCR where appropriate
       │    └── BLIP fallback
       └── Photo
            ├── EasyOCR attempt
            └── BLIP visual description fallback

  ↓
Extracted Text / Image Description
  ↓
Normalization
  ↓
Phrase Compaction
  ↓
Negation-Safe Word Reduction
  ↓
TF-IDF Near-Duplicate Detection
  ↓
Information Importance Ranking
  ↓
Redundancy-Aware Selection
  ↓
Prompt Assembly
  ↓
Token Measurement + Metrics
```

---

## 2. Component Responsibilities

### InputExtractor

`InputExtractor` is the file-type routing layer.

Responsibilities:

- validate supported extensions;
- decode TXT files;
- extract native PDF text;
- identify scanned PDF pages;
- render scanned PDF pages;
- invoke printed OCR for scanned PDF pages;
- invoke `ImageProcessor` for PNG/JPG/JPEG;
- convert image descriptions into textual input for TokenFlow;
- preserve extraction metadata.

It does not directly perform image classification or BLIP captioning.

### ImageProcessor

`ImageProcessor` is the multimodal image-routing layer.

Responsibilities:

- open and normalize images;
- perform lightweight visual classification;
- select the appropriate recognition path;
- invoke EasyOCR for printed text;
- invoke TrOCR for handwriting;
- use OCR fallbacks where appropriate;
- invoke BLIP only when visual description is required;
- reject obviously unusable OCR;
- return a structured `ImageProcessResult`.

### PhotoCaptioner

`PhotoCaptioner` is the dedicated image-description service.

Model:

```text
Salesforce/blip-image-captioning-base
```

The model is loaded lazily and reused after initialization.

```text
PhotoCaptioner object
       ↓
No BLIP model loaded
       ↓
describe() called
       ↓
Load BLIP
       ↓
Generate description
```

This keeps photo captioning independent from OCR and prevents BLIP from unnecessarily increasing application startup time.

---

## 3. Lazy Model Loading

TokenFlow uses lazy loading for expensive recognition models.

Startup does not need to load every model:

```text
Application startup
    ↓
InputExtractor
    ↓
ImageProcessor
    ↓
No BLIP initialization
```

When a photograph actually needs a description:

```text
ImageProcessor
    ↓
_safe_caption()
    ↓
_ensure_captioner()
    ↓
PhotoCaptioner()
    ↓
BLIP loaded
    ↓
Description
```

Once loaded, the model remains available for later requests in the same process.

Lazy loading primarily improves startup time and avoids loading models that are never needed. It does not remove the first-use model-loading cost.

---

## 4. TXT Processing

TXT files are decoded using UTF-8 with replacement handling:

```python
content.decode("utf-8", errors="replace")
```

The resulting text enters the common TokenFlow optimization pipeline.

```text
TXT
 ↓
UTF-8 decoding
 ↓
Extracted text
 ↓
Optimization pipeline
```

---

## 5. PDF Processing

PDF extraction is performed page by page.

The extractor first attempts native PDF text extraction through PyMuPDF:

```python
page.get_text("text")
```

If cleaned native text reaches:

```text
MIN_TYPED_CHARS_PER_PAGE = 20
```

the page is treated as machine-readable.

Otherwise, the extractor checks whether the page contains an embedded image.

### PDF decision flow

```text
PDF page
  ↓
Native PyMuPDF text
  ↓
Enough usable text?
 ├── Yes → native text
 └── No
      ↓
Embedded image?
 ├── No → blank/unusable
 └── Yes
      ↓
Render page
      ↓
PrintedTextRecognizer
      ↓
EasyOCR
```

Pages are processed independently, allowing one PDF to contain typed, scanned, and blank pages.

The extraction result records:

- `page_count`
- `processed_pages`
- `typed_pages`
- `ocr_pages`
- `blank_pages`

Page numbers are preserved in the assembled output.

---

## 6. Scanned PDF OCR

Scanned PDF pages use the printed OCR path:

```text
Scanned PDF
    ↓
PyMuPDF rendering
    ↓
PIL image
    ↓
PrintedTextRecognizer
    ↓
EasyOCR
    ↓
OCR text
```

The default rendering resolution is:

```text
150 DPI
```

It can be overridden through:

```text
OCR_RENDER_DPI
```

For example:

```text
OCR_RENDER_DPI=175
```

or:

```text
OCR_RENDER_DPI=200
```

Higher DPI can improve recognition of small text but increases processing cost.

Scanned PDFs intentionally do not use the handwriting model by default because EasyOCR is more appropriate for printed/scanned documents.

---

## 7. Image Processing

PNG, JPG, and JPEG inputs use `ImageProcessor`.

The processor performs lightweight image analysis before selecting a recognition path.

Current classification categories are:

```text
blank
handwritten
photo
printed
mixed
contextual
```

The classifier does not require a separate ML classification model. It uses lightweight image statistics such as:

- ink ratio;
- local variance;
- edge density;
- image dimensions;
- threshold-based visual signals.

---

## 8. Blank Images

Blank or nearly blank images are rejected early.

The current threshold is:

```text
BLANK_INK_RATIO = 0.003
```

The processor can return:

```text
source_type = blank
has_text = false
```

No OCR or captioning is required for an obviously blank image.

---

## 9. Printed Images

Printed images primarily use EasyOCR:

```text
Printed image
    ↓
Preprocessing
    ↓
EasyOCR
    ↓
Reliability validation
    ↓
Extracted text
```

Conservative preprocessing may include:

- deskewing;
- grayscale conversion;
- autocontrast;
- sharpening;
- moderate contrast enhancement.

The original and preprocessed image can both be evaluated.

---

## 10. Handwritten Images

Handwritten images use TrOCR first:

```text
Handwritten image
    ↓
Preprocessing
    ↓
TrOCR
    ↓
Reliability validation
```

If the result is unreliable:

```text
TrOCR
  ↓
EasyOCR fallback
  ↓
Reliability validation
```

If both OCR paths fail, BLIP can provide a visual description as the final fallback.

The intended handwriting model is:

```text
microsoft/trocr-large-handwritten
```

---

## 11. Mixed and Contextual Images

Mixed/contextual images may contain both scene information and readable text.

The routing is:

```text
Mixed / Contextual
       ↓
EasyOCR
       ↓
Reliable?
 ├── Yes → extracted text
 └── No
       ↓
TrOCR where appropriate
       ↓
BLIP fallback
```

This avoids invoking BLIP when reliable OCR already provides a useful textual representation.

---

## 12. Photograph Processing

Photographs are valid inputs even when they contain no readable text.

The photo path is:

```text
Photo
  ↓
EasyOCR attempt
  ↓
Reliable text?
 ├── Yes
 │    └── Return extracted text
 │
 └── No
      ↓
   BLIP captioning
      ↓
   Visual description
```

Therefore:

```text
No readable OCR text
        ≠
Invalid image
```

The image can instead be represented by a visual description.

---

## 13. Image Captioning

The dedicated captioning service is:

```text
image_captioner.py
```

and uses:

```text
Salesforce/blip-image-captioning-base
```

Its responsibilities are intentionally limited to visual description.

OCR answers:

> What text is visible?

Captioning answers:

> What is visually present?

Example OCR result:

```text
SALE 50% OFF
```

Example caption:

```text
A cat sitting on a chair.
```

Keeping these responsibilities separate makes the architecture easier to optimize and replace.

---

## 14. Captioning Fallback

The image processor does not call BLIP first for every image.

The intended priority is:

```text
Try appropriate OCR
       ↓
Reliable?
 ├── Yes → use OCR
 └── No
      ↓
Need visual description?
      ↓
BLIP
```

This is important for latency because BLIP is substantially more expensive than lightweight classification.

It also avoids loading the captioning model for normal text-bearing documents.

---

## 15. Image Descriptions as TokenFlow Input

When OCR fails but BLIP produces a valid description, `InputExtractor` converts the description into the textual representation passed to the normal optimization pipeline.

Example:

```text
Image
 ↓
BLIP
 ↓
"a cat sitting on a chair"
 ↓
TokenFlow optimizer
```

Thus OCR text and visual descriptions share the same downstream pipeline:

```text
TXT ────────────────┐
PDF native ─────────┤
PDF OCR ─────────────┤
Image OCR ───────────┤
Image caption ───────┤
                     ▼
              Extracted Text
                     ▼
             TokenFlow Optimizer
```

---

## 16. OCR Reliability

OCR output is not automatically accepted.

The processor checks:

- minimum useful character count;
- useful-character ratio;
- repeated-character patterns;
- suspicious symbol sequences;
- estimated confidence.

Current minimum:

```text
MIN_USEFUL_CHARS = 5
```

This prevents obviously corrupted OCR output from entering the optimization pipeline.

---

## 17. OCR Garbage Detection

The processor attempts to reject obvious OCR failures such as:

```text
aaaaaaa
%%%%%%%
########
random repeated symbols
```

Checks include:

- excessive non-alphanumeric characters;
- repeated characters;
- suspicious long symbol sequences.

The purpose is not to guarantee perfect OCR. It is to prevent clearly unusable recognition results from being treated as meaningful text.

---

## 18. Image Preprocessing

### Printed preprocessing

```text
Input
 ↓
Deskew
 ↓
Grayscale
 ↓
Autocontrast
 ↓
Sharpen
 ↓
Contrast enhancement
 ↓
RGB
```

### Handwriting preprocessing

```text
Input
 ↓
Deskew
 ↓
Grayscale
 ↓
Autocontrast
 ↓
Median filtering
 ↓
Contrast enhancement
 ↓
RGB
```

These operations are conservative and intended to improve recognition without rewriting image content.

---

## 19. Deskewing

The processor can use OpenCV-based line detection:

```text
Image
 ↓
Grayscale
 ↓
Canny edges
 ↓
Hough line detection
 ↓
Dominant angle estimation
 ↓
Rotation when necessary
```

If OpenCV is unavailable or no reliable angle is found, the original image is retained.

---

## 20. Image Result Model

Image processing returns:

```python
@dataclass
class ImageProcessResult:
    text: str
    source_type: str
    confidence: float
    has_text: bool
    description: Optional[str]
    meta: Optional[Dict[str, Any]]
```

This keeps OCR text and visual description separate.

Typical OCR result:

```text
source_type = printed
has_text = true
recognizer = EasyOCR
```

Typical photo result:

```text
source_type = photo
has_text = false
recognizer = BLIP
description = ...
```

---

## 21. Model Reuse

Expensive models are designed to be loaded once per process and reused.

For example:

```text
First OCR request
    ↓
Load EasyOCR
    ↓
Store reader
    ↓
Recognize
```

Later:

```text
Next OCR request
    ↓
Existing EasyOCR reader
    ↓
Recognize
```

The same principle applies to TrOCR and BLIP.

This is especially important for multi-page scanned PDFs.

---

## 22. OCR Output Preservation

TokenFlow intentionally avoids aggressive OCR correction.

OCR can make mistakes involving:

- spelling;
- punctuation;
- numbers;
- technical terms;
- abbreviations;
- legal terminology.

Automatic correction can introduce additional errors.

Safe cleanup includes:

- line-ending normalization;
- whitespace normalization;
- excessive blank-line removal;
- formatting cleanup;
- page-boundary preservation.

Meaningful OCR content is otherwise preserved.

---

## 23. Normalization

All extracted representations enter the common text-processing pipeline.

Supported inputs include:

- TXT;
- native PDF text;
- scanned PDF OCR;
- printed image OCR;
- handwriting recognition;
- BLIP image descriptions.

The downstream normalization stage is shared across all modalities.

---

## 24. Phrase Compaction

TokenFlow identifies repetitive or unnecessarily verbose phrases and attempts to compact them while preserving meaning.

```text
Verbose content
      ↓
Phrase analysis
      ↓
Compact representation
```

The objective is token reduction without unnecessary information loss.

---

## 25. Negation-Safe Word Reduction

TokenFlow does not blindly remove stopwords.

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

Important requirement and negation words are therefore protected.

Examples include:

```text
not
no
never
must
should
cannot
```

---

## 26. TF-IDF Near-Duplicate Detection

TokenFlow uses TF-IDF-based similarity analysis to identify content with substantial overlap.

```text
Text segments
    ↓
TF-IDF representation
    ↓
Similarity analysis
    ↓
Near-duplicate detection
```

Redundant segments can then be deprioritized during redundancy-aware selection.

---

## 27. Information Importance Ranking

Not all text is equally valuable.

Importance scoring can prioritize:

- technical terminology;
- numerical values;
- requirements;
- constraints;
- named entities;
- domain-specific terminology;
- legal or technical information;
- other information-bearing content.

---

## 28. Redundancy-Aware Selection

The selector operates under the requested token budget.

```text
Candidate content
       ↓
Importance scoring
       ↓
Redundancy analysis
       ↓
Token budget
       ↓
Select high-value non-redundant content
```

The objective is:

```text
maximize useful information retained
under the available token budget
```

rather than simply deleting the largest possible amount of text.

---

## 29. Prompt Assembly

Selected content is reassembled into the optimized input:

```text
Selected content
      ↓
Prompt assembly
      ↓
Optimized prompt
```

The result is available for downstream LLM processing.

---

## 30. Token Measurement and Metrics

TokenFlow can track:

- original token count;
- optimized token count;
- token reduction;
- compression ratio;
- processing information;
- extraction metadata.

This makes optimization measurable rather than relying only on generated output.

---

## 31. Complete Architecture

```text
                         USER INPUT
                             │
              ┌──────────────┼──────────────┐
              │              │              │
             TXT            PDF        PNG/JPG/JPEG
              │              │              │
              │       ┌──────┴──────┐       │
              │       │             │       │
              │    Native       Scanned    Image
              │     Text           Page    Processor
              │       │             │       │
              │    PyMuPDF       Render     │
              │       │             │       │
              │       │          EasyOCR     │
              │       │             │       │
              │       │             │   ┌───┴──────────────┐
              │       │             │   │                  │
              │       │             │ Printed          Handwritten
              │       │             │   │                  │
              │       │             │ EasyOCR             TrOCR
              │       │             │   │                  │
              │       │             │   └──────┬───────────┘
              │       │             │          │
              │       │             │       Fallbacks
              │       │             │          │
              │       │             │          ▼
              │       │             │         BLIP
              │       │             │          │
              └───────┴─────────────┴──────────┘
                              │
                              ▼
                    Extracted Text / Caption
                              │
                              ▼
                       Normalization
                              │
                              ▼
                      Phrase Compaction
                              │
                              ▼
                  Negation-Safe Reduction
                              │
                              ▼
                  TF-IDF Similarity Analysis
                              │
                              ▼
                    Importance Ranking
                              │
                              ▼
                 Redundancy-Aware Selection
                              │
                              ▼
                       Prompt Assembly
                              │
                              ▼
                  Token + Metric Analysis
```

---

## 32. Component Map

```text
app/
└── services/
    ├── input_extractor.py
    │      └── File routing + PDF extraction + image integration
    │
    ├── imageprocessing.py
    │      └── Image classification + multimodal routing
    │
    ├── image_captioner.py
    │      └── Lazy BLIP visual description
    │
    ├── printed_ocr.py
    │      └── EasyOCR printed/scanned recognition
    │
    └── handwriting.py
           └── TrOCR handwriting recognition
```

The downstream TokenFlow optimization services consume the extracted textual representation.

---

## 33. Design Principles

### Separation of concerns

File extraction, OCR, handwriting recognition, image captioning, and prompt optimization are separate responsibilities.

### Lazy loading

Expensive models are loaded only when their processing paths are actually used.

### Model reuse

Once loaded, models are reused for later requests in the same process.

### Modality-aware routing

Printed text, handwriting, photographs, and scanned documents are routed to appropriate recognition paths.

### Graceful fallback

Failure of one recognition path does not immediately make an image unusable.

### Conservative text modification

OCR output is cleaned but not aggressively rewritten.

### Unified downstream representation

OCR text and visual descriptions become text before entering the common optimization pipeline.

### Metadata preservation

Source type, confidence, recognizer, and processing information are retained.

### Startup efficiency

BLIP is not loaded unless image captioning is actually needed.

### Extensibility

The dedicated captioning service can be replaced or upgraded without redesigning the complete extraction pipeline.

---

## 34. Future Extension Points

The architecture can support:

- stronger vision-language captioning models;
- richer image descriptions;
- OCR layout analysis;
- table extraction;
- mathematical expression recognition;
- multilingual OCR;
- improved handwriting recognition;
- embedding-based semantic similarity;
- BART-based summarization;
- LLMLingua-style prompt compression;
- model-specific tokenizers;
- document layout understanding.

The primary extension point for photographs is `PhotoCaptioner`. A more capable vision-language model can be introduced there while keeping `ImageProcessor` and `InputExtractor` stable.

---

## 35. Architecture Summary

TokenFlow is organized into three major layers:

```text
┌─────────────────────────────────────────────┐
│              INPUT EXTRACTION               │
│                                             │
│ TXT / PDF / PNG / JPG / JPEG                │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│          MULTIMODAL RECOGNITION             │
│                                             │
│ EasyOCR / TrOCR / BLIP                     │
│ Lazy loading + fallback routing             │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│             TOKEN OPTIMIZATION              │
│                                             │
│ Normalization                               │
│ Phrase compaction                           │
│ Negation-safe reduction                     │
│ TF-IDF redundancy analysis                  │
│ Importance ranking                          │
│ Budget-aware selection                      │
│ Prompt assembly                             │
│ Token metrics                               │
└─────────────────────────────────────────────┘
```

This separation allows TokenFlow to improve multimodal extraction independently from token optimization while keeping the final interface consistent: **all supported inputs are ultimately represented as meaningful text for the optimization pipeline.**
