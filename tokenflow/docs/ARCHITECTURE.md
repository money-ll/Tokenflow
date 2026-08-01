# TokenFlow Architecture

## Pipeline

```text
Upload
  ↓
Input Type Routing
  ├── TXT → normalization
  ├── PDF → PyMuPDF extraction
  └── PNG/JPG (handwritten) → line segmentation → TrOCR recognition
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

## Why not direct stopword removal?

A naive stopword list can destroy meaning:

- `The system should not remove this.` → `system remove`
- `The model must never ignore safety constraints.` → `model ignore safety constraints`

TokenFlow keeps negation and requirement words in a protected vocabulary. It also protects numerical and technical content through sentence-level importance scoring.

## Typed vs. scanned PDF pages

There's no visual/ML classifier here — the distinction is made per page,
directly from what PyMuPDF can extract:

1. Try `page.get_text("text")`. If it returns at least
   `MIN_TYPED_CHARS_PER_PAGE` (20) real characters, the page is **typed**
   and that text is used directly.
2. If not, check whether the page contains an embedded raster image
   (`page.get_images(full=True)`):
   - **No image** → the page is genuinely **blank**, skipped, no OCR run.
   - **Has an image** → the page is almost certainly **scanned**. It's
     rendered to a bitmap (`page.get_pixmap(dpi=200)`) and passed through
     `PrintedTextRecognizer` (EasyOCR) as a printed-text OCR fallback
     (Image Pipeline - Path A).
3. Recognized/typed text from every page is reassembled **in page order**
   into a single document before compression, so a PDF with a mix of
   typed and scanned pages is handled correctly instead of silently
   dropping the scanned pages or hard-failing the whole file.

The extraction result reports `typed_pages`, `ocr_pages`, and
`blank_pages` so the frontend can show which path each page took.

Known limitations of this heuristic: a page with a large decorative
image but very little real text will still be sent through OCR
(harmless — OCR just finds nothing useful); an already-OCR'd scanned
PDF (i.e. one with an invisible text layer baked in by another tool)
is correctly treated as "typed" since real characters are present.

## Handwritten text recognition (Image Pipeline - Path B)

Uploaded `.png`/`.jpg`/`.jpeg` images are treated as handwritten notes:

1. `HandwritingRecognizer` (`app/services/handwriting.py`) segments the
   page into line-level crops using a horizontal ink-density projection
   profile (no extra model required for this step).
2. Each line crop is passed through TrOCR (`microsoft/trocr-large-handwritten`)
   to produce recognized text.
3. Recognized lines are rejoined into a single string and fed through the
   **same** Text Compression Module used for TXT/PDF input, so token
   reduction, negation-safe protection, and metrics all apply identically
   regardless of source modality.

The model is loaded lazily on first use — starting the API or processing
TXT/PDF files never requires torch/transformers to be installed.

## Future ML extension

The pipeline has a clear extension point for:
- BART summarization
- LLMLingua-style prompt compression
- embedding-based semantic similarity
- printed-text OCR fallback for scanned PDFs is now implemented (see above); remaining Image Pipeline paths are math expression recognition and object/scene description
- model-specific tokenizers
