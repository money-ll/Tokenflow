# TokenFlow

**Stop paying to re-send the same redundant tokens to every LLM call.**

TokenFlow is a multimodal middleware that sits between your documents and your LLM. Upload plain text, a PDF, or even a photo of handwritten notes — it extracts, cleans, and compresses the content down to the essentials before it ever reaches GPT, Claude, or Gemini, cutting token usage by up to **60%** while explicitly protecting the words that change meaning: negations, requirements, numbers, and named entities.

No blind truncation. No "summarize and hope." Every reduction is measured, staged, and explainable.

## Why it's different

Most "prompt compression" tools either truncate (losing information) or summarize with a black-box model (risking hallucination and silently reversing meaning — try naively stripping stopwords from *"the system should **not** remove this"*). TokenFlow instead runs a transparent, staged pipeline where every step is auditable, and the words most likely to flip meaning are explicitly protected from removal.

## What it does

**Input handling**
- Plain text (`.txt`) upload
- PDF upload with page-by-page extraction via PyMuPDF
- **Automatic per-page typed vs. scanned detection** — no manual flag needed. Pages with a real text layer are extracted directly; pages with no text but an embedded image are treated as scanned and run through OCR; genuinely blank pages are skipped. Typed and scanned pages can be mixed in the same PDF.
- **Handwritten image upload** (`.png` / `.jpg` / `.jpeg`) — line-segmented and recognized end-to-end via TrOCR, then fed through the same compression pipeline as everything else

**Compression pipeline**
1. Unicode and whitespace normalization
2. Repeated boilerplate / page-number noise removal
3. Conservative phrase compaction (e.g. *"in order to"* → *"to"*)
4. Negation-safe, protected-vocabulary stopword reduction
5. Near-duplicate sentence removal (TF-IDF cosine similarity)
6. Importance-weighted sentence ranking (numeric content, named entities, protected words, length penalty)
7. Redundancy-aware extractive selection under a target token budget
8. Final prompt assembly with hard token-budget enforcement

**Everything is measured**
- Token counts via `tiktoken` at every stage, before and after
- Token Reduction Rate (TRR), duplicate-sentence count, per-stage breakdown
- Typed / OCR / blank page counts surfaced for every PDF

**GPU-accelerated where it matters**
- Handwriting recognition (TrOCR) and scanned-PDF OCR (EasyOCR) both auto-detect CUDA and run in half precision on GPU, falling back cleanly to CPU when no GPU is available
- Heavy models load once and stay resident across requests — not reloaded per file

**Interface**
- FastAPI backend with a single `/optimize` endpoint
- React/Vite frontend: drag-and-drop upload, adjustable compression target, live token metrics, per-stage breakdown, copy/download of the optimized prompt

## Quick start

### Backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend expects the API at `http://localhost:8000`.

## GPU acceleration (optional but recommended)

Handwriting recognition and scanned-PDF OCR both run noticeably faster on a CUDA GPU. `requirements.txt` installs the CPU build of PyTorch by default — to use your GPU, install the CUDA build that matches your driver **instead**:

```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

(Swap `cu128` for the CUDA version reported by `nvidia-smi` — check [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) for the exact tag.) No code changes or config flags needed — TokenFlow detects the GPU automatically at runtime.

## Dependencies

Everything below is required (not optional) — `.txt`/typed-PDF processing works fine on a base install, but handwriting and scanned-PDF support are core features of this build, not add-ons:

| Feature | Library |
|---|---|
| PDF text extraction | `pymupdf` |
| Compression / dedup | `scikit-learn`, `nltk` |
| Token counting | `tiktoken` |
| Handwriting recognition | `torch`, `transformers` (TrOCR) |
| Scanned-page OCR | `easyocr` |

All heavy models (TrOCR, EasyOCR) are loaded **lazily** on first use — starting the server or processing plain text/typed PDFs never triggers a model download.

## API

`POST /api/v1/optimize`

Multipart fields:
| Field | Type | Notes |
|---|---|---|
| `file` | file | `.txt`, `.pdf`, `.png`, `.jpg`, or `.jpeg` |
| `query` | string, optional | user task/query to append to the assembled prompt |
| `target_reduction` | float, optional | `0.0`–`0.9`, default `0.45` |

`GET /api/v1/health` — liveness check

`GET /api/v1/history` — past optimization runs

## Project alignment

The architecture follows the full TokenFlow project concept: input processing → multimodal extraction → knowledge fusion → prompt optimization → LLM interface → evaluation → data logging → response assembly. Currently implemented: the text and PDF pipelines (with per-page scanned/typed classification), the handwritten-image pipeline, the full extractive compression module, prompt assembly, and the FastAPI/React application layer. Not yet implemented: math-expression recognition and object/scene description (the remaining Image Pipeline paths), abstractive (BART) summarization as a fluency refinement stage, and the BERTScore/ROUGE evaluation suite.
