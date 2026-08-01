# TokenFlow

A multimodal token optimization middleware for reducing LLM input tokens while preserving semantic integrity.

## Included

- Text and PDF upload
- PDF extraction with PyMuPDF
- Optional OCR fallback for scanned PDFs
- Conservative semantic compression
- Negation-safe stopword handling
- Sentence deduplication using TF-IDF cosine similarity
- Content-aware sentence importance ranking
- Extractive compression fallback that works without downloading large models
- Optional BART summarization when enabled
- Token estimation with `tiktoken` when available
- Compression metrics and stage-by-stage analytics
- React/Vite frontend matching the TokenFlow dark UI
- FastAPI backend

## Compression philosophy

TokenFlow does **not** blindly delete words. The pipeline protects:

- negations: `not`, `no`, `never`, `neither`, `nor`, `without`, etc.
- modal/critical words: `must`, `should`, `cannot`, `can't`, `may`, `might`, `required`
- numbers, dates, percentages, units, code-like strings, URLs and quoted phrases
- sentences containing high-information entities and technical terms

The default pipeline is:

1. Unicode and whitespace normalization
2. Repeated boilerplate removal
3. Conservative phrase compaction
4. Negation-safe stopword reduction
5. Duplicate and near-duplicate sentence removal
6. Importance ranking
7. Redundancy-aware extractive compression
8. Optional BART abstractive compression
9. Final prompt assembly and token budget enforcement

## Run backend

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Run frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend expects the API at `http://localhost:8000`.

## Optional dependencies

The core optimizer works without heavyweight ML downloads.

For BART:
```bash
pip install transformers torch
```

For scanned PDF OCR:
```bash
pip install pytesseract pillow
```

Then configure:
```env
ENABLE_BART=true
ENABLE_OCR=true
```

## API

`POST /api/v1/optimize`

Multipart fields:
- `file`: `.txt` or `.pdf`
- `query`: optional user task/query
- `target_reduction`: optional 0.0–0.9

`GET /api/v1/health`

`GET /api/v1/history`

## Project alignment

The architecture follows the TokenFlow project concept: input processing, extraction, knowledge fusion, prompt optimization, LLM interface, evaluation, data logging and response assembly. The optimizer adds a conservative, information-aware compression strategy so that direct stopword removal does not damage negation or meaning.
