import os
import logging
import warnings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router, pipeline

logger = logging.getLogger("uvicorn")

# Harmless: EasyOCR always requests pin_memory=True on its internal
# DataLoader regardless of whether a GPU is present. On CPU-only machines
# it's a no-op, so silence just this specific noisy warning.
warnings.filterwarnings(
    "ignore",
    message=".*pin_memory.*no accelerator is found.*",
    category=UserWarning,
)

app = FastAPI(
    title="TokenFlow API",
    version="1.0.0",
    description="Semantic-aware multimodal token optimization middleware."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
def root():
    return {"name": "TokenFlow", "status": "running", "docs": "/docs"}

@app.on_event("startup")
def warm_up_models():
    """
    Load the heavy OCR/handwriting models once at server startup instead of
    on a user's first request. Set WARM_UP_MODELS=false to skip this (e.g.
    while iterating on code with --reload, where you don't want every
    reload to pay the load cost).
    """
    if os.environ.get("WARM_UP_MODELS", "true").lower() == "false":
        logger.info("Skipping model warm-up (WARM_UP_MODELS=false)")
        return

    extractor = pipeline.extractor
    logger.info("Warming up handwriting model (this can take a while on first run)...")
    try:
        extractor._handwriting._ensure_loaded()
        logger.info("Handwriting model ready.")
    except Exception as exc:
        logger.warning(f"Handwriting model warm-up skipped: {exc}")

    logger.info("Warming up printed-text OCR model...")
    try:
        extractor._printed_ocr._ensure_loaded()
        logger.info("Printed-text OCR model ready.")
    except Exception as exc:
        logger.warning(f"Printed-text OCR warm-up skipped: {exc}")
