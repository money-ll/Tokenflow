"""
Printed-text OCR fallback (Image Pipeline - Path A).

Used when a PDF page has no usable embedded text layer but does contain
a raster image (i.e. it was scanned rather than typed/exported). The
page is rendered to an image and passed through EasyOCR, which is
built on the same torch runtime already required for handwriting
recognition, so no separate system-level OCR engine (e.g. the Tesseract
binary) needs to be installed.

Loaded lazily, same pattern as HandwritingRecognizer: importing this
module, or processing typed PDFs/TXT files, never requires easyocr or
its models to be present.
"""

import numpy as np
from PIL import Image


class PrintedTextRecognizer:
    LANGUAGES = ["en"]

    def __init__(self):
        self._reader = None

    def _ensure_loaded(self):
        if self._reader is not None:
            return

        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError(
                "Scanned-PDF OCR fallback requires 'easyocr'. "
                "Install it with: pip install easyocr"
            ) from exc

        self._reader = easyocr.Reader(self.LANGUAGES, gpu=self._has_gpu())

    @staticmethod
    def _has_gpu():
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def recognize(self, image: Image.Image) -> str:
        self._ensure_loaded()

        array = np.array(image.convert("RGB"))
        results = self._reader.readtext(array, detail=0, paragraph=True)
        lines = [line.strip() for line in results if line and line.strip()]
        return "\n".join(lines)
