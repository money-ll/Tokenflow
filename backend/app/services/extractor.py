from pathlib import Path
import io
import re
import fitz
from PIL import Image

from app.services.handwriting import HandwritingRecognizer
from app.services.printed_ocr import PrintedTextRecognizer

class InputExtractor:
    SUPPORTED_TEXT = {".txt"}
    SUPPORTED_PDF = {".pdf"}
    SUPPORTED_IMAGE = {".png", ".jpg", ".jpeg"}
    SUPPORTED = SUPPORTED_TEXT | SUPPORTED_PDF | SUPPORTED_IMAGE

    # A page with fewer real characters than this is treated as having
    # no usable embedded text layer. It's then classified as "scanned"
    # (if it contains a raster image worth OCR-ing) or "blank" (if not).
    MIN_TYPED_CHARS_PER_PAGE = 20
    OCR_RENDER_DPI = 200

    def __init__(self):
        self._handwriting = HandwritingRecognizer()
        self._printed_ocr = PrintedTextRecognizer()

    def extract(self, filename: str, content: bytes):
        ext = Path(filename).suffix.lower()

        if ext not in self.SUPPORTED:
            raise ValueError(
                "Only .txt, .pdf, and .png/.jpg/.jpeg (handwritten) "
                "files are currently supported."
            )

        if ext in self.SUPPORTED_TEXT:
            return (
                content.decode("utf-8", errors="replace"),
                {"source_type": "text"},
            )

        if ext in self.SUPPORTED_IMAGE:
            return self._extract_handwritten(content)

        doc = fitz.open(stream=content, filetype="pdf")
        pages = []
        typed_pages = 0
        ocr_pages = 0
        blank_pages = 0

        for page_number, page in enumerate(doc, start=1):
            raw_text = page.get_text("text") or ""
            cleaned = self._remove_repeated_page_noise(raw_text).strip()

            if len(cleaned) >= self.MIN_TYPED_CHARS_PER_PAGE:
                # Real, machine-readable text layer -> typed page.
                pages.append(f"[Page {page_number}]\n{cleaned}")
                typed_pages += 1
                continue

            has_embedded_image = len(page.get_images(full=True)) > 0
            if not has_embedded_image:
                # No text and no image -> genuinely blank page, nothing
                # to gain from OCR-ing it.
                blank_pages += 1
                continue

            # Little/no text layer + a raster image -> almost certainly
            # a scanned page. Render it and fall back to printed-text OCR.
            pixmap = page.get_pixmap(dpi=self.OCR_RENDER_DPI)
            page_image = Image.open(io.BytesIO(pixmap.tobytes("png")))

            try:
                ocr_text = self._printed_ocr.recognize(page_image)
            except RuntimeError as exc:
                doc.close()
                raise ValueError(str(exc))

            if ocr_text.strip():
                pages.append(f"[Page {page_number} - scanned, OCR]\n{ocr_text.strip()}")
                ocr_pages += 1
            else:
                blank_pages += 1

        doc.close()

        text = "\n\n".join(pages).strip()

        if not text:
            raise ValueError(
                "No extractable or recognizable text was found in this PDF."
            )

        return text, {
            "source_type": "pdf",
            "page_count": len(pages),
            "typed_pages": typed_pages,
            "ocr_pages": ocr_pages,
            "blank_pages": blank_pages,
        }

    def _extract_handwritten(self, content: bytes):
        try:
            text = self._handwriting.recognize(content)
        except RuntimeError as exc:
            # Missing torch/transformers -> surface as a clean 400, not a 500.
            raise ValueError(str(exc))
        except Exception as exc:
            raise ValueError(f"Could not process this image: {exc}")

        if not text.strip():
            raise ValueError(
                "No handwritten text could be recognized in this image. "
                "Try a clearer photo with good lighting and contrast, "
                "cropped to just the handwritten area."
            )

        return text, {
            "source_type": "handwritten_image",
        }

    def _remove_repeated_page_noise(self, text):
        # Simple newline cleanup; no fragile character-range regex.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return re.sub(r"\n\s*\d+\s*\n", "\n", text)
