"""
TokenFlow Input Extraction Service

Supported:

    TXT
    PDF
    PNG
    JPG
    JPEG

Image processing:

    Printed image
        -> EasyOCR

    Handwritten image
        -> TrOCR
        -> EasyOCR fallback

    Photo
        -> EasyOCR attempt
        -> BLIP image description fallback

    Mixed/contextual
        -> EasyOCR
        -> TrOCR where appropriate
        -> BLIP fallback

The extractor preserves image descriptions as usable textual
input so photographs can continue through TokenFlow's normal
optimization pipeline.
"""

from __future__ import annotations

import io
import os
import re
from pathlib import Path

import pymupdf
from PIL import Image

from app.services.handwriting import (
    HandwritingRecognizer,
)

from app.services.printed_ocr import (
    PrintedTextRecognizer,
)

from app.services.imageprocessing import (
    ImageProcessor,
)


class InputExtractor:

    SUPPORTED_TEXT = {
        ".txt",
    }

    SUPPORTED_PDF = {
        ".pdf",
    }

    SUPPORTED_IMAGE = {
        ".png",
        ".jpg",
        ".jpeg",
    }

    SUPPORTED = (
        SUPPORTED_TEXT
        | SUPPORTED_PDF
        | SUPPORTED_IMAGE
    )

    MIN_TYPED_CHARS_PER_PAGE = 20

    DEFAULT_OCR_RENDER_DPI = 150

    def __init__(self) -> None:
        """
        Create lightweight service objects.

        Heavy OCR/captioning models remain lazy-loaded.
        """

        # These objects should not load the actual models
        # unless their recognize methods are called.

        self._handwriting = (
            HandwritingRecognizer()
        )

        self._printed_ocr = (
            PrintedTextRecognizer()
        )

        # IMPORTANT:
        #
        # No BLIP object is created here.
        #
        # ImageProcessor creates PhotoCaptioner only
        # when captioning becomes necessary.

        self._image_processor = (
            ImageProcessor(
                handwriting_recognizer=(
                    self._handwriting
                ),
                printed_recognizer=(
                    self._printed_ocr
                ),
            )
        )

    # ==========================================================
    # PUBLIC ENTRY POINT
    # ==========================================================

    def extract(
        self,
        filename: str,
        content: bytes,
    ):

        ext = (
            Path(filename)
            .suffix
            .lower()
        )

        if ext not in self.SUPPORTED:

            raise ValueError(
                "Only .txt, .pdf, and "
                ".png/.jpg/.jpeg files "
                "are currently supported."
            )

        # ======================================================
        # TXT
        # ======================================================

        if ext in self.SUPPORTED_TEXT:

            text = content.decode(
                "utf-8",
                errors="replace",
            )

            return (
                text,
                {
                    "source_type": "text",
                },
            )

        # ======================================================
        # IMAGE
        # ======================================================

        if ext in self.SUPPORTED_IMAGE:

            return self._extract_image(
                content
            )

        # ======================================================
        # PDF
        # ======================================================

        return self._extract_pdf(
            content
        )

    # ==========================================================
    # IMAGE EXTRACTION
    # ==========================================================

    def _extract_image(
        self,
        content: bytes,
    ):

        try:

            result = (
                self._image_processor.process(
                    content
                )
            )

        except RuntimeError as exc:

            raise ValueError(
                str(exc)
            ) from exc

        except Exception as exc:

            raise ValueError(
                f"Could not process this image: "
                f"{exc}"
            ) from exc

        text = (
            self._clean_ocr_output(
                result.text
            )
        )

        description = (
            self._clean_ocr_output(
                result.description
                or ""
            )
        )

        # ======================================================
        # CASE 1:
        # OCR successfully extracted text
        # ======================================================

        if (
            result.has_text
            and text
        ):

            meta = (
                dict(
                    result.meta
                    or {}
                )
            )

            meta.update(
                {
                    "confidence": (
                        result.confidence
                    ),
                    "has_text": True,
                    "has_description": bool(
                        description
                    ),
                }
            )

            return (
                text,
                {
                    "source_type": (
                        result.source_type
                    ),
                    "confidence": (
                        result.confidence
                    ),
                    "has_text": True,
                    "has_description": bool(
                        description
                    ),
                    "description": (
                        description
                        or result.description
                    ),
                    "meta": meta,
                },
            )

        # ======================================================
        # CASE 2:
        # No OCR text, but BLIP produced a description
        # ======================================================

        if description:

            meta = (
                dict(
                    result.meta
                    or {}
                )
            )

            meta.update(
                {
                    "confidence": (
                        result.confidence
                    ),
                    "has_text": False,
                    "has_description": True,
                    "description_source": "BLIP",
                }
            )

            # IMPORTANT:
            #
            # The caption becomes the textual representation
            # passed to the downstream TokenFlow optimizer.
            #
            # Example:
            #
            # Image:
            #     cat sitting on a chair
            #
            # Extracted representation:
            #
            #     "a cat sitting on a chair"
            #
            # This means the normal compression pipeline can
            # still process the image-derived information.

            return (
                description,
                {
                    "source_type": (
                        result.source_type
                    ),
                    "confidence": (
                        result.confidence
                    ),
                    "has_text": False,
                    "has_description": True,
                    "description": description,
                    "meta": meta,
                },
            )

        # ======================================================
        # CASE 3:
        # Nothing useful was extracted
        # ======================================================

        raise ValueError(
            result.description
            or (
                "No reliable text or image "
                "description could be extracted."
            )
        )

    # ==========================================================
    # PDF EXTRACTION
    # ==========================================================

    def _extract_pdf(
        self,
        content: bytes,
    ):

        try:

            doc = pymupdf.open(
                stream=content,
                filetype="pdf",
            )

        except Exception as exc:

            raise ValueError(
                f"Could not open PDF: {exc}"
            ) from exc

        pages = []

        typed_pages = 0
        ocr_pages = 0
        blank_pages = 0

        total_pages = len(doc)

        try:

            for (
                page_number,
                page
            ) in enumerate(
                doc,
                start=1,
            ):

                # ==================================================
                # NATIVE TEXT
                # ==================================================

                raw_text = (
                    page.get_text(
                        "text"
                    )
                    or ""
                )

                cleaned_text = (
                    self._remove_repeated_page_noise(
                        raw_text
                    )
                    .strip()
                )

                if (
                    len(cleaned_text)
                    >= self.MIN_TYPED_CHARS_PER_PAGE
                ):

                    pages.append(
                        f"[Page {page_number}]\n"
                        f"{cleaned_text}"
                    )

                    typed_pages += 1

                    continue

                # ==================================================
                # CHECK EMBEDDED IMAGE
                # ==================================================

                has_embedded_image = bool(
                    page.get_images(
                        full=True
                    )
                )

                if not has_embedded_image:

                    blank_pages += 1

                    continue

                # ==================================================
                # RENDER PAGE
                # ==================================================

                dpi = (
                    self._get_ocr_dpi()
                )

                try:

                    pixmap = (
                        page.get_pixmap(
                            dpi=dpi,
                            alpha=False,
                        )
                    )

                    page_image = (
                        Image.open(
                            io.BytesIO(
                                pixmap.tobytes(
                                    "png"
                                )
                            )
                        )
                        .convert("RGB")
                    )

                except Exception as exc:

                    raise ValueError(
                        f"Could not render PDF "
                        f"page {page_number}: "
                        f"{exc}"
                    ) from exc

                # ==================================================
                # OCR
                # ==================================================

                try:

                    ocr_text = (
                        self._printed_ocr
                        .recognize(
                            page_image
                        )
                    )

                except RuntimeError as exc:

                    raise ValueError(
                        str(exc)
                    ) from exc

                except Exception as exc:

                    raise ValueError(
                        f"Could not OCR PDF "
                        f"page {page_number}: "
                        f"{exc}"
                    ) from exc

                ocr_text = (
                    self._clean_ocr_output(
                        ocr_text
                    )
                )

                # ==================================================
                # PRESERVE PAGE
                # ==================================================

                if ocr_text:

                    pages.append(
                        f"[Page {page_number} "
                        f"- scanned, OCR]\n"
                        f"{ocr_text}"
                    )

                    ocr_pages += 1

                else:

                    blank_pages += 1

        finally:

            doc.close()

        # ==========================================================
        # ASSEMBLE
        # ==========================================================

        text = (
            "\n\n"
            .join(pages)
            .strip()
        )

        if not text:

            raise ValueError(
                "No extractable or recognizable "
                "text was found in this PDF."
            )

        return (
            text,
            {
                "source_type": "pdf",
                "page_count": total_pages,
                "processed_pages": len(
                    pages
                ),
                "typed_pages": typed_pages,
                "ocr_pages": ocr_pages,
                "blank_pages": blank_pages,
            },
        )

    # ==========================================================
    # OCR DPI
    # ==========================================================

    @classmethod
    def _get_ocr_dpi(
        cls,
    ) -> int:

        value = os.environ.get(
            "OCR_RENDER_DPI"
        )

        if value:

            try:

                dpi = int(value)

                if (
                    100
                    <= dpi
                    <= 300
                ):

                    return dpi

            except ValueError:

                pass

        return (
            cls.DEFAULT_OCR_RENDER_DPI
        )

    # ==========================================================
    # CLEAN OCR OUTPUT
    # ==========================================================

    @staticmethod
    def _clean_ocr_output(
        text: str,
    ) -> str:

        if not text:

            return ""

        text = (
            text
            .replace(
                "\r\n",
                "\n",
            )
            .replace(
                "\r",
                "\n",
            )
        )

        text = "\n".join(
            line.rstrip()
            for line in text.split(
                "\n"
            )
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\s+([,.;:!?])",
            r"\1",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    # ==========================================================
    # PDF PAGE NUMBER NOISE
    # ==========================================================

    @staticmethod
    def _remove_repeated_page_noise(
        text: str,
    ) -> str:

        if not text:

            return ""

        text = (
            text
            .replace(
                "\r\n",
                "\n",
            )
            .replace(
                "\r",
                "\n",
            )
        )

        text = re.sub(
            r"\n\s*\d+\s*\n",
            "\n",
            text,
        )

        return text