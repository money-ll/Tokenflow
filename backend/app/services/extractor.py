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

        # Math extraction is intentionally lazy.  The existing OCR/image
        # path remains unchanged unless a document actually looks like it
        # contains mathematics.
        self._math_extractor = None

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

        # Math extraction is opt-in by a cheap text signal in AUTO mode.
        # This prevents the existing image/OCR path from paying for
        # OpenCV/Pix2Tex on ordinary images.
        math_results = []

        if (
            result.has_text
            and text
            and self._should_try_math(text)
        ):
            math_results = self._extract_math_from_image_bytes(
                content
            )

            if math_results:
                text = self._append_math_equations(
                    text,
                    math_results,
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
                    "math_regions": (
                        self._serialize_math_results(
                            math_results
                        )
                    ),
                    "math_extraction_used": bool(
                        math_results
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
                    "math_regions": (
                        self._serialize_math_results(
                            math_results
                        )
                    ),
                    "math_extraction_used": bool(
                        math_results
                    ),
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
                    "math_regions": [],
                    "math_extraction_used": False,
                }
            )

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
                    "math_regions": [],
                    "math_extraction_used": False,
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
        math_pages = 0
        math_regions_by_page = {}

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
                # NATIVE TEXT -- original fast path is preserved
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

                    # Only render a native-text page when a cheap
                    # heuristic suggests it may contain mathematics.
                    math_results = []

                    if self._should_try_math(
                        cleaned_text
                    ):
                        math_results = (
                            self._extract_math_from_pdf_page(
                                page
                            )
                        )

                    if math_results:
                        # Remove obvious scrambled Unicode math from the
                        # native text layer only after we have a reliable
                        # replacement from MathExtractor.
                        cleaned_text = (
                            self._strip_scrambled_math_lines(
                                cleaned_text
                            )
                        )
                        cleaned_text = (
                            self._append_math_equations(
                                cleaned_text,
                                math_results,
                            )
                        )
                        math_pages += 1
                        math_regions_by_page[
                            page_number
                        ] = self._serialize_math_results(
                            math_results
                        )

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
                # RENDER PAGE -- exactly the existing OCR path
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

                # Reuse the image already rendered for OCR.  No second
                # PDF render is performed for math extraction.
                math_results = []

                if (
                    ocr_text
                    and self._should_try_math(ocr_text)
                ):
                    math_results = (
                        self._extract_math_from_image(
                            page_image
                        )
                    )

                # ==================================================
                # PRESERVE PAGE
                # ==================================================

                if ocr_text:

                    if math_results:
                        ocr_text = (
                            self._strip_scrambled_math_lines(
                                ocr_text
                            )
                        )
                        ocr_text = (
                            self._append_math_equations(
                                ocr_text,
                                math_results,
                            )
                        )
                        math_pages += 1
                        math_regions_by_page[
                            page_number
                        ] = self._serialize_math_results(
                            math_results
                        )

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
                "math_pages": math_pages,
                "math_regions": math_regions_by_page,
                "math_extraction_used": bool(
                    math_regions_by_page
                ),
            },
        )

    # ==========================================================
    # MATH EXTRACTION
    # ==========================================================

    @classmethod
    def _math_mode(cls) -> str:
        mode = os.environ.get(
            "TOKENFLOW_MATH_EXTRACTION",
            "auto",
        ).strip().lower()

        if mode not in {"auto", "always", "off"}:
            return "auto"

        return mode

    @classmethod
    def _should_try_math(
        cls,
        text: str,
    ) -> bool:
        mode = cls._math_mode()

        if mode == "off":
            return False

        if mode == "always":
            return bool(text)

        return cls._looks_like_math_text(text)

    @classmethod
    def _looks_like_math_text(
        cls,
        text: str,
    ) -> bool:
        """
        Very cheap pre-check used to avoid rendering/MathExtractor work
        for ordinary prose.

        This is intentionally conservative.  It looks for either obvious
        Unicode math symbols or equation-like ASCII lines.  It does not
        attempt to recognize equations itself.
        """
        if not text:
            return False

        math_symbol_pattern = re.compile(
            r"[\u2070-\u209F"
            r"\u2200-\u22FF"
            r"\u2A00-\u2AFF"
            r"\U0001D400-\U0001D7FF]"
        )

        equation_pattern = re.compile(
            r"(?<!\w)"
            r"(?:"
            r"[A-Za-z0-9]\s*(?:=|≠|≤|≥|\^)\s*"
            r"[A-Za-z0-9]"
            r"|"
            r"(?:\\frac|\\sqrt|\\sum|\\int|\\alpha|\\beta|\\theta)"
            r")"
        )

        for line in text.splitlines():
            stripped = line.strip()

            if not stripped:
                continue

            non_space = [
                char
                for char in stripped
                if not char.isspace()
            ]

            if not non_space:
                continue

            unicode_math = sum(
                1
                for char in non_space
                if math_symbol_pattern.match(char)
            )

            if unicode_math >= 2:
                return True

            if (
                unicode_math >= 1
                and any(
                    symbol in stripped
                    for symbol in ("=", "≠", "≤", "≥")
                )
            ):
                return True

            if equation_pattern.search(stripped):
                return True

            # Multiple operator symbols on a short line are another
            # inexpensive signal for equations such as "x^2 + y^2 = z^2".
            operator_count = sum(
                1
                for char in stripped
                if char in "=+-*/^"
            )

            if (
                operator_count >= 2
                and len(stripped) <= 160
            ):
                return True

        return False

    def _get_math_extractor(self):
        """
        Import and instantiate MathExtractor only when math processing
        actually becomes necessary.
        """
        if self._math_extractor is None:
            from app.services.math_extractor import MathExtractor

            self._math_extractor = MathExtractor()

        return self._math_extractor

    def _extract_math_from_image(
        self,
        image: Image.Image,
    ):
        try:
            return self._get_math_extractor().extract(
                image
            )
        except Exception:
            # Math is an enhancement.  A missing Pix2Tex install or a
            # conversion failure must never break the existing extractor.
            return []

    def _extract_math_from_image_bytes(
        self,
        content: bytes,
    ):
        try:
            image = (
                Image.open(
                    io.BytesIO(content)
                )
                .convert("RGB")
            )
            return self._extract_math_from_image(
                image
            )
        except Exception:
            return []

    def _extract_math_from_pdf_page(
        self,
        page,
    ):
        """
        Render only a native-text PDF page that already passed the cheap
        math heuristic.  Uses the existing OCR DPI rather than introducing
        a new high-resolution render path.
        """
        try:
            pixmap = page.get_pixmap(
                dpi=self._get_ocr_dpi(),
                alpha=False,
            )

            page_image = (
                Image.open(
                    io.BytesIO(
                        pixmap.tobytes("png")
                    )
                )
                .convert("RGB")
            )

            return self._extract_math_from_image(
                page_image
            )
        except Exception:
            return []

    @staticmethod
    def _serialize_math_results(
        math_results,
    ):
        return [
            {
                "bbox": tuple(result.bbox),
                "latex": result.latex,
                "confidence": result.confidence,
            }
            for result in math_results
        ]

    @staticmethod
    def _append_math_equations(
        text: str,
        math_results,
    ) -> str:
        """
        Append only equations that are not already represented in the
        extracted text.  This avoids duplicating a formula when a PDF's
        native text layer already contains a usable ASCII/LaTeX version.
        """
        existing = re.sub(
            r"\s+",
            "",
            text or "",
        )

        equations = []

        for result in math_results:
            latex = result.latex.strip()

            if not latex:
                continue

            normalized_latex = re.sub(
                r"\s+",
                "",
                latex,
            )

            if normalized_latex and normalized_latex in existing:
                continue

            equations.append(
                f"Equation: {latex}"
            )

        if not equations:
            return text

        equation_block = "\n".join(equations)

        if text:
            return f"{text}\n\n{equation_block}"

        return equation_block

    @staticmethod
    def _strip_scrambled_math_lines(
        text: str,
    ) -> str:
        """
        Remove obvious scrambled Unicode equation lines after a valid
        MathExtractor result exists.  Ordinary prose is kept unchanged.
        """
        if not text:
            return ""

        math_symbol_pattern = re.compile(
            r"[\u2070-\u209F"
            r"\u2200-\u22FF"
            r"\u2A00-\u2AFF"
            r"\U0001D400-\U0001D7FF]"
        )

        kept_lines = []

        for line in text.split("\n"):
            stripped = line.strip()

            if not stripped:
                kept_lines.append(line)
                continue

            non_space_chars = [
                char
                for char in stripped
                if not char.isspace()
            ]

            if not non_space_chars:
                kept_lines.append(line)
                continue

            math_symbol_count = sum(
                1
                for char in non_space_chars
                if math_symbol_pattern.match(char)
            )

            ascii_letter_count = sum(
                1
                for char in non_space_chars
                if char.isascii() and char.isalpha()
            )

            math_ratio = (
                math_symbol_count
                / len(non_space_chars)
            )

            ascii_letter_ratio = (
                ascii_letter_count
                / len(non_space_chars)
            )

            is_scrambled_math = (
                math_ratio >= 0.15
                and ascii_letter_ratio <= 0.25
            )

            if not is_scrambled_math:
                kept_lines.append(line)

        return "\n".join(kept_lines)

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