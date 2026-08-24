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
        -> Florence image description fallback

Math handling:

    1. OCR is attempted first.

    2. If OCR indicates mathematical content:
        -> Pix2Tex is invoked.

    3. Pix2Tex tries the COMPLETE IMAGE first.

    4. If valid mathematical LaTeX is found:
        -> LaTeX takes priority over OCR.
        -> OCR garbage is discarded.
        -> Florence description is discarded.

    5. Region detection is used as a fallback.

This prevents output such as:

    2 33 n Li = l 2
    Image description: mathematical equation...

when the correct output should be:

    Equation:
    \\sum_{i=1}^{n} i^3 =
    \\left(\\frac{n(n+1)}{2}\\right)^2
"""

from __future__ import annotations

import io
import logging
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

from app.services.math_extractor import (
    MathExtractor,
)


logger = logging.getLogger(
    __name__
)


class InputExtractor:

    # ======================================================
    # SUPPORTED FILE TYPES
    # ======================================================

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

    # ======================================================
    # PDF / OCR SETTINGS
    # ======================================================

    MIN_TYPED_CHARS_PER_PAGE = 20

    DEFAULT_OCR_RENDER_DPI = 120

    VECTOR_RENDER_DPI = 180

    # ======================================================
    # MATH SETTINGS
    # ======================================================

    MIN_EQUATION_WIDTH = 30
    MIN_EQUATION_HEIGHT = 15

    STATUS_BAR_RE = re.compile(
        r"^\d{1,2}[.:*]\d{2}\b"
    )

    # ======================================================
    # INITIALIZATION
    # ======================================================

    def __init__(
        self,
    ) -> None:

        self._handwriting = (
            HandwritingRecognizer()
        )

        self._printed_ocr = (
            PrintedTextRecognizer()
        )

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

        # Cheap constructor.
        # Pix2Tex remains lazy-loaded.
        self._math_extractor = (
            MathExtractor()
        )

    # ======================================================
    # PUBLIC ENTRY POINT
    # ======================================================

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

        # --------------------------------------------------
        # TXT
        # --------------------------------------------------

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

        # --------------------------------------------------
        # IMAGE
        # --------------------------------------------------

        if ext in self.SUPPORTED_IMAGE:

            return self._extract_image(
                content
            )

        # --------------------------------------------------
        # PDF
        # --------------------------------------------------

        return self._extract_pdf(
            content
        )

    # ======================================================
    # EQUATION EXTRACTION
    # ======================================================

    def _detect_equations(
        self,
        image: Image.Image,
    ) -> list[str]:
        """
        Extract validated mathematical equations.

        Full image is attempted first.

        If full-image output is invalid, region-based extraction
        is attempted.
        """

        try:

            results = (
                self._math_extractor.extract(
                    image
                )
            )

        except Exception as exc:

            logger.warning(
                "Math extraction failed: %s",
                exc,
            )

            return []

        equations: list[str] = []

        seen: set[str] = set()

        for result in results:

            latex = (
                result.latex
                .strip()
            )

            if not self._looks_like_math(
                latex,
                result.bbox,
                image.size,
            ):
                continue

            equation = (
                f"Equation: {latex}"
            )

            if equation not in seen:

                seen.add(
                    equation
                )

                equations.append(
                    equation
                )

        return equations

    # ======================================================
    # MATH VALIDATION
    # ======================================================

    @classmethod
    def _looks_like_math(
        cls,
        latex: str,
        bbox,
        image_size,
    ) -> bool:
        """
        Validate Pix2Tex output.

        Rejects:

            - empty output
            - tiny crops
            - pure numbers
            - normal text without mathematical structure
            - excessive repetition
            - implausibly long output
        """

        if not latex:
            return False

        latex = latex.strip()

        if len(latex) < 3:
            return False

        if len(latex) > 500:
            return False

        try:

            x0, y0, x1, y1 = bbox

        except Exception:

            return False

        width = x1 - x0
        height = y1 - y0

        if (
            width < cls.MIN_EQUATION_WIDTH
            or height < cls.MIN_EQUATION_HEIGHT
        ):
            return False

        # Remove LaTeX commands for basic validation.
        core = re.sub(
            r"\\[a-zA-Z]+",
            "",
            latex,
        )

        core = re.sub(
            r"[{}\[\]()]",
            "",
            core,
        )

        core = core.strip()

        # Reject only bare numbers.
        if re.fullmatch(
            r"\d+",
            core,
        ):
            return False

        math_patterns = (
            r"\\frac",
            r"\\sum",
            r"\\int",
            r"\\sqrt",
            r"\\prod",
            r"\\lim",
            r"\\left",
            r"\\right",
            r"\^",
            r"_",
            r"=",
            r"[+\-*/]",
            r"[∑∫√]",
            r"\\alpha",
            r"\\beta",
            r"\\gamma",
            r"\\theta",
            r"\\pi",
            r"\\sigma",
            r"\\mu",
        )

        if not any(
            re.search(
                pattern,
                latex,
            )
            for pattern in math_patterns
        ):
            return False

        if not cls._brackets_balanced(
            latex
        ):
            return False

        if cls._has_excessive_repetition(
            latex
        ):
            return False

        return True

    @staticmethod
    def _brackets_balanced(
        latex: str,
    ) -> bool:

        stack: list[str] = []

        pairs = {
            "}": "{",
            ")": "(",
            "]": "[",
        }

        openings = set(
            pairs.values()
        )

        for char in latex:

            if char in openings:

                stack.append(
                    char
                )

            elif char in pairs:

                if (
                    not stack
                    or stack[-1]
                    != pairs[char]
                ):
                    return False

                stack.pop()

        return not stack

    @staticmethod
    def _has_excessive_repetition(
        text: str,
        min_chunk_length: int = 8,
        min_repeats: int = 3,
    ) -> bool:

        if (
            len(text)
            < min_chunk_length
        ):
            return False

        max_chunk = min(
            40,
            len(text) // min_repeats,
        )

        for size in range(
            min_chunk_length,
            max_chunk + 1,
        ):

            seen: dict[
                str,
                int,
            ] = {}

            for index in range(
                0,
                len(text) - size + 1,
            ):

                chunk = text[
                    index:index + size
                ]

                seen[chunk] = (
                    seen.get(
                        chunk,
                        0,
                    )
                    + 1
                )

                if (
                    seen[chunk]
                    >= min_repeats
                ):
                    return True

        return False

    # ======================================================
    # MATH SIGNAL DETECTION
    # ======================================================

    @staticmethod
    def _contains_math_signal(
        text: str,
    ) -> bool:
        """
        Cheap mathematical signal detection.

        This runs BEFORE Pix2Tex so the Pix2Tex model is not
        loaded for ordinary images.
        """

        if not text:
            return False

        return bool(
            re.search(
                r"(?:"
                r"="
                r"|\^"
                r"|_"
                r"|∑"
                r"|∫"
                r"|√"
                r"|≈"
                r"|≤"
                r"|≥"
                r"|≠"
                r"|±"
                r"|×"
                r"|÷"
                r"|∞"
                r"|\\frac"
                r"|\\sum"
                r"|\\int"
                r"|\\sqrt"
                r"|\b(?:"
                r"cosine|"
                r"similarity|"
                r"softmax|"
                r"logits?|"
                r"probabilit(?:y|ies)|"
                r"formula|"
                r"equation"
                r")\b"
                r")",
                text,
                re.IGNORECASE,
            )
        )

    # ======================================================
    # MATH CANDIDATE DETECTION FOR PDF
    # ======================================================

    @staticmethod
    def _is_math_candidate_text(
        text: str,
    ) -> bool:

        text = text.strip()

        if (
            len(text) < 3
            or len(text) > 350
        ):
            return False

        strong_symbols = sum(
            char
            in "=^_∑∫√≈≤≥≠±×÷∞⋅·|"
            for char in text
        )

        has_digit = bool(
            re.search(
                r"\d",
                text,
            )
        )

        has_math_word = bool(
            re.search(
                r"\b("
                r"cosine|"
                r"similarity|"
                r"softmax|"
                r"logit|"
                r"probability|"
                r"formula|"
                r"equation|"
                r"average|"
                r"sum|"
                r"score"
                r")\b",
                text,
                re.IGNORECASE,
            )
        )

        if (
            "=" in text
            and (
                has_digit
                or strong_symbols >= 2
            )
        ):
            return True

        if (
            strong_symbols >= 3
            and has_digit
        ):
            return True

        if (
            re.search(
                r"[≤≥≈≠±×÷∞→]",
                text,
            )
            and has_digit
        ):
            return True

        if (
            has_math_word
            and strong_symbols >= 1
        ):
            return True

        return False

    @classmethod
    def _remove_math_candidate_lines(
        cls,
        text: str,
    ) -> str:

        if not text:
            return text

        kept = [
            line
            for line in text.splitlines()
            if not cls._is_math_candidate_text(
                line
            )
        ]

        return (
            "\n"
            .join(kept)
            .strip()
        )

    # ======================================================
    # OCR NOISE FILTER
    # ======================================================

    def _is_garbage_ocr_line(
        self,
        line: str,
    ) -> bool:

        stripped = line.strip()

        if not stripped:
            return False

        tokens = stripped.split()

        if not tokens:
            return False

        total_chars = len(
            stripped
        )

        if total_chars == 0:
            return False

        alnum_ratio = (
            sum(
                1
                for char in stripped
                if char.isalnum()
            )
            / total_chars
        )

        real_word_tokens = sum(
            1
            for token in tokens
            if re.fullmatch(
                r"[A-Za-z]{3,}",
                token.strip(
                    ".,;:!?\'\""
                ),
            )
        )

        real_word_ratio = (
            real_word_tokens
            / len(tokens)
        )

        if self.STATUS_BAR_RE.match(
            stripped
        ):
            return True

        if (
            len(stripped) <= 40
            and real_word_ratio < 0.35
            and alnum_ratio < 0.60
        ):
            return True

        return False

    def _strip_ocr_noise(
        self,
        text: str,
    ) -> str:

        if not text:
            return text

        kept = [
            line
            for line in text.split("\n")
            if not self._is_garbage_ocr_line(
                line
            )
        ]

        return "\n".join(
            kept
        )

    # ======================================================
    # IMAGE EXTRACTION
    # ======================================================

    def _extract_image(
        self,
        content: bytes,
    ):
        """
        Extract useful information from an image.

        Priority:

            VALID MATH
                >
            OCR TEXT
                >
            IMAGE DESCRIPTION

        Therefore, when Pix2Tex successfully extracts an equation,
        garbage OCR and Florence descriptions are discarded.
        """

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
                "Could not process this image: "
                f"{exc}"
            ) from exc

        # --------------------------------------------------
        # CLEAN OCR
        # --------------------------------------------------

        text = (
            self._clean_ocr_output(
                result.text or ""
            )
        )

        text = (
            self._strip_ocr_noise(
                text
            )
        )

        description = (
            self._clean_ocr_output(
                result.description or ""
            )
        )

        # --------------------------------------------------
        # CONDITIONAL MATH EXTRACTION
        # --------------------------------------------------

        equation_lines: list[
            str
        ] = []

        if (
            text
            and self._contains_math_signal(
                text
            )
        ):

            logger.info(
                "[MATH] Mathematical signal detected "
                "in image OCR."
            )

            try:

                pil_image = (
                    Image.open(
                        io.BytesIO(
                            content
                        )
                    )
                    .convert("RGB")
                )

                equation_lines = (
                    self._detect_equations(
                        pil_image
                    )
                )

                logger.info(
                    "[MATH] Found %d validated equation(s).",
                    len(equation_lines),
                )

            except Exception as exc:

                logger.warning(
                    "[MATH] Equation extraction failed: %s",
                    exc,
                )

                equation_lines = []

        else:

            logger.info(
                "[MATH] No mathematical signal detected. "
                "Skipping Pix2Tex."
            )

        # ==================================================
        # PRIORITY 1: VALID MATHEMATICAL OUTPUT
        # ==================================================

        if equation_lines:

            math_text = (
                "\n\n"
                .join(
                    equation_lines
                )
            )

            meta = dict(
                result.meta or {}
            )

            meta.update(
                {
                    "confidence": (
                        result.confidence
                    ),
                    "has_text": True,
                    "has_description": False,
                    "math_detected": True,
                    "math_source": "pix2tex",
                }
            )

            return (
                math_text,
                {
                    "source_type": "math",
                    "confidence": (
                        result.confidence
                    ),
                    "has_text": True,
                    "has_description": False,
                    "description": None,
                    "meta": meta,
                    "math_extraction_used": True,
                    "equations_found": len(
                        equation_lines
                    ),
                },
            )

        # ==================================================
        # PRIORITY 2: NORMAL OCR TEXT
        # ==================================================

        if (
            result.has_text
            and text
        ):

            meta = dict(
                result.meta or {}
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

            combined_text = text

            # Only append image description for normal
            # non-mathematical images.
            if description:

                combined_text = (
                    f"{combined_text}\n\n"
                    "Image description: "
                    f"{description}"
                )

            return (
                combined_text,
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
                    "math_extraction_used": False,
                    "equations_found": 0,
                },
            )

        # ==================================================
        # PRIORITY 3: IMAGE DESCRIPTION
        # ==================================================

        if description:

            meta = dict(
                result.meta or {}
            )

            meta.update(
                {
                    "confidence": (
                        result.confidence
                    ),
                    "has_text": False,
                    "has_description": True,
                    "description_source": (
                        "Florence"
                    ),
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
                    "math_extraction_used": False,
                    "equations_found": 0,
                },
            )

        # ==================================================
        # NOTHING EXTRACTED
        # ==================================================

        raise ValueError(
            result.description
            or (
                "No reliable text or image "
                "description could be extracted."
            )
        )

    # ======================================================
    # PDF EQUATION EXTRACTION
    # ======================================================

    def _extract_page_equations(
        self,
        page,
    ) -> list[str]:
        """
        Extract equations from typed PDFs.

        Native PDF text is used first to identify candidate lines.
        Only those candidate regions are rendered and sent to Pix2Tex.
        """

        try:

            data = page.get_text(
                "dict"
            )

        except Exception:

            return []

        candidates = []

        for block in data.get(
            "blocks",
            [],
        ):

            if block.get("type") != 0:
                continue

            for line in block.get(
                "lines",
                [],
            ):

                spans = line.get(
                    "spans",
                    [],
                )

                text = (
                    "".join(
                        span.get(
                            "text",
                            "",
                        )
                        for span in spans
                    )
                    .strip()
                )

                bbox = line.get(
                    "bbox"
                )

                if (
                    text
                    and bbox
                    and self._is_math_candidate_text(
                        text
                    )
                ):

                    candidates.append(
                        (
                            bbox,
                            text,
                        )
                    )

        if not candidates:
            return []

        results: list[str] = []
        seen: set[str] = set()

        for bbox, _text in candidates[:12]:

            rect = pymupdf.Rect(
                bbox
            )

            rect.x0 -= 8
            rect.y0 -= 8
            rect.x1 += 8
            rect.y1 += 8

            rect &= page.rect

            if (
                rect.width < 30
                or rect.height < 15
            ):
                continue

            try:

                pixmap = (
                    page.get_pixmap(
                        dpi=(
                            self.VECTOR_RENDER_DPI
                        ),
                        clip=rect,
                        alpha=False,
                    )
                )

                image = (
                    Image.open(
                        io.BytesIO(
                            pixmap.tobytes(
                                "png"
                            )
                        )
                    )
                    .convert("RGB")
                )

                equations = (
                    self._detect_equations(
                        image
                    )
                )

            except Exception as exc:

                logger.warning(
                    "PDF equation extraction failed: %s",
                    exc,
                )

                continue

            for equation in equations:

                if (
                    equation
                    and equation not in seen
                ):

                    seen.add(
                        equation
                    )

                    results.append(
                        equation
                    )

        return results

    # ======================================================
    # PDF EXTRACTION
    # ======================================================

    def _extract_pdf(
        self,
        content: bytes,
    ):

        try:

            document = pymupdf.open(
                stream=content,
                filetype="pdf",
            )

        except Exception as exc:

            raise ValueError(
                f"Could not open PDF: {exc}"
            ) from exc

        pages: list[str] = []

        typed_pages = 0
        ocr_pages = 0
        blank_pages = 0
        equations_found = 0

        total_pages = len(
            document
        )

        try:

            for page_number, page in enumerate(
                document,
                start=1,
            ):

                # ==========================================
                # NATIVE TEXT
                # ==========================================

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

                    page_equations = (
                        self._extract_page_equations(
                            page
                        )
                    )

                    if page_equations:

                        cleaned_without_math = (
                            self._remove_math_candidate_lines(
                                cleaned_text
                            )
                        )

                        page_text = (
                            f"[Page {page_number}]\n"
                            f"{cleaned_without_math}"
                        )

                        if page_equations:

                            page_text = (
                                page_text.rstrip()
                                + "\n\n"
                                + "\n".join(
                                    page_equations
                                )
                            )

                        equations_found += len(
                            page_equations
                        )

                    else:

                        page_text = (
                            f"[Page {page_number}]\n"
                            f"{cleaned_text}"
                        )

                    pages.append(
                        page_text
                    )

                    typed_pages += 1

                    continue

                # ==========================================
                # CHECK EMBEDDED IMAGE
                # ==========================================

                has_embedded_image = bool(
                    page.get_images(
                        full=True
                    )
                )

                if not has_embedded_image:

                    blank_pages += 1

                    continue

                # ==========================================
                # RENDER PAGE
                # ==========================================

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
                        "Could not render PDF "
                        f"page {page_number}: "
                        f"{exc}"
                    ) from exc

                # ==========================================
                # OCR
                # ==========================================

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
                        "Could not OCR PDF "
                        f"page {page_number}: "
                        f"{exc}"
                    ) from exc

                ocr_text = (
                    self._clean_ocr_output(
                        ocr_text
                    )
                )

                ocr_text = (
                    self._strip_ocr_noise(
                        ocr_text
                    )
                )

                # ==========================================
                # CONDITIONAL MATH
                # ==========================================

                equation_lines: list[
                    str
                ] = []

                if (
                    ocr_text
                    and self._contains_math_signal(
                        ocr_text
                    )
                ):

                    logger.info(
                        "[MATH] PDF page %d contains "
                        "a mathematical signal.",
                        page_number,
                    )

                    equation_lines = (
                        self._detect_equations(
                            page_image
                        )
                    )

                if equation_lines:

                    ocr_without_math = (
                        self._remove_math_candidate_lines(
                            ocr_text
                        )
                    )

                    ocr_text = (
                        ocr_without_math.rstrip()
                    )

                    if ocr_text:

                        ocr_text += "\n\n"

                    ocr_text += "\n".join(
                        equation_lines
                    )

                    equations_found += len(
                        equation_lines
                    )

                # ==========================================
                # PRESERVE PAGE
                # ==========================================

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

            document.close()

        # ==================================================
        # ASSEMBLE PDF
        # ==================================================

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
                "math_extraction_used": (
                    equations_found > 0
                ),
                "equations_found": (
                    equations_found
                ),
            },
        )

    # ======================================================
    # OCR DPI
    # ======================================================

    @classmethod
    def _get_ocr_dpi(
        cls,
    ) -> int:

        value = os.environ.get(
            "OCR_RENDER_DPI"
        )

        if value:

            try:

                dpi = int(
                    value
                )

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

    # ======================================================
    # CLEAN OCR OUTPUT
    # ======================================================

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

    # ======================================================
    # PDF PAGE NUMBER NOISE
    # ======================================================

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

        # Remove isolated page numbers.
        text = re.sub(
            r"\n\s*\d+\s*\n",
            "\n",
            text,
        )

        return text