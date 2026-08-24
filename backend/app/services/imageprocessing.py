"""
backend/app/services/imageprocessing.py
TokenFlow Multimodal Image Processor

Routes PNG/JPG/JPEG images to the appropriate processing path.

Processing strategy:

1. Lightweight visual classification.
2. Blank images are rejected immediately.
3. Strong handwriting candidates:
       TrOCR -> EasyOCR fallback -> BLIP fallback
4. Printed/document-like images:
       EasyOCR -> BLIP fallback
5. Mixed/contextual images:
       EasyOCR -> BLIP fallback
6. Photos:
       EasyOCR once -> BLIP description

Important design principle:

OCR and image captioning solve different problems.

OCR:
    extracts visible text.

BLIP:
    describes visual content.

Therefore a photo with no text is NOT considered a failed image.
It can successfully produce a description.

Heavy models are lazy-loaded.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import (
    Image,
    ImageEnhance,
    ImageFilter,
    ImageOps,
)


# ============================================================
# RESULT
# ============================================================

@dataclass
class ImageProcessResult:

    text: str

    source_type: str

    confidence: float

    has_text: bool

    description: Optional[str] = None

    meta: Optional[Dict[str, Any]] = None


# ============================================================
# IMAGE PROCESSOR
# ============================================================

class ImageProcessor:
    """
    Main PNG/JPG/JPEG processing pipeline.
    """

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    BLANK_INK_RATIO = 0.003

    PHOTO_INK_RATIO = 0.012

    HANDWRITING_VARIANCE = 3200.0

    MIN_USEFUL_CHARS = 5

    # --------------------------------------------------------
    # OCR garbage detection
    # --------------------------------------------------------

    GARBAGE_CHAR_RATIO = 0.50

    def __init__(
        self,
        handwriting_recognizer=None,
        printed_recognizer=None,
        photo_captioner=None,
    ) -> None:

        self._handwriting = (
            handwriting_recognizer
        )

        self._printed = (
            printed_recognizer
        )

        # IMPORTANT:
        #
        # This object itself does not load BLIP.
        #
        # BLIP is loaded only when:
        #
        #     self._caption(...)
        #
        # is called.

        self._photo_captioner = (
            photo_captioner
        )

    # ========================================================
    # PUBLIC API
    # ========================================================

    def process(
        self,
        image_bytes: bytes,
    ) -> ImageProcessResult:

        try:

            image = (
                Image.open(
                    io.BytesIO(image_bytes)
                )
                .convert("RGB")
            )

        except Exception as exc:

            raise ValueError(
                f"Could not open image: {exc}"
            ) from exc

        image_type, scores = (
            self._classify(image)
        )

        # ====================================================
        # BLANK
        # ====================================================

        if image_type == "blank":

            return ImageProcessResult(
                text="",
                source_type="blank",
                confidence=1.0,
                has_text=False,
                description=(
                    "No significant visual "
                    "content detected."
                ),
                meta={
                    "scores": scores,
                    "recognizer": None,
                },
            )

        # ====================================================
        # STRONG HANDWRITING
        # ====================================================

        if image_type == "handwritten":

            result = (
                self._process_handwritten_candidate(
                    image,
                    scores,
                )
            )

            if result is not None:
                return result

            # If both OCR methods failed, use
            # image description as final fallback.

            description = (
                self._safe_caption(
                    image
                )
            )

            if description:

                return ImageProcessResult(
                    text="",
                    source_type="handwritten",
                    confidence=0.70,
                    has_text=False,
                    description=description,
                    meta={
                        "scores": scores,
                        "recognizer": "BLIP",
                        "fallback_from": (
                            "TrOCR + EasyOCR"
                        ),
                    },
                )

            return self._empty_result(
                image_type,
                scores,
            )

        # ====================================================
        # PHOTO
        # ====================================================

        if image_type == "photo":

            # A photo may still contain:
            #
            # signs
            # labels
            # receipts
            # screens
            # posters
            #
            # Therefore give EasyOCR one opportunity.

            text, confidence = (
                self._safe_printed_ocr(
                    image,
                    allow_preprocessed=False,
                    allow_full_resolution_retry=False,
                )
            )

            if self._is_reliable(
                text,
                confidence,
            ):

                description = (
                    self._safe_caption(
                        image
                    )
                )

                return ImageProcessResult(
                    text=text,
                    source_type="photo",
                    confidence=confidence,
                    has_text=True,
                    description=(
                        description
                        or (
                            "Readable text detected "
                            "in a photograph."
                        )
                    ),
                    meta={
                        "scores": scores,
                        "recognizer": (
                            "EasyOCR + BLIP"
                            if description
                            else "EasyOCR"
                        ),
                    },
                )

            # No useful text:
            #
            # This is the important new behavior.
            #
            # Instead of returning:
            #
            #   No reliable text...
            #
            # describe the photograph.

            description = (
                self._safe_caption(
                    image
                )
            )

            if description:

                return ImageProcessResult(
                    text="",
                    source_type="photo",
                    confidence=0.85,
                    has_text=False,
                    description=description,
                    meta={
                        "scores": scores,
                        "recognizer": "BLIP",
                        "easyocr_attempted": True,
                    },
                )

            return self._empty_result(
                image_type,
                scores,
            )

        # ====================================================
        # PRINTED / MIXED / CONTEXTUAL
        # ====================================================

        text, confidence = (
            self._safe_printed_ocr(
                image,
                allow_preprocessed=True,
                allow_full_resolution_retry=True,
            )
        )

        if self._is_reliable(
            text,
            confidence,
        ):

            # OCR succeeded, but the image may still be a
            # designed graphic (text overlaid on a photo)
            # rather than a plain document. Also generate a
            # visual description so both are available -
            # this is best-effort: if captioning fails or
            # is unavailable, we still return the OCR text.

            description = (
                self._safe_caption(
                    image
                )
            )

            return ImageProcessResult(
                text=text,
                source_type=image_type,
                confidence=confidence,
                has_text=True,
                description=(
                    description
                    or (
                        "Text extracted from "
                        "the image."
                    )
                ),
                meta={
                    "scores": scores,
                    "recognizer": (
                        "EasyOCR + BLIP"
                        if description
                        else "EasyOCR"
                    ),
                },
            )

        # ====================================================
        # HANDWRITING FALLBACK
        # ====================================================

        if image_type in {
            "mixed",
            "contextual",
        }:

            handwriting_text, handwriting_conf = (
                self._safe_handwriting(
                    image
                )
            )

            if self._is_reliable(
                handwriting_text,
                handwriting_conf,
            ):

                return ImageProcessResult(
                    text=handwriting_text,
                    source_type="handwritten",
                    confidence=handwriting_conf,
                    has_text=True,
                    description=(
                        "Handwritten text "
                        "detected in the image."
                    ),
                    meta={
                        "scores": scores,
                        "recognizer": "TrOCR",
                        "fallback_from": "EasyOCR",
                    },
                )

        # ====================================================
        # CAPTION FALLBACK
        # ====================================================

        description = (
            self._safe_caption(
                image
            )
        )

        if description:

            return ImageProcessResult(
                text="",
                source_type=image_type,
                confidence=0.80,
                has_text=False,
                description=description,
                meta={
                    "scores": scores,
                    "recognizer": "BLIP",
                    "easyocr_attempted": True,
                },
            )

        # ====================================================
        # NOTHING WORKED
        # ====================================================

        return self._empty_result(
            image_type,
            scores,
        )

    # ========================================================
    # HANDWRITING
    # ========================================================

    def _process_handwritten_candidate(
        self,
        image: Image.Image,
        scores: Dict[str, float],
    ) -> Optional[ImageProcessResult]:

        # ----------------------------------------------------
        # TrOCR first
        # ----------------------------------------------------

        handwriting_text, handwriting_conf = (
            self._safe_handwriting(
                image
            )
        )

        if self._is_reliable(
            handwriting_text,
            handwriting_conf,
        ):

            return ImageProcessResult(
                text=handwriting_text,
                source_type="handwritten",
                confidence=handwriting_conf,
                has_text=True,
                description=(
                    "Handwritten text "
                    "detected in the image."
                ),
                meta={
                    "scores": scores,
                    "recognizer": "TrOCR",
                },
            )

        # ----------------------------------------------------
        # EasyOCR fallback
        # ----------------------------------------------------

        printed_text, printed_conf = (
            self._safe_printed_ocr(
                image,
                allow_preprocessed=True,
                allow_full_resolution_retry=True,
            )
        )

        if self._is_reliable(
            printed_text,
            printed_conf,
        ):

            return ImageProcessResult(
                text=printed_text,
                source_type="printed",
                confidence=printed_conf,
                has_text=True,
                description=(
                    "Printed text detected "
                    "after handwriting OCR fallback."
                ),
                meta={
                    "scores": scores,
                    "recognizer": "EasyOCR",
                    "fallback_from": "TrOCR",
                },
            )

        return None

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    def _classify(
        self,
        image: Image.Image,
    ) -> Tuple[
        str,
        Dict[str, float],
    ]:

        gray = np.asarray(
            image.convert("L"),
            dtype=np.uint8,
        )

        height, width = gray.shape

        if (
            height == 0
            or width == 0
        ):

            return "blank", {
                "ink_ratio": 0.0,
                "mean_variance": 0.0,
                "edge_density": 0.0,
            }

        total_pixels = (
            height * width
        )

        # ----------------------------------------------------
        # Ink ratio
        # ----------------------------------------------------

        ink_ratio = float(
            (
                gray < 160
            ).sum()
            / total_pixels
        )

        # ----------------------------------------------------
        # Local variance
        # ----------------------------------------------------

        block_size = 32

        variances = []

        max_blocks_y = max(
            1,
            height // block_size,
        )

        max_blocks_x = max(
            1,
            width // block_size,
        )

        for by in range(
            max_blocks_y
        ):

            y = (
                by * block_size
            )

            y2 = min(
                y + block_size,
                height,
            )

            for bx in range(
                max_blocks_x
            ):

                x = (
                    bx * block_size
                )

                x2 = min(
                    x + block_size,
                    width,
                )

                patch = gray[
                    y:y2,
                    x:x2,
                ]

                if patch.size:

                    variances.append(
                        float(
                            patch.var()
                        )
                    )

        mean_variance = (
            float(
                np.mean(
                    variances
                )
            )
            if variances
            else 0.0
        )

        # ----------------------------------------------------
        # Edge density
        # ----------------------------------------------------

        if width > 1:

            horizontal_edges = np.abs(
                np.diff(
                    gray.astype(
                        np.float32
                    ),
                    axis=1,
                )
            )

            edge_density = float(
                horizontal_edges.mean()
            )

        else:

            edge_density = 0.0

        scores = {
            "ink_ratio": round(
                ink_ratio,
                4,
            ),
            "mean_variance": round(
                mean_variance,
                1,
            ),
            "edge_density": round(
                edge_density,
                2,
            ),
        }

        # ----------------------------------------------------
        # Blank
        # ----------------------------------------------------

        if (
            ink_ratio
            < self.BLANK_INK_RATIO
        ):

            return "blank", scores

        # ----------------------------------------------------
        # Handwriting
        # ----------------------------------------------------

        handwriting_signal = (
            mean_variance
            >= self.HANDWRITING_VARIANCE
            and
            0.015
            <= ink_ratio
            <= 0.30
        )

        if handwriting_signal:

            return (
                "handwritten",
                scores,
            )

        # ----------------------------------------------------
        # Photo
        #
        # IMPORTANT:
        #
        # We don't require OCR text here.
        #
        # Photographs are valid inputs even when they
        # contain zero readable characters.
        # ----------------------------------------------------

        if (
            ink_ratio
            < self.PHOTO_INK_RATIO
            and
            mean_variance
            < 1200
        ):

            return "photo", scores

        # ----------------------------------------------------
        # Mixed
        # ----------------------------------------------------

        if mean_variance > 2500:

            return "mixed", scores

        # ----------------------------------------------------
        # Printed
        # ----------------------------------------------------

        if ink_ratio >= 0.02:

            return "printed", scores

        # ----------------------------------------------------
        # Contextual
        # ----------------------------------------------------

        return "contextual", scores

    # ========================================================
    # PRINTED OCR
    # ========================================================

    def _safe_printed_ocr(
        self,
        image: Image.Image,
        allow_preprocessed: bool = False,
        allow_full_resolution_retry: bool = False,
    ) -> Tuple[str, float]:

        try:

            self._ensure_printed()

        except Exception:

            return "", 0.0

        # ----------------------------------------------------
        # FIRST PASS
        #
        # Always start with the original image.
        # ----------------------------------------------------

        variants = [image]

        # ----------------------------------------------------
        # OPTIONAL SECOND PASS
        #
        # Only document-like inputs should get the
        # preprocessing pass.
        #
        # Photos/artwork skip it completely.
        # ----------------------------------------------------

        if allow_preprocessed:

            variants.append(
                self._preprocess_for_printed(
                    image
                )
            )

        best_text = ""
        best_confidence = 0.0

        for index, candidate in enumerate(
            variants,
            start=1,
        ):

            try:

                print(
                    f"[OCR] Printed OCR pass "
                    f"{index}/{len(variants)}",
                    flush=True,
                )

                text = (
                    self._printed.recognize(
                        candidate,
                        allow_full_resolution_retry=(
                            allow_full_resolution_retry
                        ),
                    )
                )

            except Exception as exc:

                print(
                    f"[OCR] Printed OCR pass "
                    f"{index} failed: {exc}",
                    flush=True,
                )

                continue

            text = self._clean_text(
                text
            )

            confidence = (
                self._estimate_confidence(
                    text,
                    mode="printed",
                )
            )

            if (
                confidence
                > best_confidence
            ):

                best_text = text
                best_confidence = confidence

            # ------------------------------------------------
            # Stop immediately once reliable text is found.
            # ------------------------------------------------

            if self._is_reliable(
                text,
                confidence,
            ):

                break

        return (
            best_text,
            best_confidence,
        )

    # ========================================================
    # HANDWRITING OCR
    # ========================================================

    def _safe_handwriting(
        self,
        image: Image.Image,
    ) -> Tuple[str, float]:

        try:

            self._ensure_handwriting()

        except Exception:

            return "", 0.0

        preprocessed = (
            self._preprocess_for_handwriting(
                image
            )
        )

        try:

            text = (
                self._handwriting
                .recognize_from_pil(
                    preprocessed
                )
            )

        except Exception:

            return "", 0.0

        text = self._clean_text(
            text
        )

        confidence = (
            self._estimate_confidence(
                text,
                mode="handwritten",
            )
        )

        return (
            text,
            confidence,
        )

    # ========================================================
    # CAPTIONING
    # ========================================================

    def _safe_caption(self, image: Image.Image) -> str:
        import time

        try:
            print("[PHOTO] Captioning started...", flush=True)

            start = time.perf_counter()

            self._ensure_captioner()

            description = self._photo_captioner.describe(image)

            elapsed = time.perf_counter() - start

            print(
                f"[PHOTO] Captioning completed in {elapsed:.2f}s",
                flush=True,
            )

            description = self._clean_text(description)

            print(
                f"[PHOTO] Caption result: {description[:200]}",
                flush=True,
            )

            if len(description.strip()) >= self.MIN_USEFUL_CHARS:
                return description

        except Exception as exc:
            print(f"[PHOTO] Captioning unavailable: {exc}", flush=True)

        return ""

    def _ensure_captioner(
        self,
    ) -> None:

        if (
            self._photo_captioner
            is None
        ):

            from app.services.image_captioner import (
                PhotoCaptioner,
            )

            self._photo_captioner = (
                PhotoCaptioner()
            )

    # ========================================================
    # DESKEW
    # ========================================================

    def _deskew(
        self,
        image: Image.Image,
        max_angle: float = 15.0,
    ) -> Image.Image:

        try:

            import cv2

        except ImportError:

            return image

        try:

            gray = np.asarray(
                image.convert("L")
            )

            edges = cv2.Canny(
                gray,
                50,
                150,
                apertureSize=3,
            )

            min_line_length = max(
                50,
                gray.shape[1] // 4,
            )

            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=80,
                minLineLength=min_line_length,
                maxLineGap=20,
            )

            if lines is None:

                return image

            angles = []

            for line in lines:

                coords = (
                    np.asarray(line)
                    .reshape(-1)
                )

                if coords.size < 4:

                    continue

                x1, y1, x2, y2 = (
                    coords[:4]
                )

                dx = x2 - x1

                if dx == 0:

                    continue

                angle = float(
                    np.degrees(
                        np.arctan2(
                            y2 - y1,
                            dx,
                        )
                    )
                )

                if (
                    abs(angle)
                    <= max_angle
                ):

                    angles.append(
                        angle
                    )

            if not angles:

                return image

            median_angle = float(
                np.median(
                    angles
                )
            )

            if (
                abs(median_angle)
                < 0.4
            ):

                return image

            return image.rotate(
                -median_angle,
                expand=True,
                fillcolor=(
                    255,
                    255,
                    255,
                ),
                resample=(
                    Image.BICUBIC
                ),
            )

        except Exception:

            return image

    # ========================================================
    # PRINTED PREPROCESSING
    # ========================================================

    def _preprocess_for_printed(
        self,
        image: Image.Image,
    ) -> Image.Image:

        image = self._deskew(
            image
        )

        img = image.convert(
            "L"
        )

        img = ImageOps.autocontrast(
            img,
            cutoff=1,
        )

        img = img.filter(
            ImageFilter.SHARPEN
        )

        enhancer = (
            ImageEnhance.Contrast(
                img
            )
        )

        img = enhancer.enhance(
            1.25
        )

        return img.convert(
            "RGB"
        )

    # ========================================================
    # HANDWRITING PREPROCESSING
    # ========================================================

    def _preprocess_for_handwriting(
        self,
        image: Image.Image,
    ) -> Image.Image:

        image = self._deskew(
            image
        )

        img = image.convert(
            "L"
        )

        img = ImageOps.autocontrast(
            img,
            cutoff=2,
        )

        img = img.filter(
            ImageFilter.MedianFilter(
                size=3
            )
        )

        enhancer = (
            ImageEnhance.Contrast(
                img
            )
        )

        img = enhancer.enhance(
            1.4
        )

        return img.convert(
            "RGB"
        )

    # ========================================================
    # RELIABILITY
    # ========================================================

    def _is_reliable(
        self,
        text: str,
        confidence: float,
    ) -> bool:

        if not text:

            return False

        if (
            len(text.strip())
            < self.MIN_USEFUL_CHARS
        ):

            return False

        if self._is_garbage(
            text
        ):

            return False

        return (
            confidence >= 0.30
        )

    # ========================================================
    # GARBAGE DETECTION
    # ========================================================

    def _is_garbage(
        self,
        text: str,
    ) -> bool:

        if not text:

            return True

        text = text.strip()

        if (
            len(text)
            < self.MIN_USEFUL_CHARS
        ):

            return True

        useful = sum(
            char.isalnum()
            or char.isspace()
            for char in text
        )

        ratio = 1.0 - (
            useful
            / max(
                1,
                len(text),
            )
        )

        if (
            ratio
            > self.GARBAGE_CHAR_RATIO
        ):

            return True

        if re.search(
            r"(.)\1{5,}",
            text,
        ):

            return True

        if re.search(
            r"[^\w\s.,;:!?()'\"/%+\-]{4,}",
            text,
            re.UNICODE,
        ):

            return True

        # ----------------------------------------------------
        # Irregular internal capitalization
        #
        # Real printed/typed words don't switch from lowercase
        # back to uppercase mid-word (e.g. "AagentinA", "Alscra",
        # "Iune", "UNITeS"). This pattern is a strong signature
        # of OCR misreading stylized text - curved jersey
        # lettering, sponsor logos, small distant signage - and
        # is common on photographs even when EasyOCR reports
        # high per-character confidence.
        # ----------------------------------------------------

        words = text.split()

        if words:

            bad_words = 0

            for word in words:

                letters_only = "".join(
                    ch
                    for ch in word
                    if ch.isalpha()
                )

                if len(letters_only) < 3:

                    continue

                # lowercase letter followed later by an
                # uppercase letter, anywhere after position 0
                if re.search(
                    r"[a-z].*[A-Z]",
                    letters_only[1:],
                ):

                    bad_words += 1

            if (
                bad_words
                / len(words)
                > 0.10
            ):

                return True

        return False

    # ========================================================
    # TEXT CLEANING
    # ========================================================

    @staticmethod
    def _clean_text(
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

        text = "".join(
            char
            for char in text
            if ord(char) >= 32
            or char in "\n\t"
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        text = re.sub(
            r"[ \t]+([,.;:!?])",
            r"\1",
            text,
        )

        return text.strip()

    # ========================================================
    # CONFIDENCE
    # ========================================================

    @staticmethod
    def _estimate_confidence(
        text: str,
        mode: str,
    ) -> float:

        if not text:

            return 0.0

        length_score = min(
            1.0,
            len(text) / 120.0,
        )

        useful_ratio = (
            sum(
                char.isalnum()
                or char.isspace()
                for char in text
            )
            / max(
                1,
                len(text),
            )
        )

        confidence = (
            0.45 * length_score
            + 0.55 * useful_ratio
        )

        if (
            mode
            == "handwritten"
        ):

            confidence *= 0.92

        return round(
            min(
                0.98,
                max(
                    0.05,
                    confidence,
                ),
            ),
            3,
        )

    # ========================================================
    # LAZY RECOGNIZERS
    # ========================================================

    def _ensure_handwriting(
        self,
    ) -> None:

        if (
            self._handwriting
            is None
        ):

            from app.services.handwriting import (
                HandwritingRecognizer,
            )

            self._handwriting = (
                HandwritingRecognizer()
            )

    def _ensure_printed(
        self,
    ) -> None:

        if (
            self._printed
            is None
        ):

            from app.services.printed_ocr import (
                PrintedTextRecognizer,
            )

            self._printed = (
                PrintedTextRecognizer()
            )

    # ========================================================
    # EMPTY RESULT
    # ========================================================

    @staticmethod
    def _empty_result(
        image_type: str,
        scores: Dict[str, float],
    ) -> ImageProcessResult:

        return ImageProcessResult(
            text="",
            source_type=image_type,
            confidence=0.0,
            has_text=False,
            description=(
                "No reliable text or image "
                "description could be extracted."
            ),
            meta={
                "scores": scores,
                "recognizer": None,
            },
        )