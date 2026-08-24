"""
TokenFlow Printed-Text OCR Service

Optimized EasyOCR service for:

- scanned PDF pages
- printed PNG/JPG/JPEG images
- mixed printed documents

Optimization goals:

1. Reuse one EasyOCR reader.
2. Use GPU when CUDA is available.
3. Avoid unnecessary image resizing.
4. Reduce EasyOCR detection workload.
5. Avoid paragraph reconstruction inside EasyOCR.
6. Keep OCR quality suitable for technical documents.
7. Provide optional FAST mode for additional speed.
8. Keep detailed profiling available.

The main performance bottleneck is EasyOCR inference itself.
"""

import os
import time

import numpy as np
from PIL import Image


class PrintedTextRecognizer:

    LANGUAGES = ["en"]

    # ========================================================
    # PERFORMANCE SETTINGS
    # ========================================================

    # Some source pages (large scans/posters, high-DPI renders)
    # are far bigger than a normal 1275x1650 page. Resizing to
    # this ceiling still keeps performance reasonable.
    MAX_DIMENSION = 1600

    # EasyOCR's internal detection canvas.
    #
    # IMPORTANT: this must be >= MAX_DIMENSION. If it's smaller,
    # EasyOCR will silently downscale the image a SECOND time
    # (on top of our own resize in _prepare_for_ocr), and the
    # two compounding downscales can shrink small/technical text
    # below EasyOCR's detection threshold entirely -- producing
    # "Detected text blocks: 0" even on a perfectly normal page.
    OCR_CANVAS_SIZE = MAX_DIMENSION

    # Never upscale the input.
    OCR_MAG_RATIO = 1.0

    # EasyOCR detection thresholds.
    #
    # These are slightly less conservative than the previous
    # configuration so EasyOCR does not spend excessive time
    # trying to resolve marginal text regions.
    TEXT_THRESHOLD = 0.65
    LOW_TEXT = 0.35
    LINK_THRESHOLD = 0.35

    # Text grouping.
    WIDTH_THRESHOLD = 0.7
    YCENTER_THRESHOLD = 0.5

    # Contrast processing.
    #
    # Keeping this moderate avoids unnecessary processing.
    CONTRAST_THRESHOLD = 0.1
    ADJUST_CONTRAST = 0.5

    # Performance warnings.
    SLOW_THRESHOLD = 3.0
    VERY_SLOW_THRESHOLD = 5.0

    # Recognition batch size.
    #
    # NOTE: this does NOT batch multiple pages together. It
    # batches the individual text-region crops that EasyOCR's
    # detector finds WITHIN a single image through the CRNN
    # recognizer in one pass instead of one-crop-at-a-time.
    # With batch_size=1 (the old default), a page/retry pass
    # that turns up many candidate regions pays full Python +
    # CUDA launch overhead per region, which is what made the
    # full-resolution retry pass so slow. Batching this has no
    # effect on what gets detected/recognized -- same results,
    # far less overhead.
    RECOGNITION_BATCH_SIZE = 8

    def __init__(self):

        self._reader = None

        self._gpu = False

        self._initialization_time = None

        # Cache CUDA status so status() does not repeatedly
        # perform GPU detection.
        self._cuda_checked = False

    # ========================================================
    # MODEL INITIALIZATION
    # ========================================================

    def _ensure_loaded(self):

        if self._reader is not None:
            return

        start = time.perf_counter()

        try:

            import easyocr

        except ImportError as exc:

            raise RuntimeError(
                "Scanned-PDF OCR fallback requires 'easyocr'. "
                "Install it with: pip install easyocr"
            ) from exc

        # ----------------------------------------------------
        # Detect GPU once.
        # ----------------------------------------------------

        self._gpu = self._detect_gpu()

        print()
        print("=" * 60)
        print("INITIALIZING EASYOCR")
        print(
            f"GPU enabled: {self._gpu}"
        )
        print(
            f"Canvas size: {self.OCR_CANVAS_SIZE}"
        )
        print("=" * 60)

        # ----------------------------------------------------
        # Load EasyOCR exactly once.
        # ----------------------------------------------------

        self._reader = easyocr.Reader(
            self.LANGUAGES,
            gpu=self._gpu,
            verbose=False,
        )

        self._initialization_time = (
            time.perf_counter() - start
        )

        print(
            f"EasyOCR initialization: "
            f"{self._initialization_time:.2f}s"
        )

        print("=" * 60)
        print()

    # ========================================================
    # GPU DETECTION
    # ========================================================

    @staticmethod
    def _detect_gpu():

        try:

            import torch

            available = torch.cuda.is_available()

            if available:

                try:

                    device_name = (
                        torch.cuda.get_device_name(0)
                    )

                except Exception:

                    device_name = "CUDA GPU"

                print(
                    f"[OCR] CUDA available: "
                    f"{device_name}"
                )

            else:

                print(
                    "[OCR] CUDA unavailable - "
                    "using CPU"
                )

            return available

        except Exception as exc:

            print(
                f"[OCR] Could not check CUDA: "
                f"{exc}"
            )

            return False

    # ========================================================
    # IMAGE CONVERSION
    # ========================================================

    @staticmethod
    def _to_numpy(image: Image.Image):

        start = time.perf_counter()

        if image.mode == "RGB":

            array = np.asarray(
                image,
                dtype=np.uint8,
            )

        else:

            array = np.asarray(
                image.convert("RGB"),
                dtype=np.uint8,
            )

        # EasyOCR/PyTorch performs better with contiguous
        # arrays in many cases.
        if not array.flags["C_CONTIGUOUS"]:

            array = np.ascontiguousarray(
                array
            )

        conversion_time = (
            time.perf_counter()
            - start
        )

        return array, conversion_time

    # ========================================================
    # IMAGE PREPARATION
    # ========================================================

    @classmethod
    def _prepare_for_ocr(cls, array):

        height, width = array.shape[:2]

        max_dimension = max(
            height,
            width,
        )

        # ----------------------------------------------------
        # Most normal scanned pages should take this path.
        # ----------------------------------------------------

        if max_dimension <= cls.MAX_DIMENSION:

            return (
                array,
                False,
                1.0,
            )

        # ----------------------------------------------------
        # Only resize genuinely oversized images.
        # ----------------------------------------------------

        scale = (
            cls.MAX_DIMENSION
            / max_dimension
        )

        new_width = max(
            1,
            int(width * scale),
        )

        new_height = max(
            1,
            int(height * scale),
        )

        image = Image.fromarray(
            array
        )

        resized = image.resize(
            (
                new_width,
                new_height,
            ),
            Image.Resampling.LANCZOS,
        )

        resized_array = np.asarray(
            resized,
            dtype=np.uint8,
        )

        if not resized_array.flags[
            "C_CONTIGUOUS"
        ]:

            resized_array = (
                np.ascontiguousarray(
                    resized_array
                )
            )

        return (
            resized_array,
            True,
            scale,
        )

    # ========================================================
    # FAST MODE
    # ========================================================

    @staticmethod
    def _fast_mode_enabled():

        value = os.environ.get(
            "TOKENFLOW_OCR_FAST",
            "0",
        )

        return value.lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    # ========================================================
    # OCR
    # ========================================================

    def recognize(
        self,
        image: Image.Image,
        allow_full_resolution_retry: bool = False,
    ):

        print(
            "\n>>> PRINTED OCR "
            "RECOGNIZE() CALLED <<<",
            flush=True,
        )

        # ----------------------------------------------------
        # Load model only once.
        # ----------------------------------------------------

        self._ensure_loaded()

        print(
            ">>> EASYOCR READY - "
            "STARTING PAGE <<<",
            flush=True,
        )

        total_start = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # Image conversion
        # ----------------------------------------------------

        array, conversion_time = (
            self._to_numpy(image)
        )

        original_height, original_width = (
            array.shape[:2]
        )

        print(
            f"[OCR] Original image: "
            f"{original_width}x"
            f"{original_height} "
            f"| conversion: "
            f"{conversion_time:.3f}s"
        )

        # ----------------------------------------------------
        # Image resizing
        # ----------------------------------------------------

        resize_start = (
            time.perf_counter()
        )

        (
            ocr_array,
            resized,
            scale,
        ) = self._prepare_for_ocr(
            array
        )

        resize_time = (
            time.perf_counter()
            - resize_start
        )

        ocr_height, ocr_width = (
            ocr_array.shape[:2]
        )

        if resized:

            print(
                f"[OCR] Adaptive resize: "
                f"{original_width}x"
                f"{original_height} -> "
                f"{ocr_width}x"
                f"{ocr_height} "
                f"| scale: {scale:.3f} "
                f"| time: "
                f"{resize_time:.3f}s"
            )

        else:

            print(
                "[OCR] Adaptive resize: "
                "not required"
            )

        # ----------------------------------------------------
        # Determine OCR mode
        # ----------------------------------------------------

        fast_mode = (
            self._fast_mode_enabled()
        )

        if fast_mode:

            print(
                "[OCR] FAST MODE: enabled"
            )

        # ----------------------------------------------------
        # EasyOCR inference
        # ----------------------------------------------------

        inference_start = (
            time.perf_counter()
        )

        # batch_size batches the recognizer step across all
        # text-region crops the detector found in THIS image
        # (see RECOGNITION_BATCH_SIZE docstring above). It does
        # not change detection thresholds, so results are
        # identical to the batch_size=1 path -- just faster.

        if fast_mode:

            results = (
                self._reader.readtext(
                    ocr_array,

                    # We only need the recognized text.
                    detail=0,

                    # Paragraph reconstruction is disabled.
                    # This removes unnecessary grouping work.
                    paragraph=False,

                    # Slightly faster detection settings.
                    text_threshold=0.6,
                    low_text=0.3,
                    link_threshold=0.3,

                    canvas_size=1400,
                    mag_ratio=1.0,

                    contrast_ths=0.1,
                    adjust_contrast=0.5,

                    width_ths=0.7,
                    ycenter_ths=0.5,

                    batch_size=self.RECOGNITION_BATCH_SIZE,
                    workers=0,
                )
            )

        else:

            results = (
                self._reader.readtext(
                    ocr_array,

                    # Text only.
                    detail=0,

                    # IMPORTANT:
                    # Do not ask EasyOCR to perform paragraph
                    # reconstruction. TokenFlow performs its
                    # own downstream normalization.
                    paragraph=False,

                    # Detection thresholds.
                    text_threshold=(
                        self.TEXT_THRESHOLD
                    ),

                    low_text=(
                        self.LOW_TEXT
                    ),

                    link_threshold=(
                        self.LINK_THRESHOLD
                    ),

                    # Performance.
                    canvas_size=(
                        self.OCR_CANVAS_SIZE
                    ),

                    mag_ratio=(
                        self.OCR_MAG_RATIO
                    ),

                    # Contrast.
                    contrast_ths=(
                        self.CONTRAST_THRESHOLD
                    ),

                    adjust_contrast=(
                        self.ADJUST_CONTRAST
                    ),

                    # Text grouping.
                    width_ths=(
                        self.WIDTH_THRESHOLD
                    ),

                    ycenter_ths=(
                        self.YCENTER_THRESHOLD
                    ),

                    batch_size=self.RECOGNITION_BATCH_SIZE,
                    workers=0,
                )
            )

        inference_time = (
            time.perf_counter()
            - inference_start
        )

        # ----------------------------------------------------
        # Post-processing
        # ----------------------------------------------------

        post_start = (
            time.perf_counter()
        )

        lines = []

        for result in results:

            if result is None:
                continue

            cleaned = str(
                result
            ).strip()

            if not cleaned:
                continue

            # Normalize internal whitespace.
            cleaned = " ".join(
                cleaned.split()
            )

            if cleaned:

                lines.append(
                    cleaned
                )

        # ----------------------------------------------------
        # Optional high-resolution retry.
        #
        # IMPORTANT:
        # Normal images/photos do NOT use this retry.
        # It is only enabled by callers that explicitly know
        # they are processing a document where tiny text matters.
        #
        # The old implementation could spend 15-20+ seconds
        # retrying a large image just to recover 1-2 characters.
        # ----------------------------------------------------

        retry_time = 0.0

        if (
            not lines
            and resized
            and allow_full_resolution_retry
        ):

            retry_start = time.perf_counter()

            print(
                "[OCR] 0 blocks detected on resized "
                "image - document retry enabled"
            )

            # Do NOT return to the original 5813x4300 size.
            # Use a moderate second-pass canvas instead.
            #
            # First pass: 1600px
            # Retry:     1800px
            #
            # This is much cheaper than the old 2600px retry.
            retry_canvas = 1800

            retry_results = self._reader.readtext(
                array,
                detail=0,
                paragraph=False,

                text_threshold=self.TEXT_THRESHOLD,
                low_text=self.LOW_TEXT,
                link_threshold=self.LINK_THRESHOLD,

                canvas_size=retry_canvas,
                mag_ratio=self.OCR_MAG_RATIO,

                contrast_ths=self.CONTRAST_THRESHOLD,
                adjust_contrast=self.ADJUST_CONTRAST,

                width_ths=self.WIDTH_THRESHOLD,
                ycenter_ths=self.YCENTER_THRESHOLD,

                batch_size=self.RECOGNITION_BATCH_SIZE,
                workers=0,
            )

            for result in retry_results:

                if result is None:
                    continue

                cleaned = str(result).strip()

                if not cleaned:
                    continue

                cleaned = " ".join(
                    cleaned.split()
                )

                if cleaned:
                    lines.append(cleaned)

            retry_time = (
                time.perf_counter()
                - retry_start
            )

            print(
                f"[OCR] Document retry: "
                f"{len(lines)} block(s) recovered "
                f"({retry_time:.3f}s)"
            )

        text = "\n".join(
            lines
        )

        post_time = (
            time.perf_counter()
            - post_start
        )

        total_time = (
            time.perf_counter()
            - total_start
        )

        # ----------------------------------------------------
        # Profiling
        # ----------------------------------------------------

        print(
            f"[OCR] EasyOCR inference: "
            f"{inference_time:.3f}s"
        )

        print(
            f"[OCR] Post-processing: "
            f"{(post_time - retry_time):.3f}s"
        )

        if retry_time:
            print(
                f"[OCR] Full-resolution retry OCR: "
                f"{retry_time:.3f}s"
            )

        print(
            f"[OCR] Total page OCR: "
            f"{total_time:.3f}s"
        )

        print(
            f"[OCR] Detected text blocks: "
            f"{len(lines)}"
        )

        print(
            f"[OCR] Extracted characters: "
            f"{len(text)}"
        )

        # ----------------------------------------------------
        # Performance classification
        # ----------------------------------------------------

        if inference_time >= (
            self.VERY_SLOW_THRESHOLD
        ):

            print(
                "[OCR] WARNING: "
                "Very slow OCR page "
                f"({inference_time:.2f}s)"
            )

        elif inference_time >= (
            self.SLOW_THRESHOLD
        ):

            print(
                "[OCR] NOTICE: "
                "Slow OCR page "
                f"({inference_time:.2f}s)"
            )

        else:

            print(
                "[OCR] Performance: "
                "normal"
            )

        print(
            "-" * 60
        )

        return text

    # ========================================================
    # STATUS
    # ========================================================

    def status(self):

        return {
            "engine": "EasyOCR",
            "languages": self.LANGUAGES,
            "gpu_enabled": self._gpu,
            "initialization_time": (
                self._initialization_time
            ),
            "canvas_size": (
                self.OCR_CANVAS_SIZE
            ),
            "max_dimension": (
                self.MAX_DIMENSION
            ),
            "fast_mode": (
                self._fast_mode_enabled()
            ),
        }