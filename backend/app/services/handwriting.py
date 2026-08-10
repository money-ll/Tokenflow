"""
High-performance handwritten text recognition for TokenFlow.

Optimized for:
- NVIDIA CUDA GPUs
- RTX 3060 6 GB VRAM
- TrOCR Base Handwritten
- FP16 CUDA inference
- Batched line recognition
- Fast line segmentation

Environment variables:

HANDWRITING_MODEL
    Default:
    microsoft/trocr-base-handwritten

HANDWRITING_GPU_BATCH_SIZE
    Default: 4

HANDWRITING_CPU_BATCH_SIZE
    Default: 2

HANDWRITING_MAX_NEW_TOKENS
    Default: 64
"""

import io
import os
import threading
from typing import List, Optional

import numpy as np
from PIL import Image


class HandwritingRecognizer:

    # ------------------------------------------------------------------
    # MODEL CONFIGURATION
    # ------------------------------------------------------------------

    MODEL_NAME = os.environ.get(
        "HANDWRITING_MODEL",
        "microsoft/trocr-large-handwritten",
    )

    GPU_BATCH_SIZE = int(
        os.environ.get(
            "HANDWRITING_GPU_BATCH_SIZE",
            "4",
        )
    )

    CPU_BATCH_SIZE = int(
        os.environ.get(
            "HANDWRITING_CPU_BATCH_SIZE",
            "2",
        )
    )

    MAX_NEW_TOKENS = int(
        os.environ.get(
            "HANDWRITING_MAX_NEW_TOKENS",
            "64",
        )
    )

    # ------------------------------------------------------------------
    # LINE SEGMENTATION
    # ------------------------------------------------------------------

    MIN_LINE_GAP_PX = 6
    MIN_LINE_HEIGHT_PX = 8
    LINE_PADDING_PX = 4

    # Percentage of maximum row ink required for a row
    # to be considered part of a text line.
    ROW_INK_THRESHOLD_RATIO = 0.02

    # Ignore extremely small crops.
    MIN_CROP_WIDTH = 32
    MIN_CROP_HEIGHT = 10

    def __init__(self):
        self._processor = None
        self._model = None
        self._device = None
        self._dtype = None
        self._torch = None

        self._load_lock = threading.Lock()
        self._loaded = False

    # ==================================================================
    # MODEL LOADING
    # ==================================================================

    def _ensure_loaded(self):
        """
        Load TrOCR exactly once.

        Thread-safe so multiple API requests cannot load
        multiple copies of the model simultaneously.
        """

        if self._loaded:
            return

        with self._load_lock:

            if self._loaded:
                return

            try:
                import torch
                from transformers import (
                    TrOCRProcessor,
                    VisionEncoderDecoderModel,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Handwritten recognition requires torch and "
                    "transformers.\n\n"
                    "Install with:\n"
                    "pip install torch transformers"
                ) from exc

            self._torch = torch

            # ----------------------------------------------------------
            # DEVICE
            # ----------------------------------------------------------

            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")

            # ----------------------------------------------------------
            # PRECISION
            # ----------------------------------------------------------

            if self._device.type == "cuda":
                self._dtype = torch.float16
            else:
                self._dtype = torch.float32

            # ----------------------------------------------------------
            # CUDA OPTIMIZATION
            # ----------------------------------------------------------

            if self._device.type == "cuda":

                # RTX 30-series benefits from TF32 for supported
                # matrix operations.
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

                # Useful for repeated image inference.
                torch.backends.cudnn.benchmark = True

            # ----------------------------------------------------------
            # PROCESSOR
            # ----------------------------------------------------------

            try:

                self._processor = TrOCRProcessor.from_pretrained(
                    self.MODEL_NAME,
                    use_fast=True,
                )

            except (TypeError, ValueError):

                # Compatibility with older Transformers versions.
                self._processor = TrOCRProcessor.from_pretrained(
                    self.MODEL_NAME,
                )

            # ----------------------------------------------------------
            # MODEL
            # ----------------------------------------------------------

            self._model = VisionEncoderDecoderModel.from_pretrained(
                self.MODEL_NAME,
                torch_dtype=self._dtype,
                low_cpu_mem_usage=True,
            )

            self._model.to(self._device)
            self._model.eval()

            # ----------------------------------------------------------
            # GENERATION SETTINGS
            # ----------------------------------------------------------

            generation_config = self._model.generation_config

            generation_config.num_beams = 1
            generation_config.do_sample = False
            generation_config.use_cache = True

            # ----------------------------------------------------------
            # FIX cuDNN RNN/LSTM MEMORY WARNING
            # ----------------------------------------------------------

            if self._device.type == "cuda":

                for module in self._model.modules():

                    flatten = getattr(
                        module,
                        "flatten_parameters",
                        None,
                    )

                    if callable(flatten):

                        try:
                            flatten()
                        except Exception:
                            # Some modules may expose the method but
                            # not support it. Do not break inference.
                            pass

            self._loaded = True

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def recognize(
        self,
        image_bytes: bytes,
        max_new_tokens: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> str:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        # TrOCR expects RGB.
        if image.mode != "RGB":
            image = image.convert("RGB")

        return self.recognize_from_pil(
            image,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )

    # ==================================================================
    # MAIN OCR
    # ==================================================================

    def recognize_from_pil(
        self,
        image: Image.Image,
        max_new_tokens: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> str:

        self._ensure_loaded()

        torch = self._torch

        # --------------------------------------------------------------
        # PARAMETERS
        # --------------------------------------------------------------

        if max_new_tokens is None:
            max_new_tokens = self.MAX_NEW_TOKENS

        if batch_size is None:

            if self._device.type == "cuda":
                batch_size = self.GPU_BATCH_SIZE
            else:
                batch_size = self.CPU_BATCH_SIZE

        # Prevent invalid values.
        batch_size = max(1, int(batch_size))
        max_new_tokens = max(8, int(max_new_tokens))

        # --------------------------------------------------------------
        # LINE SEGMENTATION
        # --------------------------------------------------------------

        line_images = self._segment_lines(image)

        if not line_images:
            return ""

        recognized_lines: List[str] = []

        # --------------------------------------------------------------
        # BATCH INFERENCE
        # --------------------------------------------------------------

        for start in range(
            0,
            len(line_images),
            batch_size,
        ):

            batch_lines = line_images[
                start:start + batch_size
            ]

            # ----------------------------------------------------------
            # IMAGE PREPROCESSING
            # ----------------------------------------------------------

            inputs = self._processor(
                images=batch_lines,
                return_tensors="pt",
            )

            pixel_values = inputs.pixel_values

            # Move directly to the correct GPU/CPU + precision.
            pixel_values = pixel_values.to(
                device=self._device,
                dtype=self._dtype,
                non_blocking=(
                    self._device.type == "cuda"
                ),
            )

            # ----------------------------------------------------------
            # MODEL INFERENCE
            # ----------------------------------------------------------

            with torch.inference_mode():

                generated_ids = self._model.generate(
                    pixel_values,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    use_cache=True,
                )

            # ----------------------------------------------------------
            # DECODING
            # ----------------------------------------------------------

            decoded = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            for text in decoded:

                cleaned = text.strip()

                # Remove obviously useless outputs.
                if len(cleaned) <= 1:
                    continue

                recognized_lines.append(cleaned)

            # ----------------------------------------------------------
            # RELEASE TEMPORARY TENSORS
            # ----------------------------------------------------------

            del pixel_values
            del generated_ids
            del inputs

        return "\n".join(recognized_lines)

    # ==================================================================
    # WARMUP
    # ==================================================================

    def warmup(self):
        """
        Warm up TrOCR during FastAPI startup.

        This moves the model initialization and first CUDA kernel
        compilation away from the first real user request.
        """

        try:

            self._ensure_loaded()

            dummy = Image.new(
                "RGB",
                (384, 64),
                "white",
            )

            self.recognize_from_pil(
                dummy,
                max_new_tokens=8,
                batch_size=1,
            )

            # Synchronize CUDA so warm-up is actually completed.
            if (
                self._device is not None
                and self._device.type == "cuda"
            ):
                self._torch.cuda.synchronize()

        except Exception as exc:

            # Warm-up must never prevent the API from starting.
            print(
                f"WARNING: Handwriting model warm-up failed: {exc}"
            )

    # ==================================================================
    # LINE SEGMENTATION
    # ==================================================================

    def _segment_lines(
        self,
        image: Image.Image,
    ) -> List[Image.Image]:
        """
        Fast horizontal text-line segmentation.

        Uses NumPy instead of heavier OpenCV operations.
        """

        # --------------------------------------------------------------
        # GRAYSCALE
        # --------------------------------------------------------------

        grayscale = np.asarray(
            image.convert("L"),
            dtype=np.uint8,
        )

        if grayscale.size == 0:
            return []

        # --------------------------------------------------------------
        # INK DETECTION
        # --------------------------------------------------------------

        ink = 255 - grayscale

        row_ink = ink.sum(
            axis=1,
            dtype=np.uint64,
        )

        peak = int(row_ink.max())

        if peak <= 0:
            return [image]

        threshold = peak * self.ROW_INK_THRESHOLD_RATIO

        is_text_row = row_ink > threshold

        # --------------------------------------------------------------
        # DETECT TEXT BANDS
        # --------------------------------------------------------------

        bands = []

        start = None
        gap = 0

        for y, has_ink in enumerate(is_text_row):

            if has_ink:

                if start is None:
                    start = y

                gap = 0

            elif start is not None:

                gap += 1

                if gap >= self.MIN_LINE_GAP_PX:

                    end = y - gap

                    if (
                        end - start
                        >= self.MIN_LINE_HEIGHT_PX
                    ):
                        bands.append(
                            (
                                start,
                                end,
                            )
                        )

                    start = None
                    gap = 0

        # --------------------------------------------------------------
        # FINAL BAND
        # --------------------------------------------------------------

        if start is not None:

            end = len(is_text_row) - 1

            if (
                end - start
                >= self.MIN_LINE_HEIGHT_PX
            ):
                bands.append(
                    (
                        start,
                        end,
                    )
                )

        if not bands:
            return [image]

        # --------------------------------------------------------------
        # CREATE CROPS
        # --------------------------------------------------------------

        width, height = image.size

        crops: List[Image.Image] = []

        for top, bottom in bands:

            top = max(
                0,
                top - self.LINE_PADDING_PX,
            )

            bottom = min(
                height,
                bottom + self.LINE_PADDING_PX,
            )

            crop_height = bottom - top

            if (
                width < self.MIN_CROP_WIDTH
                or crop_height < self.MIN_CROP_HEIGHT
            ):
                continue

            crop = image.crop(
                (
                    0,
                    top,
                    width,
                    bottom,
                )
            )

            crops.append(crop)

        # If segmentation rejected everything,
        # return the original image rather than losing data.
        if not crops:
            return [image]

        return crops

