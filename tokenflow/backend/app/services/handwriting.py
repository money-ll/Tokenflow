"""
Handwritten text recognition (Image Pipeline - Path B).

Implements the TrOCR-based handwriting recognition path described in
Section 5.4.3 of the project report. The model is loaded lazily on first
use so that importing this module -- and starting the FastAPI app --
never requires torch/transformers to be installed or any weights to be
downloaded unless a handwritten image is actually submitted.

Because TrOCR (base and large "handwritten" checkpoints) is trained on
single text-line crops, a whole page of handwriting is first segmented
into line-level strips using a simple horizontal ink-projection profile
(no extra model/dependency required), and each line is recognized
independently. Recognized lines are then rejoined with newlines so the
output reads as a normal multi-line document before being handed to the
same Text Compression Module used for plain text and PDF input.
"""

import io
from PIL import Image


class HandwritingRecognizer:
    MODEL_NAME = "microsoft/trocr-la"

    # Line-segmentation tuning. These are intentionally conservative
    # defaults for typical photographed/scanned notebook pages.
    MIN_LINE_GAP_PX = 6
    MIN_LINE_HEIGHT_PX = 8
    LINE_PADDING_PX = 4
    ROW_INK_THRESHOLD_RATIO = 0.02

    def __init__(self):
        self._processor = None
        self._model = None
        self._device = None
        self._dtype = None

    def _ensure_loaded(self):
        if self._model is not None:
            return

        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise RuntimeError(
                "Handwritten text recognition requires 'torch' and "
                "'transformers'. Install them with: "
                "pip install torch transformers"
            ) from exc

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32

        self._processor = TrOCRProcessor.from_pretrained(self.MODEL_NAME)
        self._model = VisionEncoderDecoderModel.from_pretrained(
            self.MODEL_NAME, torch_dtype=self._dtype
        ).to(self._device)
        self._model.eval()

    def recognize(
        self, image_bytes: bytes, max_new_tokens: int = 256, batch_size: int = 8
    ) -> str:
        """Recognize handwritten English text from an image and return it
        as a plain multi-line string."""
        self._ensure_loaded()
        import torch

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        line_images = self._segment_lines(image)

        if not line_images:
            return ""

        recognized_lines = []

        # Process lines in mini-batches to prevent GPU OOM while maximizing speed
        for i in range(0, len(line_images), batch_size):
            batch_lines = line_images[i : i + batch_size]

            # Batch image preprocessing
            pixel_values = (
                self._processor(images=batch_lines, return_tensors="pt")
                .pixel_values.to(self._device)
                .to(self._dtype)
            )

            # Fast generation pass
            with torch.no_grad():
                generated_ids = self._model.generate(
                    pixel_values,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,  # Greedy search (much faster than beam search)
                    use_cache=True,  # Reuse KV cache during generation
                )

            # Batch decoding
            decoded_batch = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            for text in decoded_batch:
                cleaned_text = text.strip()
                if cleaned_text:
                    recognized_lines.append(cleaned_text)

        return "\n".join(recognized_lines)

    def _segment_lines(self, image: Image.Image):
        """Split a page-level image into individual text-line crops using
        a horizontal ink-density projection. Falls back to the whole
        image if no clear line bands are detected (e.g. a single word or
        short phrase)."""
        import numpy as np

        grayscale = np.array(image.convert("L"))
        ink = 255 - grayscale
        row_ink = ink.sum(axis=1)

        peak = row_ink.max()
        if peak <= 0:
            return [image]

        threshold = peak * self.ROW_INK_THRESHOLD_RATIO
        is_text_row = row_ink > threshold

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
                    if end - start >= self.MIN_LINE_HEIGHT_PX:
                        bands.append((start, end))
                    start = None
                    gap = 0

        if start is not None:
            end = len(is_text_row) - 1
            if end - start >= self.MIN_LINE_HEIGHT_PX:
                bands.append((start, end))

        if not bands:
            return [image]

        crops = []
        for top, bottom in bands:
            top = max(0, top - self.LINE_PADDING_PX)
            bottom = min(image.height, bottom + self.LINE_PADDING_PX)
            crops.append(image.crop((0, top, image.width, bottom)))

        return crops