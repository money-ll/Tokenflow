"""
TokenFlow Image Captioner

Provides lazy-loaded image description for photographs and
non-text images.

Model:
    Salesforce/blip-image-captioning-base

Important:
    The model is NOT loaded during application startup.

It is loaded only when describe() is actually called.
This prevents BLIP from increasing FastAPI/Uvicorn startup time.
"""

from __future__ import annotations

from typing import Optional

from PIL import Image


class PhotoCaptioner:
    """
    Lazy-loaded BLIP image captioner.

    The captioner is intentionally independent from OCR.

    OCR answers:
        "What text is visible?"

    Captioning answers:
        "What is visually present in the image?"
    """

    MODEL_NAME = "Salesforce/blip-image-captioning-base"

    def __init__(
        self,
        model_name: Optional[str] = None,
    ) -> None:
        self.model_name = (
            model_name or self.MODEL_NAME
        )

        self._processor = None
        self._model = None
        self._device = None
        self._loaded = False

    # ==========================================================
    # MODEL LOADING
    # ==========================================================

    def _ensure_loaded(self) -> None:
        """
        Load BLIP only when it is actually needed.
        """

        if self._loaded:
            return

        try:
            import torch
            from transformers import (
                BlipProcessor,
                BlipForConditionalGeneration,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Photo captioning requires "
                "'transformers' and 'torch'. "
                "Install them with: "
                "pip install transformers torch"
            ) from exc

        self._device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"[PHOTO] Loading BLIP "
            f"({self.model_name}) "
            f"on {self._device}..."
        )

        try:
            self._processor = (
                BlipProcessor.from_pretrained(
                    self.model_name
                )
            )

            self._model = (
                BlipForConditionalGeneration.from_pretrained(
                    self.model_name
                )
            )

            self._model.to(self._device)
            self._model.eval()

        except Exception as exc:
            self._processor = None
            self._model = None
            self._device = None

            raise RuntimeError(
                f"Could not load image captioning model: "
                f"{exc}"
            ) from exc

        self._loaded = True

        print(
            f"[PHOTO] BLIP ready on "
            f"{self._device}."
        )

    # ==========================================================
    # DESCRIPTION
    # ==========================================================

    def describe(
        self,
        image: Image.Image,
    ) -> str:
        """
        Generate a concise description of an image.
        """

        self._ensure_loaded()

        import torch

        image = image.convert("RGB")

        try:
            inputs = self._processor(
                images=image,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(self._device)
                for key, value in inputs.items()
            }

            with torch.inference_mode():

                output = self._model.generate(
                    **inputs,
                    max_new_tokens=60,
                    num_beams=3,
                    do_sample=False,
                )

            text = self._processor.decode(
                output[0],
                skip_special_tokens=True,
            )

        except Exception as exc:
            raise RuntimeError(
                f"Image captioning failed: {exc}"
            ) from exc

        return self._clean_caption(text)

    # ==========================================================
    # CLEANING
    # ==========================================================

    @staticmethod
    def _clean_caption(
        text: str,
    ) -> str:

        if not text:
            return ""

        text = " ".join(
            text.strip().split()
        )

        return text

    # ==========================================================
    # STATUS
    # ==========================================================

    @property
    def is_loaded(self) -> bool:
        """
        Useful for diagnostics.
        """

        return self._loaded

    @property
    def device(self) -> Optional[str]:
        """
        Return the active device once loaded.
        """

        return self._device