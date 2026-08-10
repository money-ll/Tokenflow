"""
TokenFlow Image Captioner

Provides lazy-loaded image description for photographs and
non-text images.

Model:
    microsoft/Florence-2-base

Important:
    The model is NOT loaded during application startup.

It is loaded only when describe() is actually called.
This prevents the captioning model from increasing FastAPI/Uvicorn
startup time.
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

    MODEL_NAME = "microsoft/Florence-2-base"

    # Florence-2 task prompt that triggers its long-form,
    # structured description mode (position, colors, background,
    # lighting, text presence, etc.) instead of a one-line caption.
    TASK_PROMPT = "<MORE_DETAILED_CAPTION>"

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
        self._dtype = None
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
                AutoConfig,
                AutoProcessor,
                AutoModelForCausalLM,
            )
            from transformers.dynamic_module_utils import (
                get_imports,
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

        # Florence-2 only supports float32 on CPU; fp16 on GPU
        # speeds up generation without hurting caption quality.
        self._dtype = (
            torch.float16
            if self._device == "cuda"
            else torch.float32
        )

        print(
            f"[PHOTO] Loading Florence-2 "
            f"({self.model_name}) "
            f"on {self._device}..."
        )

        try:
            self._processor = (
                AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                )
            )

            # Force eager attention at the config level.
            # Passing attn_implementation="eager" directly to
            # from_pretrained is ignored by Florence-2's custom
            # remote code, which otherwise tries to import
            # flash_attn (not installable on most Windows setups).
            config = AutoConfig.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            config._attn_implementation = "eager"

            # transformers scans Florence-2's remote modeling
            # file for ALL top-level imports (including
            # flash_attn, which is never actually used once
            # eager attention is forced above) and refuses to
            # load unless every one of them is installed. This
            # patch strips flash_attn out of that scan so the
            # unused import no longer blocks loading.
            _original_get_imports = get_imports

            def _patched_get_imports(filename):
                imports = _original_get_imports(filename)
                return [
                    imp
                    for imp in imports
                    if imp != "flash_attn"
                ]

            import transformers.dynamic_module_utils as _dmu
            _dmu.get_imports = _patched_get_imports

            try:
                self._model = (
                    AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        config=config,
                        torch_dtype=self._dtype,
                        trust_remote_code=True,
                    )
                )
            finally:
                _dmu.get_imports = _original_get_imports

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
            f"[PHOTO] Florence-2 ready on "
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
                text=self.TASK_PROMPT,
                images=image,
                return_tensors="pt",
            )

            inputs = {
                key: (
                    value.to(self._device, self._dtype)
                    if value.dtype.is_floating_point
                    else value.to(self._device)
                )
                for key, value in inputs.items()
            }

            with torch.inference_mode():

                output = self._model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=512,
                    num_beams=3,
                    do_sample=False,
                )

            raw_text = self._processor.batch_decode(
                output,
                skip_special_tokens=False,
            )[0]

            parsed = self._processor.post_process_generation(
                raw_text,
                task=self.TASK_PROMPT,
                image_size=(image.width, image.height),
            )

            text = parsed.get(self.TASK_PROMPT, "")

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