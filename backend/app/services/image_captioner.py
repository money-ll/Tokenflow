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
    # TIMM COMPATIBILITY SHIM
    # ==========================================================

    @staticmethod
    def _ensure_timm_layers_shim() -> None:
        """
        Make `import timm.layers` / `from timm.layers import X`
        work even when the installed timm (0.5.4, pinned by
        pix2tex) predates the timm.layers module, by pointing it
        at the equivalent legacy module, timm.models.layers.
        """

        import sys

        if "timm.layers" in sys.modules:
            return

        try:
            import timm.layers  # noqa: F401

            # Real timm.layers exists (timm was upgraded) --
            # nothing to shim.
            return

        except ImportError:
            pass

        try:
            import timm.models.layers as legacy_layers
        except ImportError as exc:
            raise RuntimeError(
                "Could not locate timm's layers module "
                "(checked both timm.layers and "
                "timm.models.layers). Is timm installed?"
            ) from exc

        sys.modules["timm.layers"] = legacy_layers

        import timm
        timm.layers = legacy_layers

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

        # Florence-2's remote modeling code does
        # `from timm.layers import DropPath, trunc_normal_`.
        # That module path only exists in timm>=0.9. This project
        # also depends on pix2tex, which hard-pins timm==0.5.4 for
        # its own vision backbone -- so we can't just upgrade timm
        # globally without breaking math extraction.
        #
        # In 0.5.4 the same two symbols live at
        # timm.models.layers, so we register a lightweight alias
        # module at "timm.layers" the first time it's needed. This
        # is a no-op if a real timm.layers already exists (i.e. if
        # timm was in fact upgraded).
        self._ensure_timm_layers_shim()

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
                    # Florence-2's trust_remote_code modeling file
                    # predates transformers' newer Cache handling:
                    # it treats an empty DynamicCache object as "we
                    # have past state" (checking `is not None`
                    # instead of length) and then indexes into it,
                    # raising "Cache only has 0 layers, attempted
                    # to access layer with index 0" on newer
                    # transformers versions. Disabling the KV cache
                    # avoids that code path entirely -- generation
                    # recomputes attention each step instead of
                    # reusing past_key_values, which is slightly
                    # slower but harmless for single-image captions.
                    use_cache=False,
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