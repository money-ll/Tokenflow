"""
Shared math extraction module.

Used by BOTH the PDF pipeline and the image pipeline.

Strategy:

1. Try the complete image with Pix2Tex first.
   This is especially important for images that contain a single equation.

2. If full-image extraction is not suitable, fall back to math-region
   detection and run Pix2Tex on detected regions.

The public MathExtractor.extract() method returns a list of MathResult
objects containing:

    - bounding box
    - LaTeX
    - confidence
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ==========================================================
# DATA CLASSES
# ==========================================================

@dataclass
class MathRegion:
    """
    A detected mathematical region.

    bbox format:

        (x0, y0, x1, y1)
    """

    bbox: Tuple[int, int, int, int]
    confidence: float


@dataclass
class MathResult:
    """
    A mathematical region converted into LaTeX.
    """

    bbox: Tuple[int, int, int, int]
    latex: str
    confidence: float


# ==========================================================
# MATH REGION DETECTOR
# ==========================================================

class MathRegionDetector:
    """
    Lightweight OpenCV-based mathematical region detector.

    This is used as a fallback when the complete image is not itself
    a single equation.

    Example:

        Normal document page
        --------------------

        Some text here.

            x^2 + y^2 = z^2

        More text here.

    The detector attempts to isolate the equation region.
    """

    def __init__(
        self,
        min_region_area: int = 200,
        padding: int = 8,
        merge_gap_x: int = 50,
        merge_gap_y: int = 25,
    ):
        self.min_region_area = min_region_area
        self.padding = padding
        self.merge_gap_x = merge_gap_x
        self.merge_gap_y = merge_gap_y

    def __call__(
        self,
        image: Image.Image,
    ) -> List[MathRegion]:

        try:
            import cv2
        except ImportError:
            logger.warning(
                "opencv-python is not installed. "
                "Math region detection is unavailable."
            )
            return []

        image = image.convert("RGB")

        gray = np.array(
            image.convert("L")
        )

        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV
            + cv2.THRESH_OTSU,
        )

        img_h, img_w = binary.shape

        # Adaptive dilation kernel.
        #
        # The previous fixed (18, 10) kernel could incorrectly
        # split or merge mathematical symbols depending on image size.
        kernel_w = max(
            12,
            min(40, img_w // 40),
        )

        kernel_h = max(
            6,
            min(20, img_h // 30),
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_w, kernel_h),
        )

        dilated = cv2.dilate(
            binary,
            kernel,
            iterations=1,
        )

        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        raw_boxes: List[
            Tuple[int, int, int, int]
        ] = []

        for contour in contours:

            x, y, w, h = cv2.boundingRect(
                contour
            )

            area = w * h

            if area < self.min_region_area:
                continue

            # Ignore tiny artifacts.
            if w < 10 or h < 10:
                continue

            raw_boxes.append(
                (
                    x,
                    y,
                    x + w,
                    y + h,
                )
            )

        merged_boxes = (
            self._merge_nearby_regions(
                raw_boxes
            )
        )

        regions: List[MathRegion] = []

        for x0, y0, x1, y1 in merged_boxes:

            px0 = max(
                0,
                x0 - self.padding,
            )

            py0 = max(
                0,
                y0 - self.padding,
            )

            px1 = min(
                img_w,
                x1 + self.padding,
            )

            py1 = min(
                img_h,
                y1 + self.padding,
            )

            regions.append(
                MathRegion(
                    bbox=(
                        px0,
                        py0,
                        px1,
                        py1,
                    ),
                    confidence=0.70,
                )
            )

        regions.sort(
            key=lambda region: (
                region.bbox[1],
                region.bbox[0],
            )
        )

        return regions

    def _merge_nearby_regions(
        self,
        boxes: List[
            Tuple[int, int, int, int]
        ],
    ) -> List[
        Tuple[int, int, int, int]
    ]:

        if not boxes:
            return []

        boxes = sorted(
            boxes,
            key=lambda box: (
                box[1],
                box[0],
            ),
        )

        changed = True

        while changed:

            changed = False

            merged: List[
                Tuple[int, int, int, int]
            ] = []

            used = [False] * len(boxes)

            for i in range(
                len(boxes)
            ):

                if used[i]:
                    continue

                current = boxes[i]
                used[i] = True

                for j in range(
                    i + 1,
                    len(boxes),
                ):

                    if used[j]:
                        continue

                    if self._boxes_close(
                        current,
                        boxes[j],
                    ):

                        current = self._union(
                            current,
                            boxes[j],
                        )

                        used[j] = True
                        changed = True

                merged.append(
                    current
                )

            boxes = merged

        return boxes

    def _boxes_close(
        self,
        a: Tuple[int, int, int, int],
        b: Tuple[int, int, int, int],
    ) -> bool:

        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b

        gap_x = max(
            0,
            max(ax0, bx0)
            - min(ax1, bx1),
        )

        gap_y = max(
            0,
            max(ay0, by0)
            - min(ay1, by1),
        )

        return (
            gap_x <= self.merge_gap_x
            and gap_y <= self.merge_gap_y
        )

    @staticmethod
    def _union(
        a: Tuple[int, int, int, int],
        b: Tuple[int, int, int, int],
    ) -> Tuple[int, int, int, int]:

        return (
            min(a[0], b[0]),
            min(a[1], b[1]),
            max(a[2], b[2]),
            max(a[3], b[3]),
        )


# ==========================================================
# PIX2TEX CONVERTER
# ==========================================================

class Pix2TexConverter:
    """
    Lazy-loaded Pix2Tex wrapper.

    The model is NOT loaded when this file is imported.

    It loads only when mathematical extraction is actually required.
    """

    def __init__(self):
        self._model = None

    def _load(self):

        if self._model is None:

            logger.info(
                "Loading Pix2Tex LatexOCR model..."
            )

            from pix2tex.cli import LatexOCR

            self._model = LatexOCR()

        return self._model

    def convert(
        self,
        image: Image.Image,
    ) -> str:

        try:

            model = self._load()

            image = image.convert(
                "RGB"
            )

            latex = model(image)

            if not latex:
                return ""

            return str(
                latex
            ).strip()

        except Exception as exc:

            logger.error(
                "Pix2Tex conversion failed: %s",
                exc,
            )

            return ""


# ==========================================================
# MAIN MATH EXTRACTOR
# ==========================================================

class MathExtractor:
    """
    Public mathematical extraction service.

    Workflow:

        image
          |
          v
    Try complete image
          |
          +---- success ----> return full equation
          |
          +---- failure ----> detect regions
                                |
                                v
                              Pix2Tex
    """

    def __init__(
        self,
        detector: Optional[
            MathRegionDetector
        ] = None,
        converter: Optional[
            Pix2TexConverter
        ] = None,
    ):

        self.detector = (
            detector
            or MathRegionDetector()
        )

        self.converter = (
            converter
            or Pix2TexConverter()
        )

    # ------------------------------------------------------
    # FULL IMAGE EXTRACTION
    # ------------------------------------------------------

    def extract_full_image(
        self,
        image: Image.Image,
    ) -> Optional[MathResult]:
        """
        Run Pix2Tex directly on the complete image.

        Best for:

            - equation screenshots
            - textbook equations
            - mathematical formula images
            - scanned equations
        """

        image = image.convert(
            "RGB"
        )

        if (
            image.width < 20
            or image.height < 20
        ):
            return None

        latex = self.converter.convert(
            image
        )

        if not latex.strip():
            return None

        return MathResult(
            bbox=(
                0,
                0,
                image.width,
                image.height,
            ),
            latex=latex.strip(),
            confidence=0.90,
        )

    # ------------------------------------------------------
    # REGION EXTRACTION
    # ------------------------------------------------------

    def extract_regions(
        self,
        image: Image.Image,
    ) -> List[MathResult]:
        """
        Detect mathematical regions and convert each one to LaTeX.
        """

        image = image.convert(
            "RGB"
        )

        regions = self.detector(
            image
        )

        if not regions:
            return []

        results: List[
            MathResult
        ] = []

        for region in regions:

            crop = image.crop(
                region.bbox
            )

            if (
                crop.width < 20
                or crop.height < 15
            ):
                continue

            latex = self.converter.convert(
                crop
            )

            if not latex.strip():
                continue

            results.append(
                MathResult(
                    bbox=region.bbox,
                    latex=latex.strip(),
                    confidence=region.confidence,
                )
            )

        return results

    # ------------------------------------------------------
    # PUBLIC METHOD
    # ------------------------------------------------------

    def extract(
        self,
        image: Image.Image,
    ) -> List[MathResult]:
        """
        Main extraction method.

        IMPORTANT:

        Full-image extraction is attempted first because Pix2Tex
        performs much better when a complete equation is provided.

        Region detection is used only as a fallback.
        """

        image = image.convert(
            "RGB"
        )

        # ==============================================
        # 1. TRY COMPLETE IMAGE FIRST
        # ==============================================

        full_result = (
            self.extract_full_image(
                image
            )
        )

        if full_result is not None:

            return [
                full_result
            ]

        # ==============================================
        # 2. FALLBACK TO REGION DETECTION
        # ==============================================

        return self.extract_regions(
            image
        )