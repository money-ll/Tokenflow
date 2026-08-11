"""
math_extractor.py

Shared math-region-detection + LaTeX-conversion module.

Used by BOTH the PDF pipeline and the image pipeline. Neither caller needs
to know pix2tex is involved -- they just hand this module an image and get
back a list of (bounding_box, latex_string) pairs.

    PDF pipeline:   render page -> image -> MathExtractor.extract(image)
    Image pipeline: uploaded image -> MathExtractor.extract(image)

Splicing the LaTeX back into surrounding text is intentionally NOT done
here (see project decision: return raw (bbox, latex) pairs, let each
caller splice using whatever positional text data it already has).
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class MathRegion:
    """A single detected math region and its bounding box, in pixel
    coordinates of the image that was passed in to extract()."""
    bbox: Tuple[int, int, int, int]   # (x0, y0, x1, y1)
    confidence: float


@dataclass
class MathResult:
    """A detected region plus its converted LaTeX string."""
    bbox: Tuple[int, int, int, int]
    latex: str
    confidence: float


class MathRegionDetector:
    """
    Detects candidate math-expression regions in a raster image.

    Default implementation is a lightweight heuristic (no training data
    required) so the pipeline is runnable end-to-end today. It looks for
    connected-component clusters with statistics typical of math (dense
    small glyphs, isolated from surrounding paragraph text, wide
    aspect-ratio-appropriate blocks) rather than trying to do real
    detection.

    A post-detection merge step then combines boxes that are close
    together (e.g. a lone exponent or "=" sign sitting apart from the
    rest of the same equation) into single regions, so pix2tex receives
    whole expressions rather than disconnected pieces.

    This is deliberately swappable: replace __call__ with a call to a
    trained detector (e.g. a YOLO/DETR model fine-tuned for formula
    detection, or Pix2Text's MFD model) without changing anything in
    MathExtractor or in the calling pipelines.
    """

    def __init__(
        self,
        min_region_area: int = 200,
        padding: int = 6,
        merge_gap_x: int = 40,
        merge_gap_y: int = 15,
    ):
        self.min_region_area = min_region_area
        self.padding = padding
        # Max gap (in pixels, on the *unpadded* boxes) between two boxes
        # for them to be considered part of the same expression and
        # merged together. merge_gap_x is generous since symbols on the
        # same line (e.g. an "=" sign, an exponent) can have visible
        # whitespace around them; merge_gap_y is tighter since we don't
        # want to merge separate lines/equations that happen to sit
        # close together vertically.
        self.merge_gap_x = merge_gap_x
        self.merge_gap_y = merge_gap_y

    def __call__(self, image: Image.Image) -> List[MathRegion]:
        try:
            import cv2
        except ImportError:
            logger.warning(
                "opencv-python not installed; math region detection "
                "returning no regions. Install opencv-python or plug in "
                "a trained detector."
            )
            return []

        gray = np.array(image.convert("L"))
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Dilate to merge nearby glyph fragments (subscripts, symbols,
        # fraction bars) into single connected blocks.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 10))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        raw_boxes = []
        img_w, img_h = image.size

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < self.min_region_area:
                continue

            # Heuristic filters: skip regions that look like full lines of
            # ordinary prose (very wide, short) or full-page noise.
            aspect_ratio = w / max(h, 1)
            if aspect_ratio > 25:
                continue
            if w > 0.95 * img_w and h < 40:
                continue

            raw_boxes.append((x, y, x + w, y + h))

        merged_boxes = self._merge_nearby_regions(raw_boxes)

        regions = []
        for (x0, y0, x1, y1) in merged_boxes:
            px0 = max(0, x0 - self.padding)
            py0 = max(0, y0 - self.padding)
            px1 = min(img_w, x1 + self.padding)
            py1 = min(img_h, y1 + self.padding)
            regions.append(MathRegion(bbox=(px0, py0, px1, py1), confidence=0.5))

        regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))  # reading order
        return regions

    def _merge_nearby_regions(
        self, boxes: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Merge boxes that are close enough together to plausibly belong
        to the same expression (e.g. a lone exponent or "=" sign
        sitting apart from the rest of an equation). Repeats until no
        more merges happen, since merging two boxes can bring a third
        box within range.

        Two boxes are merged if the horizontal gap between them is
        <= merge_gap_x AND the vertical gap is <= merge_gap_y.
        """
        if not boxes:
            return []

        boxes = list(boxes)
        changed = True

        while changed:
            changed = False
            merged = []
            used = [False] * len(boxes)

            for i in range(len(boxes)):
                if used[i]:
                    continue
                current = boxes[i]
                used[i] = True

                for j in range(i + 1, len(boxes)):
                    if used[j]:
                        continue
                    if self._boxes_close(current, boxes[j]):
                        current = self._union(current, boxes[j])
                        used[j] = True
                        changed = True

                merged.append(current)

            boxes = merged

        return boxes

    def _boxes_close(
        self,
        a: Tuple[int, int, int, int],
        b: Tuple[int, int, int, int],
    ) -> bool:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b

        # Horizontal gap: 0 if they overlap on the x-axis, else the
        # distance between the closest edges.
        gap_x = max(0, max(ax0, bx0) - min(ax1, bx1))
        # Vertical gap: same idea on the y-axis.
        gap_y = max(0, max(ay0, by0) - min(ay1, by1))

        return gap_x <= self.merge_gap_x and gap_y <= self.merge_gap_y

    def _union(
        self,
        a: Tuple[int, int, int, int],
        b: Tuple[int, int, int, int],
    ) -> Tuple[int, int, int, int]:
        return (
            min(a[0], b[0]),
            min(a[1], b[1]),
            max(a[2], b[2]),
            max(a[3], b[3]),
        )


class Pix2TexConverter:
    """Thin wrapper around pix2tex's LatexOCR model. Loaded lazily so
    importing this module doesn't require a GPU/model download unless
    conversion is actually used."""

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            from pix2tex.cli import LatexOCR
            logger.info("Loading pix2tex LatexOCR model...")
            self._model = LatexOCR()
        return self._model

    def convert(self, cropped_image: Image.Image) -> str:
        try:
            model = self._load()
        except Exception as exc:
            logger.error("pix2tex model unavailable: %s", exc)
            return ""
        try:
            return model(cropped_image)
        except Exception as exc:
            logger.error("pix2tex conversion failed: %s", exc)
            return ""


class MathExtractor:
    """
    Public entry point used by both the PDF pipeline and the image
    pipeline.

    Usage:
        extractor = MathExtractor()
        results = extractor.extract(image)
        for r in results:
            print(r.bbox, r.latex)
    """

    def __init__(
        self,
        detector: Optional[MathRegionDetector] = None,
        converter: Optional[Pix2TexConverter] = None,
    ):
        self.detector = detector or MathRegionDetector()
        self.converter = converter or Pix2TexConverter()

    def extract(self, image: Image.Image) -> List[MathResult]:
        """
        Detect math regions in `image` and convert each to LaTeX.

        Args:
            image: a PIL Image. For the PDF pipeline this is a rendered
                page (see extractor.py); for the image pipeline this is
                the raw uploaded image.

        Returns:
            List of MathResult, one per detected region, sorted in
            reading order (top-to-bottom, left-to-right). Callers are
            responsible for splicing these back into surrounding text
            using the bbox coordinates.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        regions = self.detector(image)
        if not regions:
            return []

        results = []
        for region in regions:
            crop = image.crop(region.bbox)
            if crop.width < 4 or crop.height < 4:
                continue
            latex = self.converter.convert(crop)
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