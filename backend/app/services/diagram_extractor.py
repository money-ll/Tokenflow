"""
diagram_extractor.py

Detects flowchart/block-diagram structure (boxes + connecting lines) in a
raster image and converts it to Mermaid flowchart syntax.

Used by extractor.py, mirroring math_extractor.py's structure: a detector
finds candidate regions, labels are read via OCR, and the resulting graph
is rendered as text.

IMPORTANT -- this is a heuristic CV pipeline (contour-based box detection
+ Hough line detection), NOT a trained diagram-understanding model. No
lightweight model reliably does true diagram understanding today. This
works well on clean, computer-generated flowcharts (the common case for
figures embedded in reports -- vector-rendered System Architecture, DFD,
Class Diagram style figures) and is unreliable on hand-drawn or heavily
stylized diagrams.

Known limitations (stated, not hidden):
  - Edge DIRECTION is inferred from line geometry (predominantly
    top-to-bottom, falling back to left-to-right for near-horizontal
    connectors), NOT from actual arrowhead detection. Diagrams that flow
    bottom-to-top or right-to-left will have some edges reversed.
  - Only box/rectangle shapes are modeled. Diamonds (decision nodes),
    circles, and other flowchart shapes are detected as boxes if their
    bounding rectangle passes the polygon-approximation filter, or missed
    entirely otherwise.
  - Long elbowed (right-angle) connector routes are detected as several
    separate Hough line segments; node-matching handles this reasonably
    for short routes but can miss very long ones.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class DiagramNode:
    id: str
    bbox: Tuple[int, int, int, int]
    label: str = ""


@dataclass
class DiagramEdge:
    source_id: str
    target_id: str


@dataclass
class DiagramGraph:
    nodes: List[DiagramNode] = field(default_factory=list)
    edges: List[DiagramEdge] = field(default_factory=list)


class NodeDetector:
    """
    Detects box-like regions (rectangles / rounded rectangles) via
    contour detection + polygon approximation. Same family of technique
    as MathRegionDetector in math_extractor.py, tuned for flowchart node
    shapes instead of dense-glyph math regions.
    """

    def __init__(
        self,
        min_width: int = 40,
        min_height: int = 20,
        max_node_area_ratio: float = 0.35,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.max_node_area_ratio = max_node_area_ratio

    def __call__(
        self, image: Image.Image
    ) -> List[Tuple[int, int, int, int]]:

        try:
            import cv2
        except ImportError:
            logger.warning(
                "opencv-python not installed; diagram node detection "
                "returning no regions."
            )
            return []

        gray = np.array(image.convert("L"))
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        contours, _ = cv2.findContours(
            binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        img_w, img_h = image.size
        img_area = max(1, img_w * img_h)

        boxes = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w < self.min_width or h < self.min_height:
                continue

            area = w * h
            if area / img_area > self.max_node_area_ratio:
                continue

            aspect = w / max(h, 1)
            if not (0.3 < aspect < 8):
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Boxes (including rounded-corner boxes) approximate to
            # somewhere between a plain rectangle (4 points) and a
            # rounded rectangle (up to ~8 points from corner curvature).
            if not (4 <= len(approx) <= 8):
                continue

            boxes.append((x, y, x + w, y + h))

        return self._deduplicate_nested(boxes)

    @staticmethod
    def _deduplicate_nested(
        boxes: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Contour detection on a box outline frequently returns both the
        outer and inner edge of the same stroke as separate contours.
        Drop boxes that are near-fully contained within a larger box
        already kept -- keep the outer one.
        """
        boxes = sorted(
            boxes,
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
            reverse=True,
        )

        kept: List[Tuple[int, int, int, int]] = []

        for box in boxes:
            x0, y0, x1, y1 = box
            is_nested = False

            for kx0, ky0, kx1, ky1 in kept:
                if (
                    kx0 - 3 <= x0
                    and ky0 - 3 <= y0
                    and x1 <= kx1 + 3
                    and y1 <= ky1 + 3
                ):
                    is_nested = True
                    break

            if not is_nested:
                kept.append(box)

        return kept


class EdgeDetector:
    """
    Detects connector lines between already-detected node boxes via
    Hough line detection, on the image with node interiors (plus a
    small padding margin) masked out so box borders themselves are
    never mistaken for connectors.

    Direction is inferred from line geometry, not arrowhead shape:
    lines are treated as flowing top-to-bottom unless they are close
    to horizontal (a near-horizontal connector is treated as
    left-to-right), since the overwhelming majority of flowchart
    connectors -- including diagonal ones in a top-down layout --
    represent downward flow. This is a heuristic, not a guarantee.
    """

    def __init__(
        self,
        mask_padding: int = 6,
        hough_threshold: int = 20,
        min_line_length: int = 15,
        max_line_gap: int = 5,
        node_match_tolerance: int = 25,
        horizontal_dy_ratio: float = 0.25,
    ):
        self.mask_padding = mask_padding
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.node_match_tolerance = node_match_tolerance
        self.horizontal_dy_ratio = horizontal_dy_ratio

    def __call__(
        self,
        image: Image.Image,
        nodes: Dict[str, Tuple[int, int, int, int]],
    ) -> List[Tuple[str, str]]:

        try:
            import cv2
        except ImportError:
            return []

        gray = np.array(image.convert("L"))
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        mask = binary.copy()
        pad = self.mask_padding

        for (x0, y0, x1, y1) in nodes.values():
            mask[
                max(0, y0 - pad):y1 + pad,
                max(0, x0 - pad):x1 + pad,
            ] = 0

        lines = cv2.HoughLinesP(
            mask,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        if lines is None:
            return []

        edges = set()

        for line in lines:
            x1, y1, x2, y2 = line[0]

            n1 = self._nearest_node(x1, y1, nodes)
            n2 = self._nearest_node(x2, y2, nodes)

            if not n1 or not n2 or n1 == n2:
                continue

            dx, dy = x2 - x1, y2 - y1

            if (
                abs(dx) > 0
                and abs(dy) / max(abs(dx), 1) < self.horizontal_dy_ratio
            ):
                edge = (n1, n2) if x1 <= x2 else (n2, n1)
            else:
                edge = (n1, n2) if y1 <= y2 else (n2, n1)

            edges.add(edge)

        return sorted(edges)

    def _nearest_node(
        self,
        px: int,
        py: int,
        nodes: Dict[str, Tuple[int, int, int, int]],
    ) -> Optional[str]:

        best_id = None
        best_dist = self.node_match_tolerance

        for node_id, (x0, y0, x1, y1) in nodes.items():
            dx = max(x0 - px, 0, px - x1)
            dy = max(y0 - py, 0, py - y1)
            dist = (dx ** 2 + dy ** 2) ** 0.5

            if dist < best_dist:
                best_dist = dist
                best_id = node_id

        return best_id


class DiagramExtractor:
    """
    Public entry point. Detects box+connector structure in an image and
    returns a DiagramGraph with OCR'd labels, or None if too few
    nodes/edges were found (structural plausibility gating beyond this
    -- e.g. rejecting dense tables -- lives in extractor.py, which has
    the image-size context needed for area-ratio checks).
    """

    # Negative padding shrinks the OCR crop inward from the detected
    # box edges, so the box's own border stroke isn't fed to OCR
    # alongside the label text.
    LABEL_CROP_PADDING = -3

    def __init__(
        self,
        node_detector: Optional[NodeDetector] = None,
        edge_detector: Optional[EdgeDetector] = None,
        label_recognizer=None,
    ):
        self.node_detector = node_detector or NodeDetector()
        self.edge_detector = edge_detector or EdgeDetector()
        # label_recognizer: callable(PIL.Image) -> str. Injected by the
        # caller (extractor.py passes its existing PrintedTextRecognizer)
        # so this module doesn't own a second OCR model instance.
        self._label_recognizer = label_recognizer

    def extract(self, image: Image.Image, label_provider=None) -> Optional[DiagramGraph]:

        if image.mode != "RGB":
            image = image.convert("RGB")

        boxes = self.node_detector(image)

        if len(boxes) < 2:
            return None

        node_map: Dict[str, Tuple[int, int, int, int]] = {}
        nodes: List[DiagramNode] = []

        for i, bbox in enumerate(boxes):
            node_id = f"N{i}"
            if label_provider is not None:
                try:
                    label = (label_provider(bbox) or "").strip()
                except Exception:
                    label = self._read_label(image, bbox)
            else:
                label = self._read_label(image, bbox)
            node_map[node_id] = bbox
            nodes.append(
                DiagramNode(id=node_id, bbox=bbox, label=label)
            )

        edge_pairs = self.edge_detector(image, node_map)

        if not edge_pairs:
            return None

        edges = [
            DiagramEdge(source_id=a, target_id=b)
            for a, b in edge_pairs
        ]

        return DiagramGraph(nodes=nodes, edges=edges)

    def _read_label(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
    ) -> str:

        if self._label_recognizer is None:
            return ""

        x0, y0, x1, y1 = bbox
        pad = self.LABEL_CROP_PADDING

        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(image.width, x1 + pad)
        y1 = min(image.height, y1 + pad)

        if x1 <= x0 or y1 <= y0:
            return ""

        crop = image.crop((x0, y0, x1, y1))

        try:
            text = self._label_recognizer(crop)
        except Exception as exc:
            logger.warning("Diagram label OCR failed: %s", exc)
            return ""

        return (text or "").strip()


class MermaidRenderer:
    """
    Renders a DiagramGraph as Mermaid flowchart syntax.
    """

    MAX_LABEL_LENGTH = 60

    def render(self, graph: DiagramGraph) -> str:

        direction = self._infer_direction(graph)

        lines = [f"graph {direction}"]

        connected_ids = set()
        for edge in graph.edges:
            connected_ids.add(edge.source_id)
            connected_ids.add(edge.target_id)

        for node in graph.nodes:
            if node.id not in connected_ids:
                # Isolated nodes (no detected edges) are dropped --
                # in practice these are almost always detection noise
                # (a stray box-like contour), not a genuine
                # disconnected part of the diagram.
                continue

            label = self._sanitize_label(node.label) or node.id
            lines.append(f'    {node.id}["{label}"]')

        for edge in graph.edges:
            lines.append(f"    {edge.source_id} --> {edge.target_id}")

        return "\n".join(lines)

    @classmethod
    def _sanitize_label(cls, label: str) -> str:

        if not label:
            return ""

        label = " ".join(label.split())
        label = label.replace('"', "'").replace("`", "'")
        label = label[: cls.MAX_LABEL_LENGTH]

        return label

    @staticmethod
    def _infer_direction(graph: DiagramGraph) -> str:
        """
        Picks Mermaid's top-down (TD) vs left-right (LR) layout based
        on whether edges are predominantly vertical or horizontal.
        """
        node_map = {n.id: n for n in graph.nodes}

        vertical_votes = 0
        horizontal_votes = 0

        for edge in graph.edges:
            src = node_map.get(edge.source_id)
            dst = node_map.get(edge.target_id)

            if not src or not dst:
                continue

            sx0, sy0, sx1, sy1 = src.bbox
            dx0, dy0, dx1, dy1 = dst.bbox

            scx, scy = (sx0 + sx1) / 2, (sy0 + sy1) / 2
            dcx, dcy = (dx0 + dx1) / 2, (dy0 + dy1) / 2

            if abs(dcy - scy) >= abs(dcx - scx):
                vertical_votes += 1
            else:
                horizontal_votes += 1

        return "LR" if horizontal_votes > vertical_votes else "TD"