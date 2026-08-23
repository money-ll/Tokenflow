"""
TokenFlow Input Extraction Service

Supported:

    TXT
    PDF
    PNG
    JPG
    JPEG

Image processing:

    Printed image
        -> EasyOCR

    Handwritten image
        -> TrOCR
        -> EasyOCR fallback

    Photo
        -> EasyOCR attempt
        -> BLIP image description fallback

    Mixed/contextual
        -> EasyOCR
        -> TrOCR where appropriate
        -> BLIP fallback

The extractor preserves image descriptions as usable textual
input so photographs can continue through TokenFlow's normal
optimization pipeline.

Math handling:

    Any image or rendered/scanned PDF page is also passed through
    MathExtractor. Detected formulas are converted to LaTeX and
    appended as "Equation: <latex>" lines, but only after passing
    a validation filter (_looks_like_math) that rejects page
    numbers, UI chrome, tiny hallucinated fragments, and other
    pix2tex hallucinations on non-math crops. TokenFlowPipeline
    pulls the surviving lines out before compression and restores
    them in place afterward.

Diagram handling:

    Images, scanned/rendered PDF pages, AND embedded raster images
    on typed PDF pages (e.g. a flowchart figure sitting inside an
    otherwise-typed page) are passed through DiagramExtractor. This
    is a heuristic CV pipeline (box detection + connector-line
    detection), not a trained diagram-understanding model -- see
    diagram_extractor.py's module docstring for stated limitations.
    Detected diagrams are rendered as Mermaid syntax and wrapped in
    "[DIAGRAM_START] ... [DIAGRAM_END]" markers. TokenFlowPipeline
    protects the whole block from compression and restores it
    verbatim afterward, same mechanism as equations.

OCR noise handling:

    OCR-derived text (scanned PDF pages, image OCR) is passed
    through _strip_ocr_noise to drop lines that are almost
    certainly UI chrome (phone status bars, stray icon glyphs)
    rather than genuine document content. Native typed PDF text
    is never touched by this filter.
"""

from __future__ import annotations

import io
import os
import re
from pathlib import Path

import pymupdf
from PIL import Image

from app.services.handwriting import (
    HandwritingRecognizer,
)

from app.services.printed_ocr import (
    PrintedTextRecognizer,
)

from app.services.imageprocessing import (
    ImageProcessor,
)

from app.services.math_extractor import (
    MathExtractor,
)

from app.services.diagram_extractor import (
    DiagramExtractor,
    MermaidRenderer,
)


class InputExtractor:

    SUPPORTED_TEXT = {
        ".txt",
    }

    SUPPORTED_PDF = {
        ".pdf",
    }

    SUPPORTED_IMAGE = {
        ".png",
        ".jpg",
        ".jpeg",
    }

    SUPPORTED = (
        SUPPORTED_TEXT
        | SUPPORTED_PDF
        | SUPPORTED_IMAGE
    )

    MIN_TYPED_CHARS_PER_PAGE = 20

    # 150 DPI on a Letter-size page renders ~1275x1650 -- more
    # pixels than EasyOCR needs for normal body text, and EasyOCR's
    # cost scales roughly with pixel count. 120 DPI (~1020x1320)
    # cuts the OCR canvas by ~36% while staying well above the
    # resolution needed to read ordinary printed text; drop lower
    # only if you start seeing missed characters on small fonts.
    DEFAULT_OCR_RENDER_DPI = 120

    # ==========================================================
    # MATH VALIDATION CONSTANTS
    # ==========================================================

    _STRIP_CMD_RE = re.compile(r"\\[a-zA-Z]+")
    _STRIP_BRACKETS_RE = re.compile(r"[{}\[\]()]")

    _MATH_STRUCTURE_TOKENS = (
        "\\frac", "\\sum", "\\int", "\\sqrt", "\\prod", "\\lim",
        "^", "_", "=",
        "\\alpha", "\\beta", "\\gamma", "\\theta", "\\pi", "\\sigma", "\\mu",
    )

    # Absolute pixel-size floor for a math crop. A genuine formula
    # with structure (fraction bar, summation, etc.) needs enough
    # room to actually render that structure -- tiny crops (badges,
    # score digits, icons) that still produce syntactically-plausible
    # LaTeX are almost always hallucinations, not real formulas.
    _MIN_EQUATION_WIDTH = 50
    _MIN_EQUATION_HEIGHT = 30

    # ==========================================================
    # DIAGRAM VALIDATION CONSTANTS
    # ==========================================================

    # A detection with more "nodes" than this is almost always a
    # table, a dense UI screenshot, or grid noise -- not a genuine
    # flowchart/block diagram (which in report figures rarely exceeds
    # a dozen or so boxes).
    _MAX_DIAGRAM_NODES = 20

    # A real flowchart has visible whitespace/gaps between boxes. A
    # dense table or grid of UI elements fills most of its frame with
    # "node"-like rectangles. Rejecting high area coverage filters
    # those out without needing to distinguish table cells from
    # flowchart boxes directly.
    _MAX_DIAGRAM_TOTAL_NODE_AREA_RATIO = 0.75

    # Embedded PDF images smaller than this (in either dimension) are
    # skipped for diagram detection -- logos, icons, and bullet
    # graphics are common in typed pages and not worth the pipeline
    # cost, and are a frequent source of false positives.
    _MIN_EMBEDDED_IMAGE_DIMENSION = 150

    # Typed-PDF layout detection. Vector diagrams are often drawn directly
    # into the PDF (no embedded raster image exists), so get_images() alone
    # cannot find them. These cheap thresholds let us inspect PyMuPDF
    # drawing objects first and render only plausible figure regions.
    _MIN_VECTOR_DIAGRAM_BOXES = 2
    _MIN_VECTOR_DIAGRAM_LINES = 1
    _MIN_VECTOR_DIAGRAM_AREA = 0.015
    _MAX_VECTOR_DIAGRAM_AREA = 0.80
    _VECTOR_RENDER_DPI = 160

    # ==========================================================
    # OCR NOISE FILTER CONSTANTS
    # ==========================================================

    # Matches phone status-bar time readouts like "01.23", "01*24".
    _STATUS_BAR_RE = re.compile(r"^\d{1,2}[.:*]\d{2}\b")

    def __init__(self) -> None:
        """
        Create lightweight service objects.

        Heavy OCR/captioning/math/diagram models remain lazy-loaded.
        """

        # These objects should not load the actual models
        # unless their recognize methods are called.

        self._handwriting = (
            HandwritingRecognizer()
        )

        self._printed_ocr = (
            PrintedTextRecognizer()
        )

        # IMPORTANT:
        #
        # No BLIP object is created here.
        #
        # ImageProcessor creates PhotoCaptioner only
        # when captioning becomes necessary.

        self._image_processor = (
            ImageProcessor(
                handwriting_recognizer=(
                    self._handwriting
                ),
                printed_recognizer=(
                    self._printed_ocr
                ),
            )
        )

        # MathExtractor is cheap to construct: the region detector
        # only imports cv2 when called, and the LatexOCR model is
        # loaded lazily on the first .convert() call.

        self._math_extractor = (
            MathExtractor()
        )

        # DiagramExtractor is likewise cheap to construct: its
        # detectors only import cv2 when called. Its label OCR is
        # injected as a callable wrapping the SAME PrintedTextRecognizer
        # instance used for page OCR, so there is no second EasyOCR
        # model loaded -- diagram label reads just become additional
        # calls against the one shared reader.

        self._diagram_extractor = (
            DiagramExtractor(
                label_recognizer=(
                    lambda crop: self._printed_ocr.recognize(crop)
                ),
            )
        )

        self._mermaid_renderer = (
            MermaidRenderer()
        )

    # ==========================================================
    # PUBLIC ENTRY POINT
    # ==========================================================

    def extract(
        self,
        filename: str,
        content: bytes,
    ):

        ext = (
            Path(filename)
            .suffix
            .lower()
        )

        if ext not in self.SUPPORTED:

            raise ValueError(
                "Only .txt, .pdf, and "
                ".png/.jpg/.jpeg files "
                "are currently supported."
            )

        # ======================================================
        # TXT
        # ======================================================

        if ext in self.SUPPORTED_TEXT:

            text = content.decode(
                "utf-8",
                errors="replace",
            )

            return (
                text,
                {
                    "source_type": "text",
                },
            )

        # ======================================================
        # IMAGE
        # ======================================================

        if ext in self.SUPPORTED_IMAGE:

            return self._extract_image(
                content
            )

        # ======================================================
        # PDF
        # ======================================================

        return self._extract_pdf(
            content
        )

    # ==========================================================
    # MATH DETECTION
    # ==========================================================

    def _detect_equations(
        self,
        image: Image.Image,
    ) -> list[str]:
        """
        Run MathExtractor on a page/photo image and format any
        detected formulas as 'Equation: <latex>' lines.

        Each detection is validated with _looks_like_math() before
        being trusted -- the region detector is a generic dense-glyph
        heuristic (not a math classifier), so it fires on UI chrome,
        page numbers, and icons too, and pix2tex will happily
        "convert" whatever crop it's handed. Anything that fails
        validation is dropped rather than emitted as a fake equation.

        These lines are matched by
        TokenFlowPipeline._inline_protected_blocks() and protected
        from compression, then restored in place afterward.

        Best-effort: any failure (missing opencv-python, missing
        pix2tex, model load failure, etc.) returns an empty list
        rather than breaking extraction.
        """

        try:

            results = (
                self._math_extractor.extract(
                    image
                )
            )

        except Exception as exc:

            import logging

            logging.getLogger(__name__).warning(
                "Math extraction failed, skipping: %s", exc
            )

            return []

        equations = []

        for r in results:

            latex = r.latex.strip()

            if self._looks_like_math(
                latex,
                r.bbox,
                image.size,
            ):

                equations.append(
                    f"Equation: {latex}"
                )

        return equations

    # ==========================================================
    # MATH VALIDATION
    # ==========================================================

    @classmethod
    def _looks_like_math(
        cls,
        latex: str,
        bbox,
        image_size,
    ) -> bool:
        """
        Reject pix2tex outputs that are almost certainly not real
        math: bare page numbers, formatting-only commands with no
        math structure, unbalanced/malformed LaTeX, autoregressive
        repetition loops, crops too small to physically contain the
        structure they claim to show, and crops spanning nearly the
        whole image.
        """

        if not latex:

            return False

        x0, y0, x1, y1 = bbox
        width = x1 - x0
        height = y1 - y0

        # ------------------------------------------------------
        # Absolute size floor. A genuine formula with structure
        # (fraction bar, summation, etc.) needs enough physical
        # room to render that structure -- tiny crops producing
        # "\frac{...}" or similar are almost always hallucinations
        # on UI fragments (badges, score digits, icons).
        # ------------------------------------------------------

        if (
            width < cls._MIN_EQUATION_WIDTH
            or height < cls._MIN_EQUATION_HEIGHT
        ):

            return False

        # ------------------------------------------------------
        # Bare numerals, even wrapped in formatting commands like
        # \mathsf{53} -- strip commands/brackets and check if only
        # digits remain. Catches page numbers.
        # ------------------------------------------------------

        core = cls._STRIP_BRACKETS_RE.sub(
            "",
            cls._STRIP_CMD_RE.sub(
                "",
                latex,
            ),
        ).strip()

        if re.fullmatch(r"\d+", core):

            return False

        # ------------------------------------------------------
        # Must contain at least one token that actually signals
        # math structure -- plain formatting commands (\mathsf,
        # \textstyle, \scriptstyle) don't count on their own.
        # ------------------------------------------------------

        if not any(
            tok in latex
            for tok in cls._MATH_STRUCTURE_TOKENS
        ):

            return False

        # ------------------------------------------------------
        # Malformed/unbalanced LaTeX is a strong hallucination
        # signal -- real pix2tex output on genuine formulas is
        # well-formed.
        # ------------------------------------------------------

        if not cls._brackets_balanced(latex):

            return False

        # ------------------------------------------------------
        # Autoregressive loop artifacts (repeated identical
        # chunks) -- e.g. "\frac{-}{\varepsilon}" repeated 4x,
        # "sin(sin(sin(...".
        # ------------------------------------------------------

        if cls._has_excessive_repetition(latex):

            return False

        # ------------------------------------------------------
        # Sanity check output length against crop size: a small
        # crop producing an implausibly long LaTeX string is
        # hallucination, not a real formula.
        # ------------------------------------------------------

        crop_area = max(1, width * height)
        img_area = image_size[0] * image_size[1]

        if (
            len(latex) > 400
            and crop_area < 0.15 * img_area
        ):

            return False

        # ------------------------------------------------------
        # Reject crops covering almost the entire image -- a
        # genuine isolated formula doesn't span nearly the whole
        # page/screenshot. Catches UI screenshots being read
        # whole as "one big equation".
        # ------------------------------------------------------

        if crop_area > 0.85 * img_area:

            return False

        return True

    @staticmethod
    def _brackets_balanced(latex: str) -> bool:

        counts = {c: 0 for c in "{}()[]"}

        for ch in latex:

            if ch in counts:

                counts[ch] += 1

        return (
            counts["{"] == counts["}"]
            and counts["("] == counts[")"]
            and counts["["] == counts["]"]
        )

    @staticmethod
    def _has_excessive_repetition(
        latex: str,
        min_len: int = 8,
        min_repeats: int = 3,
    ) -> bool:

        if len(latex) < min_len:

            return False

        max_chunk = min(
            40,
            len(latex) // min_repeats,
        )

        for size in range(
            min_len,
            max_chunk + 1,
        ):

            seen = {}

            for i in range(
                0,
                len(latex) - size + 1,
            ):

                chunk = latex[i:i + size]

                seen[chunk] = (
                    seen.get(chunk, 0) + 1
                )

                if seen[chunk] >= min_repeats:

                    return True

        return False

    # ==========================================================
    # DIAGRAM DETECTION
    # ==========================================================

    def _detect_diagram(
        self,
        image: Image.Image,
        label_provider=None,
    ) -> str | None:
        """
        Run DiagramExtractor on an image and, if the detected
        structure passes validation, return a fenced diagram block
        ready to be spliced into extracted text:

            [DIAGRAM_START]
```mermaid
            graph TD
                N0["..."]
                N1["..."]
                N0 --> N1
```
            [DIAGRAM_END]

        Returns None if no diagram-like structure was found, or if
        what was found fails _looks_like_diagram() validation (too
        many nodes, nodes covering too much of the frame -- signals
        a table or dense UI screenshot rather than a flowchart, not
        a genuine detection failure).

        Best-effort: any failure (missing opencv-python, OCR failure,
        etc.) returns None rather than breaking extraction.
        """

        try:

            graph = (
                self._diagram_extractor.extract(
                    image,
                    label_provider=label_provider,
                )
            )

        except Exception as exc:

            import logging

            logging.getLogger(__name__).warning(
                "Diagram extraction failed, skipping: %s", exc
            )

            return None

        if graph is None:

            return None

        if not self._looks_like_diagram(
            graph,
            image.size,
        ):

            return None

        mermaid = (
            self._mermaid_renderer.render(
                graph
            )
        )

        return (
            "[DIAGRAM_START]\n"
            f"```mermaid\n{mermaid}\n```\n"
            "[DIAGRAM_END]"
        )

    @classmethod
    def _looks_like_diagram(
        cls,
        graph,
        image_size,
    ) -> bool:
        """
        Rejects detections that are structurally implausible as a
        genuine flowchart/block diagram: too many boxes (usually a
        table or dense grid of UI elements, not a diagram), or node
        area covering most of the frame (a table fills most of its
        image; a real flowchart has visible whitespace between boxes).
        """

        node_count = len(graph.nodes)

        if (
            node_count < 2
            or node_count > cls._MAX_DIAGRAM_NODES
        ):

            return False

        img_area = max(
            1,
            image_size[0] * image_size[1],
        )

        total_node_area = sum(
            (x1 - x0) * (y1 - y0)
            for (x0, y0, x1, y1) in (
                n.bbox for n in graph.nodes
            )
        )

        if (
            total_node_area / img_area
            > cls._MAX_DIAGRAM_TOTAL_NODE_AREA_RATIO
        ):

            return False

        return True

    def _extract_page_equations(self, page) -> list[str]:
        """Extract equations from a typed PDF without OCR'ing the page.

        PyMuPDF already exposes text-line bounding boxes.  We use those
        boxes as a cheap candidate detector and render only the few lines
        that contain real mathematical signals.  The cropped line is then
        sent to pix2tex as one complete expression, preserving fractions,
        superscripts and grouped symbols.
        """
        try:
            data = page.get_text("dict")
        except Exception:
            return []

        lines = []
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(span.get("text", "") for span in spans).strip()
                bbox = line.get("bbox")
                if text and bbox and self._is_math_candidate_text(text):
                    lines.append((bbox, text))

        if not lines:
            return []

        # Merge vertically adjacent candidate lines.  This handles equations
        # whose numerator/denominator or aligned parts are represented as
        # separate PDF text lines.
        groups = []
        for bbox, text in sorted(lines, key=lambda x: (x[0][1], x[0][0])):
            if not groups:
                groups.append([bbox, text])
                continue
            prev = groups[-1][0]
            gap = bbox[1] - prev[3]
            overlap = max(0.0, min(prev[2], bbox[2]) - max(prev[0], bbox[0]))
            min_width = max(1.0, min(prev[2] - prev[0], bbox[2] - bbox[0]))
            if gap <= 8 and overlap / min_width >= 0.20:
                groups[-1][0] = (
                    min(prev[0], bbox[0]), min(prev[1], bbox[1]),
                    max(prev[2], bbox[2]), max(prev[3], bbox[3])
                )
                groups[-1][1] += " " + text
            else:
                groups.append([bbox, text])

        results = []
        seen = set()
        for bbox, _native_text in groups[:12]:
            # Expand enough to include fraction bars and superscripts, but
            # keep the crop local so pix2tex never sees an entire page.
            rect = pymupdf.Rect(bbox)
            rect.x0 -= 8
            rect.y0 -= 8
            rect.x1 += 8
            rect.y1 += 8
            rect &= page.rect
            if rect.width < 45 or rect.height < 18:
                continue
            try:
                pix = page.get_pixmap(dpi=self._VECTOR_RENDER_DPI, clip=rect, alpha=False)
                image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                extracted = self._detect_equations(image)
            except Exception:
                continue
            for equation in extracted:
                key = equation.strip()
                if key and key not in seen:
                    seen.add(key)
                    results.append(equation)
        return results

    @staticmethod
    def _looks_like_math_text(text: str) -> bool:
        """Backward-compatible text-only math candidate heuristic."""
        if not text:
            return False
        t = text.strip()
        if re.search(r"\b(?:E\s*=|[A-Za-z]\s*\^\s*\d|\\(?:frac|sum|int|sqrt))", t):
            return True
        if re.search(r"[∑∫√≤≥≠±×÷∞]", t):
            return bool(re.search(r"\d|[A-Za-z]", t))
        if "=" in t and re.search(r"[A-Za-z]", t) and re.search(r"\d|[\^_]", t):
            return True
        return False

    @staticmethod
    def _is_math_candidate_text(text: str) -> bool:
        """Cheap high-recall equation candidate test; no model invocation."""
        t = text.strip()
        if len(t) < 3 or len(t) > 350:
            return False
        # Strong mathematical operators/symbols.
        strong = sum(ch in "=^_∑∫√≈≤≥≠±×÷∞⋅·|" for ch in t)
        has_digit = bool(re.search(r"\d", t))
        has_math_word = bool(re.search(
            r"\b(cosine|similarity|softmax|logit|probability|formula|equation|average|sum|score)\b",
            t,
            re.IGNORECASE,
        ))
        # '=' plus a numeric/symbolic expression is the strongest signal.
        if "=" in t and (has_digit or strong >= 2):
            return True
        if strong >= 3 and has_digit:
            return True
        if re.search(r"[≤≥≈≠±×÷∞→]", t) and has_digit:
            return True
        # A math-labelled line may have symbols encoded poorly in the PDF.
        if has_math_word and strong >= 1:
            return True
        return False

    @classmethod
    def _remove_math_candidate_lines(cls, text: str) -> str:
        if not text:
            return text
        kept = [
            line for line in text.splitlines()
            if not cls._is_math_candidate_text(line)
        ]
        return "\n".join(kept).strip()

    def _extract_vector_diagram_regions(self, page):
        """Return plausible vector-drawing figure clips on a typed page."""
        try:
            drawings = page.get_drawings()
        except Exception:
            return []
        if not drawings:
            return []

        page_area = max(1.0, page.rect.width * page.rect.height)
        rects = []
        lines = []
        for d in drawings:
            r = d.get("rect")
            if not r:
                continue
            items = d.get("items", [])
            has_rect = any(item and item[0] == "re" for item in items)
            has_line = any(item and item[0] == "l" for item in items)
            area = max(0.0, r.width * r.height)
            if has_rect and area > 25:
                rects.append(r)
            if has_line:
                lines.append(r)

        if len(rects) < self._MIN_VECTOR_DIAGRAM_BOXES or len(lines) < self._MIN_VECTOR_DIAGRAM_LINES:
            return []

        # Cluster drawing objects that are spatially close.  A report page
        # can contain more than one figure, so don't render the entire page.
        clusters = []
        for r in sorted(rects + lines, key=lambda x: (x.y0, x.x0)):
            placed = False
            for i, c in enumerate(clusters):
                expanded = pymupdf.Rect(c)
                expanded.x0 -= 36; expanded.y0 -= 36
                expanded.x1 += 36; expanded.y1 += 36
                if expanded.intersects(r):
                    clusters[i] = pymupdf.Rect(
                        min(c.x0, r.x0), min(c.y0, r.y0),
                        max(c.x1, r.x1), max(c.y1, r.y1)
                    )
                    placed = True
                    break
            if not placed:
                clusters.append(pymupdf.Rect(r))

        # Merge overlapping/nearby clusters transitively.  A connector can
        # touch two boxes and initially create two clusters; they must become
        # one figure region before rendering, otherwise each crop contains
        # only one node and diagram extraction correctly rejects it.
        changed = True
        while changed:
            changed = False
            merged = []
            used = [False] * len(clusters)
            for i, c in enumerate(clusters):
                if used[i]:
                    continue
                current = pymupdf.Rect(c)
                used[i] = True
                for j in range(i + 1, len(clusters)):
                    if used[j]:
                        continue
                    other = clusters[j]
                    expanded = pymupdf.Rect(current)
                    expanded.x0 -= 36; expanded.y0 -= 36
                    expanded.x1 += 36; expanded.y1 += 36
                    if expanded.intersects(other):
                        current = pymupdf.Rect(
                            min(current.x0, other.x0),
                            min(current.y0, other.y0),
                            max(current.x1, other.x1),
                            max(current.y1, other.y1),
                        )
                        used[j] = True
                        changed = True
                merged.append(current)
            clusters = merged

        valid = []
        for c in clusters:
            area_ratio = (c.width * c.height) / page_area
            if self._MIN_VECTOR_DIAGRAM_AREA <= area_ratio <= self._MAX_VECTOR_DIAGRAM_AREA:
                c.x0 = max(page.rect.x0, c.x0 - 10)
                c.y0 = max(page.rect.y0, c.y0 - 10)
                c.x1 = min(page.rect.x1, c.x1 + 10)
                c.y1 = min(page.rect.y1, c.y1 + 10)
                valid.append(c)
        return valid[:6]

    def _extract_page_diagrams(self, doc, page) -> list[str]:
        """Extract both embedded-raster and vector diagrams from typed PDFs."""
        blocks: list[str] = []

        # 1) Embedded raster figures.
        try:
            image_refs = page.get_images(full=True)
        except Exception:
            image_refs = []
        for img_info in image_refs:
            xref = img_info[0]
            try:
                extracted = doc.extract_image(xref)
                embedded = Image.open(io.BytesIO(extracted["image"])).convert("RGB")
            except Exception:
                continue
            if min(embedded.width, embedded.height) < self._MIN_EMBEDDED_IMAGE_DIMENSION:
                continue
            block = self._detect_diagram(embedded)
            if block and block not in blocks:
                blocks.append(block)

        # 2) Vector figures: flowcharts/architectures drawn directly in the
        # PDF have no image xref.  Render only the drawing cluster, not the
        # whole page, and feed that crop through the same detector.
        for clip in self._extract_vector_diagram_regions(page):
            try:
                pix = page.get_pixmap(dpi=self._VECTOR_RENDER_DPI, clip=clip, alpha=False)
                image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            except Exception:
                continue

            # Vector diagrams usually contain real PDF text labels. Reuse
            # that text layer instead of invoking EasyOCR once per node.
            # This is both faster and more accurate than OCR for diagrams
            # such as System Architecture, DFD, ER and class diagrams.
            try:
                words = page.get_text("words") or []
            except Exception:
                words = []

            scale = self._VECTOR_RENDER_DPI / 72.0

            def native_label_provider(pixel_bbox, _clip=clip, _words=words, _scale=scale):
                px0, py0, px1, py1 = pixel_bbox
                page_box = pymupdf.Rect(
                    _clip.x0 + px0 / _scale,
                    _clip.y0 + py0 / _scale,
                    _clip.x0 + px1 / _scale,
                    _clip.y0 + py1 / _scale,
                )
                parts = []
                for word in _words:
                    if len(word) < 5:
                        continue
                    wx0, wy0, wx1, wy1, word_text = word[:5]
                    cx = (wx0 + wx1) / 2
                    cy = (wy0 + wy1) / 2
                    if page_box.contains(pymupdf.Point(cx, cy)):
                        parts.append((wy0, wx0, str(word_text)))
                parts.sort(key=lambda item: (item[0], item[1]))
                return " ".join(item[2] for item in parts)

            block = self._detect_diagram(image, label_provider=native_label_provider)
            if block and block not in blocks:
                blocks.append(block)

        return blocks

    # ==========================================================
    # OCR NOISE FILTER
    # ==========================================================

    def _is_garbage_ocr_line(
        self,
        line: str,
    ) -> bool:
        """
        Flags OCR lines that are almost certainly UI noise (phone
        status bars, stray icon glyphs, battery/signal indicators)
        rather than genuine document content -- short lines
        dominated by symbols and single/double-character fragments
        instead of real words.
        """

        stripped = line.strip()

        if not stripped:

            return False

        tokens = stripped.split()

        if not tokens:

            return False

        total_chars = len(stripped)

        alnum_ratio = (
            sum(
                1
                for c in stripped
                if c.isalnum()
            )
            / total_chars
        )

        real_word_tokens = sum(
            1
            for t in tokens
            if re.fullmatch(
                r"[A-Za-z]{3,}",
                t.strip(".,;:!?'\""),
            )
        )

        real_word_ratio = (
            real_word_tokens
            / len(tokens)
        )

        if self._STATUS_BAR_RE.match(
            stripped
        ):

            return True

        if (
            len(stripped) <= 40
            and real_word_ratio < 0.35
            and alnum_ratio < 0.6
        ):

            return True

        return False

    def _strip_ocr_noise(
        self,
        text: str,
    ) -> str:
        """
        Drops OCR lines that are almost certainly noise, not
        content. Applied ONLY to OCR-derived text (scanned PDF
        pages, image OCR) -- never to native typed PDF text, which
        pymupdf extracts reliably and needs no filtering. This
        doesn't remove any real meaning: there was no meaning in a
        misread status bar to begin with.
        """

        if not text:

            return text

        kept = [
            line
            for line in text.split("\n")
            if not self._is_garbage_ocr_line(line)
        ]

        return "\n".join(kept)

    # ==========================================================
    # IMAGE EXTRACTION
    # ==========================================================

    def _extract_image(
        self,
        content: bytes,
    ):

        try:

            result = (
                self._image_processor.process(
                    content
                )
            )

        except RuntimeError as exc:

            raise ValueError(
                str(exc)
            ) from exc

        except Exception as exc:

            raise ValueError(
                f"Could not process this image: "
                f"{exc}"
            ) from exc

        # ======================================================
        # MATH + DIAGRAM DETECTION
        #
        # Runs independently of the OCR/BLIP outcome above, on the
        # original uploaded image.
        # ======================================================

        equation_lines: list[str] = []
        diagram_block: str | None = None

        try:

            pil_image = (
                Image.open(
                    io.BytesIO(content)
                )
                .convert("RGB")
            )

            equation_lines = (
                self._detect_equations(
                    pil_image
                )
            )

            diagram_block = (
                self._detect_diagram(
                    pil_image
                )
            )

        except Exception:

            equation_lines = []
            diagram_block = None

        text = (
            self._clean_ocr_output(
                result.text
            )
        )

        text = (
            self._strip_ocr_noise(
                text
            )
        )

        description = (
            self._clean_ocr_output(
                result.description
                or ""
            )
        )

        extra_blocks = list(equation_lines)

        if diagram_block:

            extra_blocks.append(
                diagram_block
            )

        # ======================================================
        # CASE 1:
        # OCR successfully extracted text
        # ======================================================

        if (
            result.has_text
            and text
        ):

            meta = (
                dict(
                    result.meta
                    or {}
                )
            )

            meta.update(
                {
                    "confidence": (
                        result.confidence
                    ),
                    "has_text": True,
                    "has_description": bool(
                        description
                    ),
                }
            )

            # Combine OCR text with the visual description
            # (when available) so the downstream optimizer -
            # and therefore the "Optimized Prompt" shown to
            # the user - reflects both what the image SAYS
            # and what it SHOWS, not just the extracted text.

            combined_text = (
                text
                if not description
                else (
                    f"{text}\n\n"
                    f"Image description: "
                    f"{description}"
                )
            )

            if extra_blocks:

                combined_text = (
                    combined_text.rstrip()
                    + "\n\n"
                    + "\n\n".join(extra_blocks)
                )

            return (
                combined_text,
                {
                    "source_type": (
                        result.source_type
                    ),
                    "confidence": (
                        result.confidence
                    ),
                    "has_text": True,
                    "has_description": bool(
                        description
                    ),
                    "description": (
                        description
                        or result.description
                    ),
                    "meta": meta,
                    "math_extraction_used": bool(
                        equation_lines
                    ),
                    "equations_found": len(
                        equation_lines
                    ),
                    "diagram_extraction_used": bool(
                        diagram_block
                    ),
                    "diagrams_found": (
                        1 if diagram_block else 0
                    ),
                },
            )

        # ======================================================
        # CASE 2:
        # No OCR text, but BLIP produced a description
        # ======================================================

        if description:

            meta = (
                dict(
                    result.meta
                    or {}
                )
            )

            meta.update(
                {
                    "confidence": (
                        result.confidence
                    ),
                    "has_text": False,
                    "has_description": True,
                    "description_source": "BLIP",
                }
            )

            # IMPORTANT:
            #
            # The caption becomes the textual representation
            # passed to the downstream TokenFlow optimizer.
            #
            # Example:
            #
            # Image:
            #     cat sitting on a chair
            #
            # Extracted representation:
            #
            #     "a cat sitting on a chair"
            #
            # This means the normal compression pipeline can
            # still process the image-derived information.

            combined_description = description

            if extra_blocks:

                combined_description = (
                    combined_description.rstrip()
                    + "\n\n"
                    + "\n\n".join(extra_blocks)
                )

            return (
                combined_description,
                {
                    "source_type": (
                        result.source_type
                    ),
                    "confidence": (
                        result.confidence
                    ),
                    "has_text": False,
                    "has_description": True,
                    "description": description,
                    "meta": meta,
                    "math_extraction_used": bool(
                        equation_lines
                    ),
                    "equations_found": len(
                        equation_lines
                    ),
                    "diagram_extraction_used": bool(
                        diagram_block
                    ),
                    "diagrams_found": (
                        1 if diagram_block else 0
                    ),
                },
            )

        # ======================================================
        # CASE 3:
        # Nothing useful was extracted
        # ======================================================

        raise ValueError(
            result.description
            or (
                "No reliable text or image "
                "description could be extracted."
            )
        )

    @staticmethod
    def _contains_math_signal(text: str) -> bool:
        if not text:
            return False
        return bool(re.search(
            r"(?:=|\^|_|∑|∫|√|≈|≤|≥|≠|±|×|÷|∞|\\frac|\\sum|\\int|\\sqrt|"
            r"\b(?:cosine|similarity|softmax|logits?|probabilit(?:y|ies)|formula|equation)\b)",
            text,
            re.IGNORECASE,
        ))

    @staticmethod
    def _contains_diagram_signal(text: str) -> bool:
        if not text:
            return False
        return bool(re.search(
            r"\b(?:figure|diagram|architecture|flowchart|DFD|ER\s*-?\s*diagram|"
            r"class\s+diagram|use\s+case|system\s+architecture)\b",
            text,
            re.IGNORECASE,
        ))

    # ==========================================================
    # PDF EXTRACTION
    # ==========================================================

    def _extract_pdf(
        self,
        content: bytes,
    ):

        try:

            doc = pymupdf.open(
                stream=content,
                filetype="pdf",
            )

        except Exception as exc:

            raise ValueError(
                f"Could not open PDF: {exc}"
            ) from exc

        pages = []

        typed_pages = 0
        ocr_pages = 0
        blank_pages = 0
        equations_found = 0
        diagrams_found = 0

        total_pages = len(doc)

        try:

            for (
                page_number,
                page
            ) in enumerate(
                doc,
                start=1,
            ):

                # ==================================================
                # NATIVE TEXT
                # ==================================================

                raw_text = (
                    page.get_text(
                        "text"
                    )
                    or ""
                )

                cleaned_text = (
                    self._remove_repeated_page_noise(
                        raw_text
                    )
                    .strip()
                )

                if (
                    len(cleaned_text)
                    >= self.MIN_TYPED_CHARS_PER_PAGE
                ):

                    page_text = (
                        f"[Page {page_number}]\n"
                        f"{cleaned_text}"
                    )

                    # Typed pages normally use the fast native PDF text
                    # path.  We do NOT OCR the whole page.  Instead, run
                    # targeted math extraction only when the native text
                    # contains a likely equation line, and targeted
                    # diagram extraction only when PyMuPDF reports vector
                    # drawing structure.  This fixes the two common cases
                    # that were previously skipped entirely.
                    page_equations = self._extract_page_equations(page)
                    if page_equations:
                        # Remove the raw/mangled native equation lines so the
                        # LaTeX representation is not duplicated in output.
                        cleaned_without_math = self._remove_math_candidate_lines(cleaned_text)
                        page_text = (
                            f"[Page {page_number}]\n"
                            f"{cleaned_without_math}"
                            + "\n\n"
                            + "\n".join(page_equations)
                        )
                        equations_found += len(page_equations)

                    page_diagrams = self._extract_page_diagrams(doc, page)
                    if page_diagrams:
                        page_text = (
                            page_text.rstrip()
                            + "\n\n"
                            + "\n\n".join(page_diagrams)
                        )
                        diagrams_found += len(page_diagrams)

                    pages.append(page_text)
                    typed_pages += 1
                    continue

                # ==================================================
                # CHECK EMBEDDED IMAGE
                # ==================================================

                has_embedded_image = bool(
                    page.get_images(
                        full=True
                    )
                )

                if not has_embedded_image:

                    blank_pages += 1

                    continue

                # ==================================================
                # RENDER PAGE
                # ==================================================

                dpi = (
                    self._get_ocr_dpi()
                )

                try:

                    pixmap = (
                        page.get_pixmap(
                            dpi=dpi,
                            alpha=False,
                        )
                    )

                    page_image = (
                        Image.open(
                            io.BytesIO(
                                pixmap.tobytes(
                                    "png"
                                )
                            )
                        )
                        .convert("RGB")
                    )

                except Exception as exc:

                    raise ValueError(
                        f"Could not render PDF "
                        f"page {page_number}: "
                        f"{exc}"
                    ) from exc

                # ==================================================
                # OCR
                # ==================================================

                try:

                    ocr_text = (
                        self._printed_ocr
                        .recognize(
                            page_image
                        )
                    )

                except RuntimeError as exc:

                    raise ValueError(
                        str(exc)
                    ) from exc

                except Exception as exc:

                    raise ValueError(
                        f"Could not OCR PDF "
                        f"page {page_number}: "
                        f"{exc}"
                    ) from exc

                ocr_text = (
                    self._clean_ocr_output(
                        ocr_text
                    )
                )

                ocr_text = (
                    self._strip_ocr_noise(
                        ocr_text
                    )
                )

                # ==================================================
                # MATH DETECTION
                #
                # Reuses the already-rendered page_image, so this
                # adds no extra render cost.
                # ==================================================

                equation_lines = []
                if self._contains_math_signal(ocr_text):
                    equation_lines = self._detect_equations(page_image)

                if equation_lines:

                    ocr_text = (
                        (ocr_text or "").rstrip()
                        + "\n\n"
                        + "\n".join(equation_lines)
                    )

                    equations_found += len(
                        equation_lines
                    )

                # ==================================================
                # DIAGRAM DETECTION
                #
                # Also reuses the already-rendered page_image.
                # ==================================================

                diagram_block = None
                if self._contains_diagram_signal(ocr_text):
                    diagram_block = self._detect_diagram(page_image)

                if diagram_block:

                    ocr_text = (
                        (ocr_text or "").rstrip()
                        + "\n\n"
                        + diagram_block
                    )

                    diagrams_found += 1

                # ==================================================
                # PRESERVE PAGE
                # ==================================================

                if ocr_text:

                    pages.append(
                        f"[Page {page_number} "
                        f"- scanned, OCR]\n"
                        f"{ocr_text}"
                    )

                    ocr_pages += 1

                else:

                    blank_pages += 1

        finally:

            doc.close()

        # ==========================================================
        # ASSEMBLE
        # ==========================================================

        text = (
            "\n\n"
            .join(pages)
            .strip()
        )

        if not text:

            raise ValueError(
                "No extractable or recognizable "
                "text was found in this PDF."
            )

        return (
            text,
            {
                "source_type": "pdf",
                "page_count": total_pages,
                "processed_pages": len(
                    pages
                ),
                "typed_pages": typed_pages,
                "ocr_pages": ocr_pages,
                "blank_pages": blank_pages,
                "math_extraction_used": (
                    equations_found > 0
                ),
                "equations_found": equations_found,
                "diagram_extraction_used": (
                    diagrams_found > 0
                ),
                "diagrams_found": diagrams_found,
            },
        )

    # ==========================================================
    # OCR DPI
    # ==========================================================

    @classmethod
    def _get_ocr_dpi(
        cls,
    ) -> int:

        value = os.environ.get(
            "OCR_RENDER_DPI"
        )

        if value:

            try:

                dpi = int(value)

                if (
                    100
                    <= dpi
                    <= 300
                ):

                    return dpi

            except ValueError:

                pass

        return (
            cls.DEFAULT_OCR_RENDER_DPI
        )

    # ==========================================================
    # CLEAN OCR OUTPUT
    # ==========================================================

    @staticmethod
    def _clean_ocr_output(
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

        text = "\n".join(
            line.rstrip()
            for line in text.split(
                "\n"
            )
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\s+([,.;:!?])",
            r"\1",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    # ==========================================================
    # PDF PAGE NUMBER NOISE
    # ==========================================================

    @staticmethod
    def _remove_repeated_page_noise(
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

        text = re.sub(
            r"\n\s*\d+\s*\n",
            "\n",
            text,
        )

        return text