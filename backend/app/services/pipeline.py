import re
import hashlib
from datetime import datetime, timezone

from app.services.extractor import InputExtractor
from app.services.optimizer import SemanticOptimizer
from app.services.tokenizer import TokenCounter
from app.services.section_selector import SectionSelector


class TokenFlowPipeline:
    # Pure letters-digits-letters, no symbols -- optimizer.py's
    # tokenizer (safe_stopword_reduction) always splits digits away
    # from letter runs and rejoins everything with single spaces, so
    # any placeholder containing punctuation/brackets gets corrupted.
    # This format always splits into exactly 3 tokens in the same
    # order -- PREFIX, digits, SUFFIX -- so the restore regex just
    # tolerates the inserted spaces. Shared by BOTH equations and
    # diagrams so both survive optimization the same way.
    _PLACEHOLDER_PREFIX = "TFPROTECTEDSTART"
    _PLACEHOLDER_SUFFIX = "TFPROTECTEDEND"
    _PLACEHOLDER_RE = re.compile(
        rf"{_PLACEHOLDER_PREFIX}\s*(\d+)\s*{_PLACEHOLDER_SUFFIX}"
    )

    # Matches a single "Equation: <latex>" line.
    _EQUATION_LINE_RE = re.compile(r"^Equation:\s*(.+)$")

    # Matches a full "[DIAGRAM_START] ... [DIAGRAM_END]" block,
    # including the fenced ```mermaid ... ``` content inside it.
    # DOTALL so '.' matches newlines -- diagram blocks are multi-line.
    _DIAGRAM_BLOCK_RE = re.compile(
        r"\[DIAGRAM_START\](.*?)\[DIAGRAM_END\]",
        re.DOTALL,
    )

    def __init__(self):
        self.extractor = InputExtractor()
        self.optimizer = SemanticOptimizer()
        self.counter = TokenCounter()
        self.section_selector = SectionSelector()

    def process(self, filename, content, query="", target_reduction=0.45):
        raw_text, source_meta = self.extractor.extract(filename, content)

        # If the query is asking to be restricted to one section (e.g.
        # "only give me the introduction, summarized"), slice the
        # document down to that section BEFORE compression, so both the
        # token counts and the optimized output reflect just that
        # section rather than the whole document.
        working_text = raw_text
        section_info = None
        requested_section = self.section_selector.detect_requested_section(query)
        if requested_section:
            section_text = self.section_selector.extract_section(raw_text, requested_section)
            if section_text:
                working_text = section_text
                section_info = {"requested": requested_section, "found": True}
            else:
                section_info = {"requested": requested_section, "found": False}

        raw_tokens = self.counter.count(working_text)

        text_for_compression = working_text
        protected_blocks = []

        if source_meta.get("math_extraction_used") or source_meta.get(
            "diagram_extraction_used"
        ):
            text_for_compression, protected_blocks = (
                self._inline_protected_blocks(working_text)
            )

        optimized = self.optimizer.optimize(
            text_for_compression,
            target_reduction=target_reduction,
        )

        context = optimized["text"]

        if protected_blocks:
            context, missing = self._restore_protected_blocks(
                context, protected_blocks
            )
            if missing:
                context = (
                    context.rstrip()
                    + "\n\n"
                    + "\n\n".join(missing)
                )

        if query.strip():
            assembled = (
                "TASK: " + query.strip() + "\n"
                "CONTEXT: " + context
            )
        else:
            assembled = context

        optimized_tokens = self.counter.count(assembled)
        reduction = self.counter.reduction_rate(raw_tokens, optimized_tokens)

        digest = hashlib.sha1(
            (filename + str(datetime.now(timezone.utc).timestamp())).encode()
        ).hexdigest()[:12]

        # Diagram blocks always start with "```mermaid"; equation
        # blocks always start with "$$". Cheap discriminator, no need
        # to carry two separate lists through the restore machinery.
        diagrams_count = sum(
            1 for b in protected_blocks if b.startswith("```mermaid")
        )
        equations_count = len(protected_blocks) - diagrams_count

        return {
            "id": digest,
            "filename": filename,
            "source": source_meta,
            "section": section_info,
            "optimized_text": context,
            "assembled_prompt": assembled,
            "metrics": {
                "original_tokens": raw_tokens,
                "optimized_tokens": optimized_tokens,
                "token_reduction_rate": reduction,
                "original_sentence_count": optimized["original_sentence_count"],
                "optimized_sentence_count": optimized["optimized_sentence_count"],
                "duplicate_sentences_removed": optimized["duplicate_sentences_removed"],
                "equations_preserved": equations_count,
                "diagrams_preserved": diagrams_count,
                "stage_tokens": {
                    name: self.counter.count(value)
                    for name, value in optimized["stages"].items()
                },
            },
            "strategy": {
                "negation_safe": True,
                "semantic_selection": True,
                "phrase_compaction": True,
                "near_duplicate_removal": True,
                "abstractive_model": False,
                "handwriting_recognition": source_meta.get("source_type")
                == "handwritten",
                "ocr_fallback_used": source_meta.get("ocr_pages", 0) > 0,
                "equation_preservation": equations_count > 0,
                "diagram_preservation": diagrams_count > 0,
            },
        }

    @classmethod
    def _extract_equations(cls, text):
        """Compatibility helper used by tests and external callers."""
        equations = []
        kept = []
        for line in text.splitlines():
            match = cls._EQUATION_LINE_RE.match(line.strip())
            if match:
                equations.append(f"Equation: {match.group(1).strip()}")
            elif line.strip():
                kept.append(line.strip())
        return "\n".join(kept), equations

    @classmethod
    def _inline_protected_blocks(cls, text):
        """
        Replace every "Equation: <latex>" line AND every
        "[DIAGRAM_START]...[DIAGRAM_END]" block with a single-line
        placeholder token IN PLACE, so optimization sees a marker at
        each block's original position instead of losing it, and so
        the optimizer's line/sentence-based logic never sees (and
        can't corrupt) raw LaTeX or Mermaid syntax.

        Diagram blocks are extracted FIRST -- since they span multiple
        lines, extracting them before the line-by-line equation pass
        prevents any internal diagram line from being misread as an
        "Equation:" line.

        Returns (text_with_placeholders, blocks) where blocks[i] is
        the exact, fully-formatted string placeholder i stands in for
        -- "$$...$$" for equations, "```mermaid...```" for diagrams.
        """
        blocks = []

        def _diagram_sub(match):
            mermaid_content = match.group(1).strip()
            formatted = f"```mermaid\n{mermaid_content}\n```"
            idx = len(blocks)
            blocks.append(formatted)
            return f"{cls._PLACEHOLDER_PREFIX}{idx}{cls._PLACEHOLDER_SUFFIX}"

        text = cls._DIAGRAM_BLOCK_RE.sub(_diagram_sub, text)

        out_lines = []
        for line in text.split("\n"):
            m = cls._EQUATION_LINE_RE.match(line.strip())
            if m:
                latex = m.group(1).strip()
                formatted = f"$${latex}$$"
                idx = len(blocks)
                blocks.append(formatted)
                out_lines.append(
                    f"{cls._PLACEHOLDER_PREFIX}{idx}{cls._PLACEHOLDER_SUFFIX}"
                )
            else:
                out_lines.append(line)

        return "\n".join(out_lines), blocks

    @classmethod
    def _restore_protected_blocks(cls, text, blocks):
        """
        Swap placeholders back for their original (formatted) block
        strings. Tolerates whitespace the optimizer's tokenizer may
        have inserted between the placeholder's letter/digit segments.

        Returns (restored_text, missing) -- blocks whose placeholder
        wasn't found at all (the optimizer dropped that sentence
        entirely) get appended by the caller instead of silently lost.
        """
        found = set()

        def _sub(match):
            idx = int(match.group(1))
            found.add(idx)
            return blocks[idx]

        restored = cls._PLACEHOLDER_RE.sub(_sub, text)
        missing = [b for i, b in enumerate(blocks) if i not in found]
        return restored, missing