import re
import hashlib
from datetime import datetime, timezone

from app.services.extractor import InputExtractor
from app.services.optimizer import SemanticOptimizer
from app.services.tokenizer import TokenCounter


class TokenFlowPipeline:
    # Pure letters-digits-letters, no symbols. optimizer.py's tokenizer
    # (safe_stopword_reduction) always splits digits away from letter
    # runs and rejoins everything with single spaces, so any placeholder
    # containing punctuation/brackets gets corrupted. This format always
    # splits into exactly 3 tokens in the same order — PREFIX, digits,
    # SUFFIX — so the restore regex just tolerates the inserted spaces.
    _PLACEHOLDER_PREFIX = "TFMATHEQSTART"
    _PLACEHOLDER_SUFFIX = "TFMATHEQEND"
    _PLACEHOLDER_RE = re.compile(
        rf"{_PLACEHOLDER_PREFIX}\s*(\d+)\s*{_PLACEHOLDER_SUFFIX}"
    )

    def __init__(self):
        self.extractor = InputExtractor()
        self.optimizer = SemanticOptimizer()
        self.counter = TokenCounter()

    def process(self, filename, content, query="", target_reduction=0.45):
        raw_text, source_meta = self.extractor.extract(filename, content)
        raw_tokens = self.counter.count(raw_text)

        text_for_compression = raw_text
        equations = []

        if source_meta.get("math_extraction_used"):
            text_for_compression, equations = (
                self._inline_equations(raw_text)
            )

        optimized = self.optimizer.optimize(
            text_for_compression,
            target_reduction=target_reduction,
        )

        context = optimized["text"]

        if equations:
            context, missing = self._restore_equations(context, equations)
            if missing:
                context = (
                    context.rstrip()
                    + "\n\n"
                    + "\n".join(missing)
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

        return {
            "id": digest,
            "filename": filename,
            "source": source_meta,
            "optimized_text": context,
            "assembled_prompt": assembled,
            "metrics": {
                "original_tokens": raw_tokens,
                "optimized_tokens": optimized_tokens,
                "token_reduction_rate": reduction,
                "original_sentence_count": optimized["original_sentence_count"],
                "optimized_sentence_count": optimized["optimized_sentence_count"],
                "duplicate_sentences_removed": optimized["duplicate_sentences_removed"],
                "equations_preserved": len(equations),
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
                "equation_preservation": bool(equations),
            },
        }

    @classmethod
    def _inline_equations(cls, text):
        """
        Replace each "Equation: <latex>" line with a placeholder token
        IN PLACE, so optimization sees a marker at the equation's
        original position. The raw LaTeX is reformatted as display
        math ($$...$$) here.

        Returns (text_with_placeholders, equations) where equations[i]
        is the fully-formatted string that placeholder i stands in for.
        """
        equations = []
        out_lines = []

        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("Equation:"):
                latex = stripped[len("Equation:"):].strip()
                formatted = f"$${latex}$$"

                idx = len(equations)
                equations.append(formatted)
                out_lines.append(
                    f"{cls._PLACEHOLDER_PREFIX}{idx}{cls._PLACEHOLDER_SUFFIX}"
                )
            else:
                out_lines.append(line)

        return "\n".join(out_lines), equations

    @classmethod
    def _restore_equations(cls, text, equations):
        """
        Swap placeholders back for their original (formatted) equation
        strings. Tolerates whitespace the optimizer's tokenizer may
        have inserted between the placeholder's letter/digit segments.

        Returns (restored_text, missing) — equations whose placeholder
        wasn't found at all (the optimizer dropped that sentence
        entirely) get appended by the caller instead of silently lost.
        """
        found = set()

        def _sub(match):
            idx = int(match.group(1))
            found.add(idx)
            return equations[idx]

        restored = cls._PLACEHOLDER_RE.sub(_sub, text)
        missing = [eq for i, eq in enumerate(equations) if i not in found]
        return restored, missing