import hashlib
from datetime import datetime, timezone

from app.services.extractor import InputExtractor
from app.services.optimizer import SemanticOptimizer
from app.services.tokenizer import TokenCounter


class TokenFlowPipeline:
    def __init__(self):
        self.extractor = InputExtractor()
        self.optimizer = SemanticOptimizer()
        self.counter = TokenCounter()

    def process(self, filename, content, query="", target_reduction=0.45):
        raw_text, source_meta = self.extractor.extract(filename, content)
        raw_tokens = self.counter.count(raw_text)

        # Math is protected only when the extractor actually found
        # equations.  For every ordinary document this follows the original
        # optimization path without additional processing.
        text_for_compression = raw_text
        equations = []

        if source_meta.get("math_extraction_used"):
            text_for_compression, equations = (
                self._extract_equations(raw_text)
            )

        optimized = self.optimizer.optimize(
            text_for_compression,
            target_reduction=target_reduction,
        )

        context = optimized["text"]

        # Reattach the original LaTeX exactly as produced by MathExtractor.
        # The optimizer never sees or modifies these equation lines.
        if equations:
            context = (
                context.rstrip()
                + "\n\n"
                + "\n".join(equations)
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

    @staticmethod
    def _extract_equations(text):
        """
        Pull every "Equation: <latex>" line out of the text so the
        compression pipeline never sees or touches it.  The lines are
        returned unchanged and restored after optimization.
        """
        equation_lines = []
        remaining_lines = []

        for line in text.split("\n"):
            if line.strip().startswith("Equation:"):
                equation_lines.append(line.strip())
            else:
                remaining_lines.append(line)

        return "\n".join(remaining_lines), equation_lines
