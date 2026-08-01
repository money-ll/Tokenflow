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

        optimized = self.optimizer.optimize(
            raw_text,
            target_reduction=target_reduction,
        )

        context = optimized["text"]
        if query.strip():
            assembled = (
                "TASK: " + query.strip() + "\\n"
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
                == "handwritten_image",
                "ocr_fallback_used": source_meta.get("ocr_pages", 0) > 0,
            },
        }
