import re
import math
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SAFE_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than",
    "of", "to", "in", "on", "at", "by", "for", "from", "with",
    "as", "into", "about", "over", "after", "before", "during",
    "through", "between", "within", "among", "per", "via",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its",
    "which", "who", "whom", "where", "when", "while",
    "also", "very", "just", "often", "usually", "generally",
    "can", "could", "would", "shall", "will",
}

PROTECTED = {
    "not", "no", "never", "neither", "nor", "without", "hardly",
    "barely", "cannot", "can't", "won't", "isn't", "aren't",
    "wasn't", "weren't", "don't", "doesn't", "didn't",
    "should", "must", "required", "require", "need", "needs",
    "may", "might", "except", "only", "unless", "despite",
}

class SemanticOptimizer:
    def __init__(self, similarity_threshold=0.86):
        self.similarity_threshold = similarity_threshold

    def optimize(self, text: str, target_reduction=0.45):
        normalized = self.normalize(text)
        compact = self.compact_phrases(normalized)
        reduced = self.safe_stopword_reduction(compact)

        sentences = self.split_sentences(reduced)
        deduped, duplicate_count = self.deduplicate(sentences)
        ranked = self.rank_sentences(deduped)

        budget_ratio = max(0.15, min(0.95, 1.0 - target_reduction))
        compressed = self.select_redundancy_aware(ranked, budget_ratio)
        final = self.restore_readability(" ".join(compressed))

        stages = {
            "raw": text,
            "normalized": normalized,
            "phrase_compaction": compact,
            "safe_word_reduction": reduced,
            "deduplicated": " ".join(deduped),
            "semantic_selection": " ".join(compressed),
            "final": final,
        }

        return {
            "text": final,
            "stages": stages,
            "duplicate_sentences_removed": duplicate_count,
            "original_sentence_count": len(self.split_sentences(normalized)),
            "optimized_sentence_count": len(self.split_sentences(final)),
        }

    def normalize(self, text):
        # Avoid a fragile regex character range entirely.
        text = text.replace("\u00a0", " ")
        text = "".join(
            " " if ord(ch) < 32 and ch not in "\n\t\r" else ch
            for ch in text
        )
        return re.sub(r"\s+", " ", text).strip()

    def compact_phrases(self, text):
        replacements = [
            (r"\bin order to\b", "to"),
            (r"\bdue to the fact that\b", "because"),
            (r"\bat this point in time\b", "now"),
            (r"\bin the event that\b", "if"),
            (r"\bhas the ability to\b", "can"),
            (r"\ba number of\b", "many"),
            (r"\bin close proximity to\b", "near"),
            (r"\bfor the purpose of\b", "for"),
            (r"\bwith regard to\b", "about"),
            (r"\bin spite of the fact that\b", "although"),
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def safe_stopword_reduction(self, text):
        # No fragile character-range regex. Tokenize with simple, safe patterns.
        tokens = re.findall(
            r"https?://\S+|\d+(?:\.\d+)?%?|[A-Za-z]+(?:'[A-Za-z]+)?|[^\w\s]",
            text,
        )
        out = []
        for token in tokens:
            low = token.lower()
            if low in PROTECTED or low not in SAFE_STOPWORDS:
                out.append(token)

        result = " ".join(out)
        result = re.sub(r"\s+([,.!?;:])", r"\1", result)
        result = re.sub(r'(["\'])\s+', r"\1", result)
        return result

    def split_sentences(self, text):
        return [
            p.strip()
            for p in re.split(r"(?<=[.!?])\s+", text.strip())
            if p.strip()
        ]

    def deduplicate(self, sentences):
        if len(sentences) <= 1:
            return sentences, 0

        try:
            matrix = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1
            ).fit_transform(sentences)
            sims = cosine_similarity(matrix)
        except Exception:
            return sentences, 0

        kept = []
        removed = 0
        for i in range(len(sentences)):
            if any(sims[i, j] >= self.similarity_threshold for j in kept):
                removed += 1
            else:
                kept.append(i)

        return [sentences[i] for i in kept], removed

    def rank_sentences(self, sentences):
        if not sentences:
            return []

        words = re.findall(r"[A-Za-z][A-Za-z0-9'-]+", " ".join(sentences).lower())
        word_freq = Counter(words)
        scored = []

        for idx, sentence in enumerate(sentences):
            sentence_words = re.findall(
                r"[A-Za-z][A-Za-z0-9'-]+",
                sentence.lower()
            )
            if not sentence_words:
                continue

            freq_score = sum(
                math.log1p(word_freq[w])
                for w in sentence_words
            ) / math.sqrt(len(sentence_words))

            numeric_bonus = 1.0 if re.search(
                r"\d|%|\$|€|£",
                sentence
            ) else 0.0

            entity_bonus = len(
                re.findall(r"\b[A-Z][a-z]{2,}\b", sentence)
            ) * 0.18

            protected_bonus = sum(
                0.45
                for word in PROTECTED
                if re.search(
                    r"\b" + re.escape(word) + r"\b",
                    sentence,
                    re.IGNORECASE,
                )
            )

            length_penalty = 0.003 * max(
                0,
                len(sentence_words) - 35
            )

            scored.append((
                freq_score
                + numeric_bonus
                + entity_bonus
                + protected_bonus
                - length_penalty,
                idx,
                sentence,
            ))

        return sorted(scored, reverse=True)

    def select_redundancy_aware(self, ranked, budget_ratio):
        if not ranked:
            return []

        original_words = sum(
            len(item[2].split())
            for item in ranked
        )
        target_words = max(
            30,
            int(original_words * budget_ratio)
        )

        selected = []
        selected_words = 0

        for score, idx, sentence in ranked:
            words = len(sentence.split())

            if selected_words + words <= target_words or not selected:
                selected.append((idx, sentence))
                selected_words += words

            if selected_words >= target_words:
                break

        return [
            sentence
            for _, sentence in sorted(selected)
        ]

    def restore_readability(self, text):
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        return re.sub(r"\s{2,}", " ", text).strip()
