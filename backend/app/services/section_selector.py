"""
TokenFlow Section Selector

Lets a query like "only give me the summarized introduction" restrict
optimization to a single section of the document instead of the whole
thing.

This is a heading-detection heuristic, not a layout/ML model. It scans
the extracted text for numbered or ALL-CAPS section headings -- "1.
Introduction", "1: INTRODUCTION", "II. Background", plain "INTRODUCTION"
-- and slices the document between the matched heading and the next
heading. Matching is done against the raw text directly (not split into
lines first) because PDF text extraction frequently collapses a
cover-page/table-style layout so the heading ends up on the same line
as surrounding content; requiring headings to sit alone on their own
line missed real documents in practice.

Two-step process:

    1. detect_requested_section(query)
       Does the query actually ask to be restricted to one section?
       Requires a restrictive word ("only", "just", etc.) AND a known
       section name/alias -- "introduction" alone in a query doesn't
       trigger this (e.g. "explain the introduction more") since that
       isn't asking to throw away the rest of the document.

    2. extract_section(text, canonical_section)
       Given the extracted document text and a canonical section name,
       find that section's heading and return everything up to the
       next heading (recognized or not -- see _GENERIC_HEADING_RE) or
       end of document. Returns None if no matching heading is found,
       so the caller can fall back to the full document instead of
       silently returning nothing.
"""

from __future__ import annotations

import re

# Canonical section name -> the heading words/phrases that count as
# that section.
SECTION_ALIASES: dict[str, list[str]] = {
    "abstract": ["abstract", "executive summary"],
    "introduction": ["introduction", "background"],
    "problem statement": ["problem statement", "problem definition"],
    "objective": ["objectives", "objective", "aims and objectives", "goals"],
    "scope": ["scope", "project scope"],
    "literature review": ["literature review", "related work", "literature survey"],
    "requirements": ["requirements", "requirement analysis", "requirements analysis"],
    "methodology": ["methodology", "methods", "approach", "materials and methods"],
    "system architecture": ["system architecture", "architecture", "system design"],
    "design": ["design", "design and implementation"],
    "implementation": ["implementation"],
    "working principle": ["working principle", "working principles", "system workflow"],
    "tools and technologies": ["tools and technologies", "technologies used", "tools used", "technology stack"],
    "testing": ["testing", "test plan", "testing and validation"],
    "results": ["results", "findings"],
    "evaluation": ["evaluation", "performance evaluation"],
    "discussion": ["discussion"],
    "limitations": ["limitations", "limitation"],
    "future work": ["future work", "future scope", "future enhancements"],
    "recommendations": ["recommendations", "recommendation"],
    "risk assessment": ["risk assessment", "risk analysis", "risks"],
    "timeline": ["timeline", "project timeline", "schedule"],
    "budget": ["budget", "cost estimation", "cost analysis"],
    "feasibility study": ["feasibility study", "feasibility"],
    "case study": ["case study", "case studies"],
    "conclusion": ["conclusion", "conclusions", "closing remarks", "summary and conclusion"],
    "acknowledgements": ["acknowledgements", "acknowledgments"],
    "references": ["references", "bibliography", "works cited"],
    "appendix": ["appendix"],
    "team members": ["team members", "project team"],
}

_ALIAS_TO_CANONICAL: dict[str, str] = {
    alias: canonical
    for canonical, aliases in SECTION_ALIASES.items()
    for alias in aliases
}
# Longest alias first so "materials and methods" is checked before
# "methods" when scanning for a match.
_ALIASES_BY_LENGTH = sorted(_ALIAS_TO_CANONICAL, key=len, reverse=True)

_RESTRICTIVE_RE = re.compile(
    r"\b(only|just|solely|purely|exclusively)\b",
    re.IGNORECASE,
)

# Words that don't carry any meaning of their own when deciding whether
# a query is "basically just naming a section" (see detect_requested_
# section below) -- e.g. "summarize introduction" or "give me the
# results" should trigger section selection even with no "only"/"just".
_FILLER_WORDS = {
    "summary", "summarize", "summarise", "summarised", "summarized",
    "give", "me", "the", "a", "an", "of", "for", "please", "only",
    "just", "section", "part", "please", "get", "show", "provide",
    "need", "want", "looking", "this", "report", "document", "doc",
    "text", "content", "from",
}

# Generic "this looks like some numbered heading" detector: a number
# (optionally dotted, "1" / "1.2") followed immediately by ':', '.', or
# ')', then 1-6 capitalized words. Doesn't need to know the heading's
# NAME, just that a new heading has started -- this is what lets a
# custom section title we don't recognize (e.g. "2. OBJECTIVE") still
# correctly end a previous section instead of swallowing the rest of
# the document.
_GENERIC_HEADING_RE = re.compile(
    r"(?:^|(?<=\s))\d+(?:\.\d+)*\s*[:.\)]\s*([A-Z][A-Za-z\-/]*(?:\s+[A-Z][A-Za-z\-/]*){0,5})"
)


def _alias_positions(text: str, alias: str) -> list[tuple[int, int]]:
    """Find every place `alias` appears as a heading in `text`, as
    (start, end) character spans covering the heading including any
    leading numbering. Matches two forms:

      - numbered: "1: Introduction", "1. Introduction", "II) Background"
      - shouted:  a standalone ALL-CAPS run, e.g. "INTRODUCTION"

    Deliberately NOT matching plain lowercase/Title-Case mentions with
    no numbering and no caps, since those are usually just the word
    used in a sentence ("the introduction explains...") rather than an
    actual heading.
    """
    words_pattern = r"\s+".join(re.escape(w) for w in alias.split())
    spans = []

    numbered_re = re.compile(
        r"((?:^|(?<=\s))\d+(?:\.\d+)*\s*[:.\)]\s*)(" + words_pattern + r")\b",
        re.IGNORECASE,
    )
    for m in numbered_re.finditer(text):
        spans.append((m.start(1), m.end(2)))

    caps_re = re.compile(r"(?<![A-Za-z])(" + words_pattern.upper() + r")(?![A-Za-z])")
    for m in caps_re.finditer(text):
        spans.append((m.start(1), m.end(1)))

    return spans


class SectionSelector:
    def detect_requested_section(self, query: str) -> str | None:
        """Return the canonical section name the query is asking to be
        restricted to, or None if the query isn't making that kind of
        request at all.

        Two ways a query counts as a section request:
          1. It contains an explicit restrictive word ("only", "just",
             etc.) alongside a known section name.
          2. It's short enough that, once the section name and filler
             words ("summarize", "give me", "the", ...) are stripped
             out, nothing meaningful is left -- e.g. "summarize
             introduction" or "results please" are unambiguously
             asking for just that section, even with no "only"/"just".
        """
        if not query or not query.strip():
            return None

        normalized = query.lower().strip()
        restrictive = bool(_RESTRICTIVE_RE.search(normalized))

        for alias in _ALIASES_BY_LENGTH:
            pattern = rf"\b{re.escape(alias)}\b"
            if not re.search(pattern, normalized):
                continue

            if restrictive:
                return _ALIAS_TO_CANONICAL[alias]

            remainder = re.sub(pattern, " ", normalized)
            remaining_words = [
                w.strip(",.!?") for w in remainder.split()
            ]
            meaningful = [w for w in remaining_words if w and w not in _FILLER_WORDS]
            if not meaningful:
                return _ALIAS_TO_CANONICAL[alias]

        return None

    def extract_section(self, text: str, canonical: str) -> str | None:
        """Return just the requested section's text (heading through the
        next heading, recognized or not), or None if that section's
        heading can't be found in the document at all."""
        if not text:
            return None

        start = None
        for alias in SECTION_ALIASES.get(canonical, [canonical]):
            for span_start, _span_end in _alias_positions(text, alias):
                if start is None or span_start < start:
                    start = span_start

        if start is None:
            return None

        end = len(text)
        for match in _GENERIC_HEADING_RE.finditer(text, start + 1):
            end = match.start()
            break

        section_text = text[start:end].strip()
        return section_text or None


section_selector = SectionSelector()