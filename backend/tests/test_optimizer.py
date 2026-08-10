from app.services.optimizer import SemanticOptimizer

def test_negations_are_preserved():
    result = SemanticOptimizer().optimize(
        "The system must not remove important information. It should never delete negations."
    )
    text = result["text"].lower()
    assert "not" in text or "must not" in text
    assert "never" in text

def test_phrase_compaction():
    result = SemanticOptimizer().optimize(
        "In order to reduce tokens, the system should remove redundancy."
    )
    assert "in order to" not in result["text"].lower()
