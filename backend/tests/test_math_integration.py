from app.services.extractor import InputExtractor
from app.services.pipeline import TokenFlowPipeline


def test_math_heuristic_ignores_ordinary_prose():
    assert not InputExtractor._looks_like_math_text(
        "This is ordinary prose with no equation."
    )


def test_math_heuristic_detects_common_equation():
    assert InputExtractor._looks_like_math_text(
        "E = mc^2"
    )


def test_math_heuristic_avoids_simple_arithmetic_in_prose():
    assert not InputExtractor._looks_like_math_text(
        "The system has 2 + 2 items."
    )


def test_equations_are_removed_from_optimizer_input():
    text, equations = TokenFlowPipeline._extract_equations(
        "Important context.\n\nEquation: x^2 + y^2 = z^2"
    )

    assert "Equation:" not in text
    assert equations == ["Equation: x^2 + y^2 = z^2"]


def test_equation_line_is_preserved_exactly():
    text, equations = TokenFlowPipeline._extract_equations(
        "A sentence.\nEquation: \\frac{x^2}{y}"
    )

    assert text == "A sentence."
    assert equations == ["Equation: \\frac{x^2}{y}"]
