"""Тесты LaTeX-ввода и рендеринга (quantor.latex_input)."""

import pytest

import sympy as sp

from quantor.latex_input import (
    LatexParseError, is_latex, mathtext_safe, parse_input, to_latex,
)


def test_is_latex_detection():
    assert is_latex(r"\frac{\sin(x)}{2}")
    assert is_latex(r"\int_{0}^{1} x dx")
    assert is_latex(r"\pi")
    assert not is_latex("sin(x)/2")
    assert not is_latex("x^2 + 1")


def test_plain_expression():
    assert parse_input("sin(x)/2") == sp.sin(sp.Symbol("x")) / 2
    assert parse_input("2^10") == 1024


def test_latex_fractions_exact():
    """DoD: 1/2 + 1/3 -> 5/6, а не 0.8333."""
    expr = parse_input(r"\frac{1}{2} + \frac{1}{3}")
    assert expr == sp.Rational(5, 6)
    assert to_latex(expr) == r"\frac{5}{6}"


def test_latex_mixed_plain():
    assert parse_input(r"\frac{\sin(x)}{2}") == sp.sin(sp.Symbol("x")) / 2
    assert parse_input(r"x^2 + \frac{1}{2}") == sp.Symbol("x")**2 + sp.Rational(1, 2)


def test_latex_pi_is_constant():
    """parse_latex даёт pi как символ — ядро обязано подменить на sp.pi."""
    expr = parse_input(r"\sin(\pi)")
    assert expr == 0


def test_bad_latex_friendly_error():
    """DoD: \frac{1} -> понятная ошибка, не падение."""
    with pytest.raises(LatexParseError):
        parse_input(r"\frac{1}")
    with pytest.raises(LatexParseError):
        parse_input(r"\frac{d}{dx")  # незакрытая скобка
    with pytest.raises(LatexParseError):
        parse_input("")


def test_mathtext_safe_strips_extras():
    assert mathtext_safe(r"\int\limits_{0}^{1} x dx") == r"\int_{0}^{1} x dx"
    assert "|" in mathtext_safe(r"\big|_{x=0}")


def test_render_latex_pixmap(qapp):
    """Рендер возвращает ненулевой QPixmap (нужен QGuiApplication)."""
    from quantor.latex_input import render_latex
    pixmap = render_latex(r"\frac{\sin(x)}{2}")
    assert pixmap is not None and not pixmap.isNull()
    assert pixmap.width() > 10
    # повторный вызов — из кэша, тот же объект
    assert render_latex(r"\frac{\sin(x)}{2}") is pixmap


def test_render_latex_bad_formula_returns_none(qapp):
    from quantor.latex_input import render_latex
    assert render_latex(r"\frac{1}{") is None  # не падает, возвращает None
