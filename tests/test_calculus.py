"""Тесты интегралов и производных (quantor.calculus)."""

import pytest

import sympy as sp

from quantor.calculus import CalculusEngine, is_calculus_query
from quantor.engine import CalculatorEngine
from quantor.latex_input import LatexParseError, parse_input


@pytest.fixture()
def calculus() -> CalculusEngine:
    return CalculusEngine(CalculatorEngine("DEG"))


# ---------------------------------------------------------------- интегралы
def test_indefinite_polynomial(calculus):
    """DoD: \int x^2 dx -> x^3/3 + C."""
    result, latex = calculus.indefinite_integral("x^2", "x")
    x = sp.Symbol("x")
    assert sp.simplify(sp.diff(result, x) - x**2) == 0
    assert latex == r"\frac{x^{3}}{3} + C"


def test_indefinite_trig(calculus):
    result, latex = calculus.indefinite_integral(r"\sin(x)", "x")
    assert sp.simplify(result - (-sp.cos(sp.Symbol("x")) + sp.Symbol("C"))) == 0
    assert r"\cos" in latex


def test_indefinite_rational(calculus):
    result, _ = calculus.indefinite_integral(r"\frac{1}{x}", "x")
    assert result.has(sp.log)


def test_indefinite_failure_is_none(calculus):
    """Неэлементарная первообразная exp(sin(x)) -> (None, ''), без исключения."""
    result, latex = calculus.indefinite_integral(r"e^{\sin(x)}", "x")
    assert result is None and latex == ""


def test_definite_acceptance(calculus):
    """DoD: \int_{0}^{\pi} \sin(x) dx -> точно 2 и численно 2.0000000000."""
    data = calculus.definite_integral(r"\sin(x)", "x", "0", r"\pi")
    assert data["symbolic"] == 2
    assert data["numeric"] == pytest.approx(2.0)
    assert data["lines"] == ["точно: 2", "численно: 2.0000000000"]
    assert data["value"] == 2.0


def test_definite_numeric_only(calculus):
    """Символьно не берётся — остаётся только mpmath.quad."""
    data = calculus.definite_integral(r"e^{\sin(x)}", "x", "0", "1")
    assert data["symbolic"] is None
    assert data["numeric"] == pytest.approx(1.6318696084, abs=1e-8)
    assert "символьное решение не найдено" in data["lines"][0]
    assert data["lines"][1].startswith("численно: ")


def test_definite_rational_function(calculus):
    data = calculus.definite_integral(r"\frac{1}{1+x^2}", "x", "0", "1")
    assert data["symbolic"] == sp.pi / 4
    assert data["numeric"] == pytest.approx(float(sp.pi / 4), abs=1e-9)


def test_query_definite_from_string(calculus):
    res = calculus.evaluate_query(r"\int_{0}^{\pi} \sin(x) dx")
    assert res.value == 2.0
    assert res.lines == ["точно: 2", "численно: 2.0000000000"]


def test_query_indefinite_from_string(calculus):
    res = calculus.evaluate_query(r"\int x^2 dx")
    assert res.latex == r"\int x^{2}\,dx = \frac{x^{3}}{3} + C"


def test_query_plain_integrate(calculus):
    res = calculus.evaluate_query("integrate(x^2, x)")
    assert res.lines == ["первообразная: C + x^3/3"]


# ---------------------------------------------------------------- производные
def test_derivative_first(calculus):
    """DoD: \frac{d}{dx} \sin(x) -> cos(x)."""
    result, latex = calculus.derivative(r"\sin(x)", "x")
    assert result == sp.cos(sp.Symbol("x"))
    assert r"\cos" in latex


def test_query_derivative(calculus):
    res = calculus.evaluate_query(r"\frac{d}{dx} \sin(x)")
    assert res.lines == ["cos(x)"]


def test_derivative_second_order(calculus):
    """DoD: \frac{d^2}{dx^2} x^4 -> 12x^2."""
    result, latex = calculus.derivative("x^4", "x", 2)
    assert result == 12 * sp.Symbol("x")**2
    assert latex == r"12 x^{2}"


def test_derivative_third_order(calculus):
    result, _ = calculus.derivative(r"\sin(x)", "x", 3)
    assert sp.simplify(result + sp.cos(sp.Symbol("x"))) == 0


def test_query_derivative_high_order(calculus):
    res = calculus.evaluate_query(r"\frac{d^3}{dx^3} x^4")
    assert res.lines == ["24*x"]


def test_derivative_at_point(calculus):
    """DoD: \frac{d}{dx} \sin(x) \big|_{x=0} -> 1.0."""
    assert calculus.derivative_at_point(r"\sin(x)", "x", 0) == 1.0


def test_query_derivative_at_point(calculus):
    res = calculus.evaluate_query(r"\frac{d}{dx} \sin(x) \big|_{x=0}")
    assert res.value == 1.0
    assert res.lines == ["значение в точке x=0: 1.0"]


def test_derivative_at_point_order_mismatch(calculus):
    with pytest.raises(LatexParseError):
        calculus.evaluate_query(r"\frac{d^2}{dx} x^4")


def test_query_var_mismatch(calculus):
    with pytest.raises(LatexParseError):
        calculus.evaluate_query(r"\frac{d}{dx} \sin(x) \big|_{t=0}")


# ---------------------------------------------------------------- диспетчер
def test_is_calculus_query():
    assert is_calculus_query(r"\int x dx")
    assert is_calculus_query(r"\frac{d}{dx} x")
    assert is_calculus_query("integrate(x^2, x)")
    assert is_calculus_query("diff(sin(x), x)")
    assert not is_calculus_query("sin(x)")
    assert not is_calculus_query(r"\frac{1}{2}")
    assert not is_calculus_query("")


def test_non_query_returns_none(calculus):
    assert calculus.evaluate_query("sin(x)") is None


def test_bad_latex_in_query(calculus):
    with pytest.raises(LatexParseError):
        calculus.evaluate_query(r"\int_{0}^{1} \frac{1}{ dx")


def test_result_uses_engine_constants(calculus):
    """Тело запроса может использовать константы ядра (C — скорость света)."""
    res = calculus.evaluate_query(r"\int_{0}^{1} C x dx")
    assert res.value == pytest.approx(299792458.0 / 2)
