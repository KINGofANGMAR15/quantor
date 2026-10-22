"""Тесты решения общих уравнений."""

import pytest

from quantor.engine import CalculatorEngine, CalculatorError


@pytest.fixture()
def engine() -> CalculatorEngine:
    return CalculatorEngine("DEG")


def test_transcendental_power(engine):
    """2^x = 8 -> x = 3."""
    assert engine.solve_equation("2^x=8") == [3]


def test_linear(engine):
    assert engine.solve_equation("3x+2=8") == [2]


def test_with_fractions(engine):
    assert engine.solve_equation("x/2+1/4=0") == [-0.5]


def test_quadratic_via_equals(engine):
    assert engine.solve_equation("x^2-4=0") == [-2, 2]


def test_trig_equation():
    rad = CalculatorEngine("RAD")
    roots = rad.solve_equation("sin(x)=0.5")
    assert len(roots) >= 1
    assert any(abs(r - 0.5235987756) < 1e-6 for r in roots)


def test_rhs_omitted(engine):
    """Без правой части считаем RHS = 0."""
    assert engine.solve_equation("x^2-9") == [-3, 3]


def test_no_solutions(engine):
    with pytest.raises(CalculatorError):
        engine.solve_equation("x+1=x+2")


def test_infinite_solutions(engine):
    with pytest.raises(CalculatorError):
        engine.solve_equation("2x=2x")


def test_bad_syntax(engine):
    with pytest.raises(CalculatorError):
        engine.solve_equation("x^2=")


# ------------------------------------------- adversarial 2026-10: предел степени
def test_dense_high_degree_equation_rejected(engine):
    """sp.solve на плотном полиноме 10-й степени = 12.3 с, 15+ — зависание."""
    with pytest.raises(CalculatorError, match="Степень уравнения \\(9\\)"):
        engine.solve_equation("x^9+x^8+x^2+1=0")
    with pytest.raises(CalculatorError, match="выше поддерживаемой"):
        engine.solve_equation("+".join(f"x^{k}" for k in range(99, 7, -1)) + "=0")


def test_quintic_and_quartic_still_solved(engine):
    """Лимит 8: обычные степени решаются как прежде."""
    assert engine.solve_equation("x^4-1=0") == [-1, 1, -1j, 1j]
    assert engine.solve_equation("x^3-8=0")[0] == 2
    assert engine.solve_equation("2^x=8") == [3]        # трансцендентное — вне лимита


def test_high_degree_polynomial_rejected(engine):
    with pytest.raises(CalculatorError, match="выше поддерживаемой"):
        engine.solve_polynomial("x^99+x^98+x^2+1")
    with pytest.raises(CalculatorError, match="выше поддерживаемой"):
        engine.solve_polynomial([1] * 10)               # 9-я степень списком
    assert engine.solve_polynomial([1, -6, 11, -6]) == [1, 2, 3]
