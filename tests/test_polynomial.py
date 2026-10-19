"""Тесты решения полиномов (критерии приёмки)."""

import cmath
import pytest

from quantor.engine import CalculatorEngine, CalculatorError


@pytest.fixture()
def engine() -> CalculatorEngine:
    return CalculatorEngine("DEG")


def test_cubic_acceptance(engine):
    """x^3 - 6x^2 + 11x - 6 = 0 -> [1, 2, 3]."""
    assert engine.solve_polynomial([1, -6, 11, -6]) == [1, 2, 3]


def test_quartic_acceptance(engine):
    """x^4 - 5x^2 + 4 = 0 -> [-2, -1, 1, 2]."""
    assert engine.solve_polynomial([1, 0, -5, 0, 4]) == [-2, -1, 1, 2]


def test_quadratic_complex_roots(engine):
    """x^2 + 1 = 0 -> корни ±i."""
    roots = engine.solve_polynomial([1, 0, 1])
    assert len(roots) == 2
    assert all(isinstance(r, complex) for r in roots)
    assert roots == sorted(roots, key=lambda z: (z.real, z.imag))
    assert cmath.isclose(roots[0], -1j)
    assert cmath.isclose(roots[1], 1j)


def test_quadratic_mixed_roots(engine):
    """x^2 - 1 = 0 -> [-1, 1]; вещественные идут первыми."""
    assert engine.solve_polynomial([1, 0, -1]) == [-1, 1]


def test_linear(engine):
    assert engine.solve_polynomial([2, -4]) == [2]


def test_from_expression_string(engine):
    assert engine.solve_polynomial("x^3 - 6x^2 + 11x - 6") == [1, 2, 3]


def test_bad_input(engine):
    with pytest.raises(CalculatorError):
        engine.solve_polynomial([1])  # один коэффициент — не полином
