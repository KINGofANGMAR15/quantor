"""Тесты интегралов и производных (quantor.calculus)."""

import math

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
    r"""DoD: \frac{d}{dx} \sin(x) -> cos(x)."""
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
    r"""DoD: \frac{d}{dx} \sin(x) \big|_{x=0} -> 1.0."""
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


# ------------------------------------- границы интегрирования: безопасность и ввод
def test_definite_pow_bomb_rejected(calculus):
    r"""Предел минует гард литеральных степеней парсера — гард обязателен.

    `\int_{0}^{10^15/3} 2^x dx` раньше строил 2^(10^15/3) внутри integrate:
    ~10 с и MemoryError (измерено в изоляции). Должен быть понятный отказ.
    """
    from quantor.engine import CalculatorError

    with pytest.raises(ValueError, match="Степень слишком велика"):
        calculus.evaluate_query(r"\int_{0}^{10^15/3} 2^x dx")
    with pytest.raises(CalculatorError, match="Степень"):
        calculus.definite_integral("2^x", "x", "0", "10^15/3")


def test_derivative_at_point_pow_bomb_rejected(calculus):
    """Подстановка точки в производную 2^x — тот же класс «бомбы»."""
    with pytest.raises(ValueError, match="Степень слишком велика"):
        calculus.derivative_at_point("2^x", "x", "10^15/3")


def test_derivative_at_point_factorial_bomb_rejected(calculus):
    """factorial(Integer) SymPy вычисляет жадно — гард и на факториалы."""
    with pytest.raises(ValueError, match="факториал"):
        calculus.derivative_at_point("x!", "x", "10^15")


def test_polynomial_growth_bound_rejected(calculus):
    """∫₀^b xⁿ при огромных b: та же политика цифр, что у парсера степеней."""
    from quantor.engine import CalculatorError

    with pytest.raises(CalculatorError, match="Степень слишком велика"):
        calculus.definite_integral("x^20000", "x", "0", "10^15")


def test_normal_large_bounds_still_work(calculus):
    """Большие, но безопасные пределы не пострадали от гарда."""
    data = calculus.definite_integral("x^2", "x", "0", "10^15")
    assert data["symbolic"] == sp.Rational(10**45, 3)
    data = calculus.definite_integral("2^x", "x", "0", "100")
    assert data["value"] == pytest.approx((2.0**100 - 1.0) / math.log(2))


def test_definite_bound_constant_name(calculus):
    r"""Предел «e»: движок возвращает константу python-числом — AttributeError.

    `\int_{1}^{e} 1/x dx` = ln(e) = 1 (регрессия: 'float' has no 'free_symbols').
    """
    data = calculus.definite_integral(r"\frac{1}{x}", "x", "1", "e")
    # e приходит python-числом (15 знаков), точной единицы не будет — и не надо
    assert float(data["symbolic"]) == pytest.approx(1.0)
    assert data["numeric"] == pytest.approx(1.0)


def test_definite_infinite_bound(calculus):
    r"""`oo` в пределе — бесконечность, а не неизвестный Symbol.

    Раньше выводился бессмысленный «точно: 1 - exp(-oo)».
    """
    data = calculus.evaluate_query(r"\int_{0}^{oo} e^{-x} dx")
    assert data.value == pytest.approx(1.0)
    assert data.lines[0] == "точно: 1"
    data = calculus.evaluate_query(r"\int_{0}^{\infty} e^{-x} dx")
    assert data.value == pytest.approx(1.0)


def test_definite_limits_without_braces(calculus):
    r"""\int_0^\pi — пределы одиночными атомами, как в тетради."""
    res = calculus.evaluate_query(r"\int_0^\pi \sin(x) dx")
    assert res.value == pytest.approx(2.0)


def test_definite_nested_brace_limit(calculus):
    r"""Пределы с вложенными скобками: \int_{\frac{\pi}{2}}^{1} 2x dx = 3/4."""
    res = calculus.evaluate_query(r"\int_{\frac{\pi}{2}}^{\pi} \sin(x) dx")
    assert res.value == pytest.approx(1.0)


def test_malformed_int_queries_return_none(calculus):
    r"""Не разобранные \int-запросы — None (раньше вели к падению GUI)."""
    assert calculus.evaluate_query(r"\int x") is None          # нет dx
    assert calculus.evaluate_query(r"\int_{0}^{1} x") is None  # нет dx
    assert calculus.evaluate_query(r"\int_0") is None          # нет предела/тела
