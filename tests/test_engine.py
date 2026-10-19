"""Тесты базовых операций, тригонометрии, логарифмов, памяти."""

import pytest

from quantor.engine import CalculatorEngine, CalculatorError


@pytest.fixture()
def engine() -> CalculatorEngine:
    return CalculatorEngine("DEG")


def test_basic_arithmetic(engine):
    assert engine.calculate("2+2*2")[1] == 6
    assert engine.calculate("(2+2)*2")[1] == 8
    assert engine.calculate("2^10")[1] == 1024
    assert engine.calculate("10/4")[1] == 2.5


def test_sqrt_factorial_abs_mod(engine):
    assert engine.calculate("sqrt(16)")[1] == 4
    assert engine.calculate("5!")[1] == 120
    assert engine.calculate("abs(-7)")[1] == 7
    assert engine.calculate("mod(7,3)")[1] == 1


def test_trig_degrees(engine):
    assert engine.calculate("sin(30)")[1] == pytest.approx(0.5)
    assert engine.calculate("cos(60)")[1] == pytest.approx(0.5)
    assert engine.calculate("tan(45)")[1] == pytest.approx(1.0)


def test_trig_inverse_degrees(engine):
    assert engine.calculate("asin(0.5)")[1] == pytest.approx(30.0)
    assert engine.calculate("acos(0.5)")[1] == pytest.approx(60.0)
    assert engine.calculate("atan(1)")[1] == pytest.approx(45.0)


def test_trig_radians():
    rad = CalculatorEngine("RAD")
    assert rad.calculate("sin(pi/2)")[1] == pytest.approx(1.0)
    assert rad.calculate("cos(pi)")[1] == pytest.approx(-1.0)
    assert rad.calculate("atan(1)")[1] == pytest.approx(0.7853981634, rel=1e-9)


def test_hyperbolic(engine):
    assert engine.calculate("sinh(0)")[1] == 0
    assert engine.calculate("cosh(0)")[1] == 1


def test_logs(engine):
    assert engine.calculate("ln(e)")[1] == pytest.approx(1.0)
    assert engine.calculate("log10(100)")[1] == pytest.approx(2.0)
    assert engine.calculate("log2(8)")[1] == pytest.approx(3.0)
    assert engine.calculate("log(8,2)")[1] == pytest.approx(3.0)  # произвольное основание


def test_constants(engine):
    assert engine.calculate("C")[1] == 299792458.0
    assert engine.calculate("h")[1] == pytest.approx(6.62607015e-34)
    assert engine.calculate("G")[1] == pytest.approx(6.67430e-11)
    assert engine.calculate("N_A")[1] == pytest.approx(6.02214076e23)
    assert engine.calculate("k")[1] == pytest.approx(1.380649e-23)


def test_ans_and_implicit(engine):
    engine.calculate("2+3")
    assert engine.calculate("Ans+1")[1] == 6
    assert engine.calculate("2pi")[1] == pytest.approx(6.283185307, rel=1e-9)


def test_memory(engine):
    engine.mem_add(engine.calculate("5")[1])
    engine.mem_add(engine.calculate("3")[1])
    assert engine.mem_recall() == 8
    engine.mem_sub(engine.calculate("4")[1])
    assert engine.mem_recall() == 4
    engine.mem_clear()
    assert engine.mem_recall() == 0


def test_errors(engine):
    with pytest.raises(CalculatorError):
        engine.calculate("5/0")
    with pytest.raises(CalculatorError):
        engine.calculate("sin(")
    with pytest.raises(CalculatorError):
        engine.calculate("")
    with pytest.raises(CalculatorError):
        engine.calculate("x+1")  # неизвестная переменная в калькуляторе


def test_formatting(engine):
    assert engine.calculate("sin(30)")[0] == "0.5"
    assert engine.calculate("sqrt(2)")[0] == "1.414213562"
    assert engine.calculate("1/3")[0] == "0.3333333333"
