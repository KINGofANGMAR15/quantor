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


# ------------------------------ log_2: основание подчёркиванием (обычный текст)
def test_log_subscript_plain_text(engine):
    """log_2 8 == 3, log_10 1000 == 3, log_25 625 == 2 — запись как в учебниках."""
    assert engine.calculate("log_2 8")[1] == pytest.approx(3.0)
    assert engine.calculate("log_10 1000")[1] == pytest.approx(3.0)
    assert engine.calculate("log_25 625")[1] == pytest.approx(2.0)
    assert engine.calculate("log_2(8)")[1] == pytest.approx(3.0)
    assert engine.calculate("2*log_2 8")[1] == pytest.approx(6.0)
    assert engine.calculate("log_2 8 - log_3 27")[1] == pytest.approx(0.0)
    assert engine.calculate("log_2 2^10")[1] == pytest.approx(10.0)  # показатель входит в аргумент


def test_log_subscript_missing_arg(engine):
    with pytest.raises(CalculatorError, match="log_2"):
        engine.calculate("log_2")
    with pytest.raises(CalculatorError, match="log_7"):
        engine.calculate("log_7+")


def test_log_subscript_equation(engine):
    assert engine.solve_equation("log_2 x = 10") == [1024]
    assert engine.solve_equation("log_2(2x-1) = 5") == [16.5]
    assert engine.solve_equation("2*log_2 8 + x = 10") == [4]


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
    # Точный «табличный» вид (как S⇔D у Casio): дроби и радикалы
    assert engine.calculate("sin(30)")[0] == "1/2"
    assert engine.calculate("sqrt(2)")[0] == "√2"
    assert engine.calculate("1/3")[0] == "1/3"


def test_sqrt_binds_to_next_primary(engine):
    """√ связывается со следующим первичным выражением, а не со всем хвостом.

    Регрессия: '√2+1' считался как sqrt(2+1) = √3.
    """
    import math

    cases = {
        "√2 + 1/2": math.sqrt(2) + 0.5,
        "√2+1": math.sqrt(2) + 1,
        "√2×√8": 4.0,
        "√(2+2)": 2.0,
        "√2^2": 2.0,
        "√pi": math.sqrt(math.pi),
        "√sin(30)": math.sqrt(0.5),
    }
    for text, want in cases.items():
        shown, value = engine.calculate(text)
        assert abs(value - want) < 1e-9, f"{text}: {value} != {want}"


def test_table_trig_values(engine):
    """Табличные тригонометрические значения — обыкновенными дробями/радикалами."""
    table = {
        "sin(30)": "1/2", "cos(30)": "√3/2", "tan(30)": "√3/3",
        "sin(45)": "√2/2", "cos(45)": "√2/2", "tan(45)": "1",
        "sin(60)": "√3/2", "cos(60)": "1/2", "tan(60)": "√3",
        "sin(90)": "1", "cos(0)": "1", "tan(0)": "0",
        "sin(15)": "(√6 - √2)/4", "cos(15)": "(√6 + √2)/4",
        "tan(15)": "2 - √3", "cos(36)": "(√5 + 1)/4",
        "sin(120)": "√3/2", "cos(135)": "-√2/2", "tan(135)": "-1",
        "sin(150)": "1/2", "cos(150)": "-√3/2", "tan(150)": "-√3/3",
        "sin(210)": "-1/2", "cos(225)": "-√2/2", "tan(225)": "1",
        "sin(315)": "-√2/2", "cos(330)": "√3/2", "tan(330)": "-√3/3",
        "asin(0.5)": "30", "acos(0.5)": "60", "atan(1)": "45",
    }
    for expr, expected in table.items():
        assert engine.calculate(expr)[0] == expected, expr
    # В RAD-режиме те же значения через π
    rad = CalculatorEngine("RAD")
    assert rad.calculate("sin(pi/6)")[0] == "1/2"
    assert rad.calculate("tan(pi/4)")[0] == "1"
    assert rad.calculate("sin(pi/12)")[0] == "(√6 - √2)/4"
    assert rad.calculate("tan(pi/8)")[0] == "√2 - 1"
    # Десятичный ввод остаётся десятичным, нетабличные углы — десятичные
    assert engine.calculate("0.5")[0] == "0.5"
    assert engine.calculate("sin(0.5)")[0] == "0.008726535498"
    # Числовое значение (Ans) не меняется — точный вид только на дисплее
    _, value = engine.calculate("cos(30)")
    assert abs(value - 0.8660254037844386) < 1e-12


def test_exact_display_misc(engine):
    assert engine.calculate("10/4")[0] == "5/2"
    assert engine.calculate("sqrt(8)")[0] == "2√2"
    assert engine.calculate("2pi")[0] == "2π"
    assert engine.calculate("pi/6")[0] == "π/6"
    assert engine.calculate("2^10")[0] == "1024"
    assert engine.calculate("0.5+0.25")[0] == "0.75"
    assert engine.calculate("0.1+0.2")[0] == "0.3"
    assert engine.calculate("sqrt(-4)")[0] == "2i"
    assert engine.calculate("123456/7")[0] == "123456/7"
