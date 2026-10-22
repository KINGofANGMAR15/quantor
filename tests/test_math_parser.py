"""Тесты безопасного математического парсера (quantor.math_parser).

* позитивные: грамматика покрывает прежние возможности движка;
* негативные: запрещённые конструкции отвергаются ДО вычисления —
  вредоносные строки не исполняются, побочных эффектов нет;
* ресурсные: лимиты длины, сложности, вложенности, степеней, факториала.
"""

import builtins

import pytest
import sympy as sp

from quantor.engine import CalculatorEngine, CalculatorError, _preprocess
from quantor import math_parser
from quantor.math_parser import (
    MAX_DEPTH, MAX_INPUT_LEN, MAX_NODES, MathParseError, parse,
)


@pytest.fixture()
def ns():
    """Минимальный «белый список» движка для прямых тестов парсера."""
    x = sp.Symbol("x")
    return {
        "x": x,
        "sqrt": sp.sqrt, "abs": sp.Abs, "exp": sp.exp,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "ln": sp.log, "log": lambda a, b=None: sp.log(a) if b is None
        else sp.log(a) / sp.log(b),
        "mod": lambda a, b: sp.Mod(a, b),
    }


def p(text, ns_=None):
    """Разобрать текст с предобработкой кнопок (как это делает движок)."""
    return parse(_preprocess(text), ns if ns_ is None else ns_)


# ------------------------------------------------------------------- позитив
def test_arithmetic_precedence(ns):
    assert p("2+2*2", ns) == 6
    assert p("(2+2)*2", ns) == 8
    assert p("10-4-3", ns) == 3              # левая ассоциативность
    assert p("10/4", ns) == sp.Rational(5, 2)
    assert p("1/3", ns) == sp.Rational(1, 3)  # точная дробь, не 0.333
    assert p("2**10", ns) == 1024


def test_caret_is_power(ns):
    assert p("2^10", ns) == 1024
    assert p("x^2", ns) == sp.Symbol("x")**2


def test_right_associative_power_and_unary(ns):
    assert p("2^3^2", ns) == 512
    assert p("-2^2", ns) == -4               # -(2^2), как в Python
    assert p("2^-2", ns) == sp.Rational(1, 4)
    assert p("--2", ns) == 2
    assert p("+3", ns) == 3


def test_implicit_multiplication(ns):
    x = sp.Symbol("x")
    assert p("2x", ns) == 2 * x
    assert p("3sin(x)", ns) == 3 * sp.sin(x)
    assert p("2(3+4)", ns) == 14
    assert sp.expand(p("(x-1)(x+2)", ns)) == x**2 + x - 2
    assert p("x x", ns) == x**2
    assert p("2 x sin(x)", ns) == 2 * x * sp.sin(x)


def test_factorial(ns):
    assert p("5!", ns) == 120
    assert p("5!!", ns) == 15                # двойной факториал
    assert p("(2+3)!", ns) == 120


def test_numbers(ns):
    assert float(p("6.02e23", ns)) == pytest.approx(6.02e23)
    assert float(p("1e-9", ns)) == pytest.approx(1e-9)
    assert p("1_000", ns) == 1000
    assert float(p(".5+.5", ns)) == 1.0    # (как у старого парсера: Float, не Integer)


def test_percent_remainder(ns):
    assert p("7%3", ns) == 1
    assert p("-7%3", ns) == 2


def test_functions(ns):
    assert p("sqrt(16)", ns) == 4
    assert p("abs(-7)", ns) == 7
    assert sp.simplify(p("log(8,2)", ns)) == 3    # два аргумента
    assert p("mod(7,3)", ns) == 1
    assert p("exp(0)", ns) == 1


def test_pi_builtin_and_unknown_symbol(ns):
    assert p("2pi", ns) == 2 * sp.pi
    assert p("pi/6", ns) == sp.pi / 6
    assert sp.Symbol("q") in p("q+1", ns).free_symbols


def test_engine_ns_functions_are_called(ns):
    """Имена функций без скобки — ошибка, со скобкой — вызов."""
    with pytest.raises(MathParseError):
        p("sin*2", ns)
    with pytest.raises(MathParseError):
        p("sqrt(x))", ns)


# ------------------------------------------------------------------ негатив
@pytest.mark.parametrize("text", [
    "__import__('os')",
    "import os",
    "from os import path",
    "x.__class__",
    "x.__class__.__base__",
    "().__class__",
    "sqrt.__globals__",
    "open('README.md')",
    "eval('1')",
    "exec('1')",
    "compile('1','','')",
    "lambda x: x",
    "1 if 2 else 3",
    "[1,2]",
    "{1,2}",
    "(1,2)",
    "x[0]",
    "[1][0]",
    "'a'*10**9",
    '"a"',
    "f(x)",
    "q(x)+1",
    "lambda",
    "None",
    "x=y",
    "x<y",
    "x;y",
    "x:y",
    "x&y",
    "x|y",
    "x@y",
    "5 & 3",
    "2 // 1",
])
def test_forbidden_constructs_rejected(ns, text):
    with pytest.raises(MathParseError):
        p(text, ns)


def test_sympify_chain_is_closed(ns):
    """Прежний обход: parse_expr('sympify(open…)') давал builtins — закрыт."""
    with pytest.raises(MathParseError):
        p("sympify('open')", ns)


def test_unknown_becomes_symbol_not_python_object(ns):
    """Неизвестные имена — Symbol, а не найденный в namespace объект."""
    assert p("oo", ns) == sp.Symbol("oo")           # не sp.oo
    assert p("I", ns) == sp.Symbol("I")             # не мнимая единица
    assert p("open", ns) == sp.Symbol("open")       # не builtin


def test_no_side_effects(ns, monkeypatch):
    """Отказ не исполняет код: open не вызывается, файлы не трогаются."""
    calls = []
    real_open = builtins.open

    def spy(*args, **kwargs):
        calls.append(args)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    for text in ("open('README.md')", "eval('open(1)')", "exec('x')"):
        with pytest.raises(MathParseError):
            p(text, ns)
    assert calls == []


# ------------------------------------------------------------------- лимиты
def test_input_length_limit(ns):
    with pytest.raises(MathParseError):
        parse("1+" * (MAX_INPUT_LEN // 2 + 1) + "1", ns)
    # на границе — работает
    assert parse("0" * (MAX_INPUT_LEN - 1) + "1", ns) == int(
        "0" * (MAX_INPUT_LEN - 1) + "1")


def test_depth_limit(ns):
    ok = "(" * MAX_DEPTH + "1" + ")" * MAX_DEPTH
    assert parse(ok, ns) == 1
    with pytest.raises(MathParseError):
        parse("(" * (MAX_DEPTH + 1) + "1" + ")" * (MAX_DEPTH + 1), ns)


def test_node_limit(ns, monkeypatch):
    monkeypatch.setattr(math_parser, "MAX_NODES", 10)
    with pytest.raises(MathParseError):
        parse("1+1+1+1+1+1+1+1+1+1+1+1", ns)


def test_huge_power_rejected(ns):
    with pytest.raises(MathParseError):
        p("2^100000", ns)
    with pytest.raises(MathParseError):
        p("9^9^9^9", ns)
    with pytest.raises(MathParseError):
        p("(2^5000)^10000", ns)
    assert p("2^10000", ns) == 2**10000       # на границе — работает


def test_huge_factorial_rejected(ns):
    with pytest.raises(MathParseError):
        p("50001!", ns)
    assert p("50000!", ns) == sp.factorial(50000)


def test_float_huge_power_is_fast(ns):
    """Float-основания не «взрываются»: считаются с фиксированной точностью."""
    assert p("2.0^1000000", ns) == sp.Float(2.0)**1000000


# ------------------------------------------------- интеграция через движок
@pytest.fixture()
def engine():
    return CalculatorEngine("DEG")


def test_engine_calculates_normal_input(engine):
    assert engine.calculate("2^10")[1] == 1024
    assert engine.calculate("sin(30)")[0] == "1/2"
    assert engine.calculate("2pi")[1] == pytest.approx(6.283185307, rel=1e-9)
    assert engine.calculate("Ans+1")[1] == pytest.approx(7.283185307, rel=1e-9)
    assert engine.calculate("5!")[1] == 120
    assert engine.calculate("sqrt(-4)")[0] == "2i"
    assert engine.calculate("0.1+0.2")[0] == "0.3"


@pytest.mark.parametrize("text", [
    "__import__('os').system('true')",
    "sympify('open')",
    "open('/etc/passwd')",
    "eval('2+2')",
    "x.__class__",
    "__builtins__",
])
def test_engine_rejects_hostile_input(engine, text):
    with pytest.raises(CalculatorError):
        engine.calculate(text)


def test_engine_rejects_python_statements(engine):
    for text in ("import os", "lambda x: x", "[1,2]", "x[0]", "'a'"):
        with pytest.raises(CalculatorError):
            engine.calculate(text)


def test_engine_equations_and_inequalities_still_safe(engine):
    assert engine.solve_equation("2^x=8") == [3]
    assert engine.solve_inequality("(x-1)(x+2)>0") == ["x < -2 или x > 1"]
    with pytest.raises(CalculatorError):
        engine.solve_equation("sympify('open')=0")


def test_engine_error_messages_are_friendly(engine):
    """Понятные сообщения, а не traceback."""
    with pytest.raises(CalculatorError, match="Ошибка ввода"):
        engine.calculate("sin(")
    with pytest.raises(CalculatorError, match="Неизвестная функция"):
        engine.calculate("foo(1)")
    with pytest.raises(CalculatorError, match="не поддерживается"):
        engine.calculate("x[0]")
