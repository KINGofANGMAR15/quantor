"""Безопасный математический парсер «Квантора».

Замена ``sympy.parsing.sympy_parser.parse_expr`` для пользовательского ввода.
``parse_expr`` построен на Python ``eval()`` и, по предупреждению документации
SymPy, не должен применяться к неочищенному вводу: неймспейс SymPy экспортирует
``sympify``, который сам исполняет строки через ``eval`` — цепочка
``parse_expr("sympify('open')")`` даёт доступ к встроенным функциям Python.

Здесь разбор идёт иначе: небольшой токенизатор + рекурсивный спуск по
ограниченной грамматике. Каждый разрешённый узел превращается в объекты SymPy
напрямую (Integer/Float/Rational/Pow/Mul/…), всё остальное отвергается ДО
какого-либо вычисления. Ни ``eval``, ни ``exec``, ни ``compile``, ни
``sympify`` не используются.

Поддерживаемый синтаксис (совместим с прежним поведением движка):

* числа: целые, десятичные, экспоненциальная запись (``6.02e23``, ``1e-9``,
  разделители разрядов ``1_000``);
* операции: ``+ - * / %`` (остаток), ``**`` и ``^`` (возведение в степень,
  право-ассоциативно, как в Python), унарные ``+/-``;
* скобки, постфиксный факториал ``!`` и двойной ``!!``;
* неявное умножение: ``2x``, ``3sin(x)``, ``2(3+4)``, ``(x-1)(x+2)``, ``x y``;
* вызовы функций — ТОЛЬКО имена, переданные в ``ns`` (белый список движка);
  неизвестное имя со скобкой — ошибка, а не вызов;
* неизвестные идентификаторы (без скобки) -> ``Symbol`` — так работают
  переменные уравнений/графиков; в калькуляторе такие выражения далее
  отвергаются проверкой ``_exact_to_number`` («Не заданы переменные»).

Запрещено (отвергается на этапе разбора, до вычислений): доступ к атрибутам
через точку, индексация ``[]``, строки, списки/кортежи/множества, лямбды,
условные выражения, ключевые слова Python, имена с подчёркиванием, любые
вызовы вне белого списка.

Ограничения ресурсозатрат (защита от «вычислительных бомб»; сами по себе
лимиты длины строки не достаточны — поэтому есть семантические пределы):

======================  ========  =============================================
параметр                значение  смысл
======================  ========  =============================================
MAX_INPUT_LEN           600       длина одного выражения в символах
MAX_NODES               2000      число узлов (чисел, имён, операций)
MAX_DEPTH               50        вложенность скобок и вызовов
модуль целой степени    10 000    ``2^100000`` отвергается
порядок цифр результата 200 000   оценка размера Integer/Rational степеней
аргумент факториала     50 000    ``50001!`` отвергается
======================  ========  =============================================

Ошибки вычисления переданных функций (например ``CalculatorError`` от
``log(1, 1)``) пробрасываются как есть — движок сам решает, что показать
пользователю.
"""

from __future__ import annotations

import keyword
import re
from typing import Any, Dict, List, Optional, Tuple

import sympy as sp

__all__ = ["MathParseError", "parse", "MAX_INPUT_LEN", "MAX_NODES", "MAX_DEPTH"]


class MathParseError(Exception):
    """Ошибка разбора/ограничений; текст предназначен пользователю."""


# --- пределы ресурсозатрат --------------------------------------------------

#: Максимальная длина одного выражения в символах
MAX_INPUT_LEN = 600
#: Максимальное число узлов (чисел, имён, операций) в выражении
MAX_NODES = 2000
#: Максимальная вложенность скобок и вызовов функций
MAX_DEPTH = 50
#: Максимальный модуль ЦЕЛОЙ степени (2^100000 отвергается до вычисления)
_MAX_POW_EXP = 10_000
#: Оценка максимума десятичных цифр результата степени/факториала
_MAX_RESULT_DIGITS = 200_000
#: Максимальный целый аргумент факториала
_MAX_FACTORIAL_N = 50_000

#: Константы самого парсера (остальные имена движок передаёт через ns)
_BUILTIN_CONSTANTS: Dict[str, sp.Basic] = {"pi": sp.pi}

_NUM_RE = re.compile(
    r"(?:\d[\d_]*(?:\.\d[\d_]*)?|\.\d[\d_]*)(?:[eE][+-]?\d+)?")
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

Token = Tuple[str, str]


# --- токенизатор --------------------------------------------------------------

def _tokenize(text: str) -> List[Token]:
    """Разбить строку на числа, имена и операторы. Всё прочее — ошибка."""
    tokens: List[Token] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        m = _NUM_RE.match(text, i)
        if m:
            tokens.append(("num", m.group()))
            i = m.end()
            continue
        m = _NAME_RE.match(text, i)
        if m:
            name = m.group()
            if name.startswith("_"):
                raise MathParseError(f"Недопустимое имя «{name}»")
            if keyword.iskeyword(name):
                raise MathParseError(
                    f"Конструкция «{name}» не поддерживается")
            tokens.append(("name", name))
            i = m.end()
            continue
        if text.startswith("**", i):
            tokens.append(("op", "**"))
            i += 2
            continue
        if ch == "^":                      # степень, а не побитовый XOR
            tokens.append(("op", "**"))
            i += 1
            continue
        if ch in "+-*/(),!%":
            tokens.append(("op", ch))
            i += 1
            continue
        raise MathParseError(f"Символ «{ch}» не поддерживается")
    return tokens


# --- построение узлов SymPy с защитой от «бомб» -------------------------------

def _make_number(text: str) -> sp.Basic:
    """Литерал числа: Integer либо Float (точность как у прежнего парсера)."""
    t = text.replace("_", "")
    if "." in t or "e" in t or "E" in t:
        return sp.Float(t, 15)
    return sp.Integer(t)


def _integer_digits(value: sp.Basic) -> int:
    """Оценка числа десятичных цифр целого/рационального значения."""
    if isinstance(value, sp.Integer):
        return len(str(abs(int(value))))
    p, q = value.as_numer_denom()
    return len(str(abs(int(p)))) + len(str(abs(int(q))))


def _make_pow(base: sp.Basic, exp: sp.Basic) -> sp.Basic:
    """Возведение в степень с проверкой размера ДО вычисления SymPy.

    Опасны только точные (Integer/Rational) основания с целым показателем:
    SymPy вычисляет их точно, и ``2^(10^10)`` исчерпал бы память.
    Float-основания считаются mpmath с фиксированной точностью,
    символические остаются невычисленными — считать безопасно.
    """
    if isinstance(exp, sp.Integer) and isinstance(base, sp.Basic):
        e = abs(int(exp))
        if base.is_Integer or base.is_Rational:
            if e > _MAX_POW_EXP:
                raise MathParseError(
                    f"Степень слишком велика (максимум {_MAX_POW_EXP})")
            if e:
                digits = e * _integer_digits(base)
                if digits > _MAX_RESULT_DIGITS:
                    raise MathParseError(
                        "Слишком большая степень: результат не поместится "
                        f"(~{digits} цифр)")
    return base ** exp


def _make_factorial(node: sp.Basic, bangs: int) -> sp.Basic:
    """Постфиксный факториал: ``x!`` и ``x!!`` (двойной)."""
    if bangs > 2:
        raise MathParseError("Многократный факториал «!!!» не поддерживается")
    if isinstance(node, sp.Integer) and int(node) > _MAX_FACTORIAL_N:
        raise MathParseError(
            f"Аргумент факториала слишком велик (максимум {_MAX_FACTORIAL_N})")
    return sp.factorial2(node) if bangs == 2 else sp.factorial(node)


# --- рекурсивный спуск ---------------------------------------------------------

class _Parser:
    """Разбор потока токенов по грамматике:

        expr   := term (('+' | '-') term)*
        term   := unary (('*' | '/' | '%') unary | ОПЕРАНД)*   # и неявное «*»
        unary  := ('+' | '-')* power
        power  := postfix ('**' unary)?          # право-ассоциативно
        postfix:= primary '!'*
        primary:= ЧИСЛО | ИМЯ [ '(' args ')' ] | '(' expr ')'
    """

    def __init__(self, tokens: List[Token],
                 ns: Dict[str, Any]) -> None:
        self._toks = tokens
        self._i = 0
        self._ns = ns
        self._nodes = 0
        self._depth = 0

    # --- служебное ---------------------------------------------------------
    def _peek(self) -> Optional[Token]:
        return self._toks[self._i] if self._i < len(self._toks) else None

    def _next(self) -> Token:
        tok = self._toks[self._i]
        self._i += 1
        return tok

    def _tick(self) -> None:
        self._nodes += 1
        if self._nodes > MAX_NODES:
            raise MathParseError("Выражение слишком сложное")

    def _enter(self) -> None:
        self._depth += 1
        if self._depth > MAX_DEPTH:
            raise MathParseError("Слишком глубокая вложенность выражения")

    # --- правила -----------------------------------------------------------
    def parse(self) -> sp.Basic:
        node = self._expr()
        rest = self._peek()
        if rest is not None:
            raise MathParseError(f"Лишний элемент «{rest[1]}»")
        return node

    def _expr(self) -> sp.Basic:
        node = self._term()
        while (tok := self._peek()) and tok[1] in ("+", "-"):
            op = self._next()[1]
            rhs = self._term()
            node = node + rhs if op == "+" else node - rhs
            self._tick()
        return node

    def _term(self) -> sp.Basic:
        node = self._unary()
        while (tok := self._peek()):
            if tok[1] in ("*", "/", "%"):
                op = self._next()[1]
                rhs = self._unary()
                if op == "*":
                    node = node * rhs
                elif op == "/":
                    node = node / rhs          # 1/0 -> zoo (проверит движок)
                else:
                    node = sp.Mod(node, rhs)
                self._tick()
            elif tok[0] in ("num", "name") or tok[1] == "(":
                # неявное умножение: 2x, 3sin(x), 2(3+4), (x-1)(x+2)
                rhs = self._unary()
                node = node * rhs
                self._tick()
            else:
                return node
        return node

    def _unary(self) -> sp.Basic:
        tok = self._peek()
        if tok and tok[1] in ("+", "-"):
            self._next()
            node = self._unary()
            return node if tok[1] == "+" else -node
        return self._power()

    def _power(self) -> sp.Basic:
        base = self._postfix()
        if (tok := self._peek()) and tok[1] == "**":
            self._next()
            exp = self._unary()      # право-ассоциативно: 2^3^2 = 2^(3^2)
            return _make_pow(base, exp)
        return base

    def _postfix(self) -> sp.Basic:
        node = self._primary()
        while (tok := self._peek()) and tok[1] == "!":
            bangs = 0
            while (tok := self._peek()) and tok[1] == "!":
                self._next()
                bangs += 1
            node = _make_factorial(node, bangs)
            self._tick()
        return node

    def _primary(self) -> sp.Basic:
        tok = self._peek()
        if tok is None:
            raise MathParseError("Неожиданный конец выражения")
        kind, text = tok
        if kind == "num":
            self._next()
            self._tick()
            return _make_number(text)
        if kind == "name":
            self._next()
            self._tick()
            return self._name(text)
        if text == "(":
            self._next()
            self._enter()
            node = self._expr()
            if not ((nxt := self._peek()) and nxt[1] == ")"):
                raise MathParseError("Не закрыта скобка")
            self._next()
            self._depth -= 1
            return node
        raise MathParseError(f"Неожиданный элемент «{text}»")

    def _name(self, name: str) -> sp.Basic:
        """Имя: функция из белого списка, значение из ns или новый Symbol."""
        if name.startswith("_"):            # двойная страховка к лексеру
            raise MathParseError(f"Недопустимое имя «{name}»")
        is_call = (tok := self._peek()) is not None and tok[1] == "("
        value = self._ns.get(name)
        if value is None:
            value = _BUILTIN_CONSTANTS.get(name)
        if is_call:
            if value is None:
                # вызовы разрешены ТОЛЬКО для функций из белого списка
                raise MathParseError(f"Неизвестная функция «{name}»")
            if not callable(value):
                # x(x+1) — это неявное умножение, а не вызов;
                # скобку обработает _term как начало нового операнда
                return value
            return self._call(name, value)
        if value is None:
            return sp.Symbol(name)          # неизвестный идентификатор
        if callable(value):
            raise MathParseError(
                f"Функция «{name}» требует аргументы: {name}(…)")
        return value

    def _call(self, name: str, func) -> sp.Basic:
        self._next()                        # '('
        self._enter()
        args: List[sp.Basic] = []
        if (tok := self._peek()) and tok[1] == ")":
            self._next()
            self._depth -= 1
        else:
            args.append(self._expr())
            while (tok := self._peek()) and tok[1] == ",":
                self._next()
                args.append(self._expr())
            if not ((end := self._peek()) and end[1] == ")"):
                raise MathParseError(
                    f"Не закрыта скобка вызова «{name}»")
            self._next()
            self._depth -= 1
        # Ошибки вычисления (CalculatorError, TypeError при неверном числе
        # аргументов и т.п.) пробрасываются как есть — их переводит движок.
        return func(*args)


# --- публичный вход -------------------------------------------------------------

def parse(text: str, ns: Dict[str, Any]) -> sp.Basic:
    """Разобрать математическое выражение в объект SymPy БЕЗ eval.

    ``ns`` — словарь имён, собранный движком: значения (числа, Symbol,
    константы) и разрешённые функции (вызываемые объекты). Вызов по имени
    возможен только для ключей этого словаря; неизвестные имена без скобки
    становятся ``Symbol``.
    """
    text = text or ""
    if len(text) > MAX_INPUT_LEN:
        raise MathParseError(
            f"Выражение слишком длинное (максимум {MAX_INPUT_LEN} символов)")
    tokens = _tokenize(text)
    if not tokens:
        raise MathParseError("Пустой ввод")
    return _Parser(tokens, ns).parse()
