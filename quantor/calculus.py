"""Символьное и численное интегрирование и дифференцирование «Квантора».

CalculusEngine:
    indefinite_integral(expr, var)             -> (F(x)+C, latex) или (None, '')
    definite_integral(expr, var, a, b)         -> dict(symbolic, numeric, latex, ...)
    derivative(expr, var, order)               -> (результат, latex)
    derivative_at_point(expr, var, x0, order)  -> float

Плюс разбор запросов из строки ввода (evaluate_query): LaTeX-конструкции::

    \\int x^2 dx,  \\int_{0}^{\\pi} \\sin(x) dx,
    \\frac{d}{dx} \\sin(x),  \\frac{d^2}{dx^2} x^4,
    \\frac{d}{dx} \\sin(x) \\big|_{x=0}

и текстовые формы integrate(f, x) / diff(f, x[, n]).

Численное интегрирование — только mpmath.quad (SciPy не используется).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import mpmath
import sympy as sp

from .engine import CalculatorEngine, CalculatorError
from .latex_input import LatexParseError, is_latex, parse_input, to_latex

Number = Union[int, float, complex]


@dataclass
class CalculusResult:
    """Результат запроса для GUI: текстовые строки + LaTeX + значение для Ans."""

    lines: List[str]
    latex: str
    value: Optional[Number] = None


# ---------------------------------------------------------------------------
# Регулярные выражения запросов
# ---------------------------------------------------------------------------

_RE_INT_DEF = re.compile(
    r"^\s*\\int\s*_\{([^{}]+)\}\s*\^\{([^{}]+)\}\s*(.+?)\s*d\s*([a-zA-Z])\s*$")
_RE_INT_INDEF = re.compile(
    r"^\s*\\int\s+(.+?)\s*d\s*([a-zA-Z])\s*$")
_RE_DERIV = re.compile(
    r"^\s*\\frac\s*\{\s*d(\^\{?(\d+)\}?)?\s*\}"
    r"\s*\{\s*d([a-zA-Z])(\^\{?(\d+)\}?)?\s*\}\s*(.+?)\s*$")
_RE_EVAL_AT = re.compile(
    r"^(.*?)\s*(?:\\[bB]igg?e?r?)?\s*\|\s*_\{\s*([a-zA-Z])\s*=\s*(.+?)\s*\}\s*$")
_RE_INTEGRATE_CALL = re.compile(
    r"^\s*integrate\s*\(\s*(.+?)\s*,\s*([a-zA-Z])\s*\)\s*$", re.IGNORECASE)
_RE_DIFF_CALL = re.compile(
    r"^\s*diff\s*\(\s*(.+?)\s*,\s*([a-zA-Z])(?:\s*,\s*(\d+))?\s*\)\s*$",
    re.IGNORECASE)


def is_calculus_query(text: str) -> bool:
    """Похоже ли на запрос интеграла/производной (до попытки парсинга)."""
    t = (text or "").strip()
    if not t:
        return False
    if t.startswith("\\int") or t.startswith("\\frac{d"):
        return True
    return bool(re.match(r"^\s*(integrate|diff)\s*\(", t, re.IGNORECASE))


class CalculusEngine:
    """Интегралы и производные поверх SymPy + mpmath."""

    def __init__(self, parse_engine: Optional[CalculatorEngine] = None) -> None:
        self._parse_engine = parse_engine

    # ------------------------------------------------------------------ API
    def indefinite_integral(self, expr: Any, var: str = "x"):
        """∫f dx -> (F(x)+C, latex). Если символьно не взялось — (None, '')."""
        f = self._as_expr(expr, var)
        F = sp.integrate(f, sp.Symbol(var))
        if F.has(sp.Integral):  # SymPy оставил первообразную невычисленной
            return None, ""
        # ' + C' добавляем вручную, чтобы в LaTeX константа стояла последней
        return F + sp.Symbol("C"), to_latex(F) + " + C"

    def definite_integral(self, expr: Any, var: str = "x",
                          a: Any = 0, b: Any = 0) -> dict:
        """∫ₐᵇ f dx -> dict(symbolic, numeric, latex, lines, value).

        symbolic — точное значение (или None, если SymPy не справился);
        numeric  — число по mpmath.quad (или None при особой точке).
        """
        sym = sp.Symbol(var)
        f = self._as_expr(expr, var)
        a_s, b_s = self._as_expr(a), self._as_expr(b)

        symbolic = None
        try:
            exact = sp.integrate(f, (sym, a_s, b_s))
            if not exact.has(sp.Integral):
                symbolic = sp.simplify(exact)
        except Exception:
            symbolic = None

        numeric, value = self._numeric_integral(f, sym, a_s, b_s)

        head = to_latex(sp.Integral(f, (sym, a_s, b_s)))
        if symbolic is not None:
            tail = to_latex(symbolic)
            value = value if value is not None else _to_number(symbolic)
        else:
            tail = to_latex(f"{numeric:.10f}") if numeric is not None else "?"
        return {
            "symbolic": symbolic,
            "numeric": numeric,
            "latex": f"{head} = {tail}",
            "lines": _integral_lines(symbolic, numeric),
            "value": value,
        }

    def derivative(self, expr: Any, var: str = "x", order: int = 1):
        """d^n f/dx^n -> (результат, latex)."""
        order = int(order)
        if order < 1 or order > 10:
            raise ValueError("Порядок производной: от 1 до 10")
        f = self._as_expr(expr, var)
        d = sp.diff(f, sp.Symbol(var), order)
        return d, to_latex(d)

    def derivative_at_point(self, expr: Any, var: str = "x",
                            x0: Any = 0, order: int = 1) -> float:
        """Значение d^n f/dx^n в точке x0 (evalf)."""
        d, _ = self.derivative(expr, var, order)
        val = d.subs(sp.Symbol(var), self._as_expr(x0))
        if val.free_symbols:
            raise ValueError("Точка должна быть числом")
        return float(sp.N(val))

    # ------------------------------------------------- разбор строки запроса
    def evaluate_query(self, text: str) -> CalculusResult:
        """Распознать и вычислить интеграл/производную из строки ввода.

        Возвращает CalculusResult или бросает LatexParseError/ValueError
        с понятным сообщением. Если это вообще не calculus-запрос — None.
        """
        t = (text or "").strip()
        if not t:
            return None

        m = _RE_INT_DEF.match(t)
        if m:
            a, b, body, var = m.group(1), m.group(2), m.group(3), m.group(4)
            return self._definite_from_parts(body, var, a, b)

        m = _RE_INT_INDEF.match(t)
        if m:
            body, var = m.group(1), m.group(2)
            result, latex = self.indefinite_integral(body, var)
            if result is None:
                raise ValueError(
                    "Символьное решение не найдено "
                    "(первообразная не выражается в элементарных функциях)")
            return CalculusResult(
                lines=[f"первообразная: {_plain(result)}"],
                latex=f"\\int {to_latex(self._as_expr(body, var))}\\,d{var} = {latex}",
                value=None)

        m = _RE_DERIV.match(t)
        if m:
            return self._derivative_from_match(m)

        m = _RE_INTEGRATE_CALL.match(t)
        if m:
            body, var = m.group(1), m.group(2)
            result, latex = self.indefinite_integral(body, var)
            if result is None:
                raise ValueError("Символьное решение не найдено")
            return CalculusResult(lines=[f"первообразная: {_plain(result)}"],
                                  latex=latex, value=None)

        m = _RE_DIFF_CALL.match(t)
        if m:
            body, var = m.group(1), m.group(2)
            order = int(m.group(3) or 1)
            result, latex = self.derivative(body, var, order)
            return CalculusResult(lines=[_plain(result)], latex=latex, value=None)

        return None  # не calculus-запрос

    # ------------------------------------------------------------ внутренности
    def _definite_from_parts(self, body: str, var: str, a: str, b: str) -> CalculusResult:
        try:
            data = self.definite_integral(body, var, a, b)
        except LatexParseError:
            raise
        except CalculatorError as exc:
            raise ValueError(str(exc)) from exc
        return CalculusResult(lines=data["lines"], latex=data["latex"],
                              value=data["value"])

    def _derivative_from_match(self, m: "re.Match") -> CalculusResult:
        num_order = int(m.group(2) or 1)
        var = m.group(3)
        den_order = int(m.group(5) or 1)
        rest = m.group(6)
        if num_order != den_order:
            raise LatexParseError(
                "Порядок производной в числителе и знаменателе должен совпадать")

        at_point = None
        em = _RE_EVAL_AT.match(rest)
        if em:
            rest, at_var, at_val = em.group(1), em.group(2), em.group(3)
            if at_var != var:
                raise LatexParseError(
                    f"Переменная в |_{at_var}=... не совпадает с d{var}")
            at_point = at_val

        result, latex = self.derivative(rest, var, num_order)
        order_tex = f"^{{{num_order}}}" if num_order > 1 else ""
        head = f"\\frac{{d{order_tex}}}{{d{var}{order_tex}}}"
        body_latex = to_latex(self._as_expr(rest, var))

        if at_point is not None:
            value = self.derivative_at_point(rest, var, at_point, num_order)
            shown = f"{value:.10g}"
            if value == int(value):
                shown = f"{value:.1f}"
            return CalculusResult(
                lines=[f"значение в точке {at_var}={_plain(at_point)}: {shown}"],
                latex=f"{head}{body_latex}\\big|_{{{var}={_plain(at_point)}}} = {shown}",
                value=value)
        return CalculusResult(
            lines=[_plain(result)],
            latex=f"{head}{body_latex} = {latex}",
            value=None)

    def _numeric_integral(self, f: sp.Basic, sym: sp.Symbol,
                          a_s: sp.Basic, b_s: sp.Basic):
        """Численное значение через mpmath.quad; (None, None) при неудаче."""
        if a_s.free_symbols or b_s.free_symbols:
            return None, None
        saved_dps = mpmath.mp.dps
        mpmath.mp.dps = 30
        try:
            func = sp.lambdify(sym, f, modules=["mpmath"])
            lo = self._to_mpmath(a_s)
            hi = self._to_mpmath(b_s)
            val = mpmath.quad(func, [lo, hi])
            num = float(val)
            if not _finite(num):
                return None, None
            return num, num
        except Exception:
            return None, None
        finally:
            mpmath.mp.dps = saved_dps

    @staticmethod
    def _to_mpmath(limit: sp.Basic):
        if limit in (sp.oo, -sp.oo):
            return mpmath.inf if limit is sp.oo else -mpmath.inf
        return mpmath.mpf(str(sp.N(limit, 25)))

    def _as_expr(self, obj: Any, var: str = "x") -> sp.Basic:
        """Строка -> SymPy-выражение (LaTeX или обычная), Expr -> как есть."""
        if isinstance(obj, sp.Basic):
            return obj
        return parse_input(str(obj), self._parse_engine)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _plain(value: Any) -> str:
    """Компактная текстовая форма SymPy-выражения/числа (для строки результата)."""
    if isinstance(value, sp.Basic):
        if value.free_symbols:
            return str(value).replace("**", "^")
        from .engine import _exact_to_number
        from .engine import _fmt_number
        return _fmt_number(_exact_to_number(value))
    return str(value)


def _to_number(expr: sp.Basic) -> Optional[Number]:
    try:
        from .engine import _exact_to_number
        return _exact_to_number(expr)
    except Exception:
        return None


def _finite(v: float) -> bool:
    return v == v and v not in (float("inf"), float("-inf"))


def _integral_lines(symbolic: Optional[sp.Basic], numeric: Optional[float]) -> List[str]:
    lines: List[str] = []
    if symbolic is not None:
        lines.append(f"точно: {_plain(symbolic)}")
    else:
        lines.append("символьное решение не найдено")
    if numeric is not None:
        lines.append(f"численно: {numeric:.10f}")
    elif symbolic is None:
        lines.append("численно не удалось (особая точка?)")
    return lines
