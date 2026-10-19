"""Математическое ядро «Квантора».

Все вычисления выполняются локально (SymPy), без сети и внешних сервисов.

CalculatorEngine — фасад для GUI и тестов:

    calculate(expr)          -> (текст результата, число)
    solve_polynomial(coeffs) -> все корни полинома (включая комплексные)
    solve_equation(eq)       -> корни общего уравнения (линейные, дроби, 2^x=8)

Ячейка памяти: mem_add / mem_sub / mem_recall / mem_clear.
Результат последнего вычисления: engine.ans (вводится кнопкой Ans).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple, Union

import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from . import constants as K

Number = Union[int, float, complex]
Result = Tuple[str, Number]

ANGLE_MODES = ("DEG", "RAD")

#: Преобразования парсера: стандартные + неявное умножение ("2pi" -> 2*pi,
#: "3(4+5)" -> 3*(4+5)) и неявное применение функций ("sin 30" -> sin(30)).
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

# Точки старта для численного поиска корней nsolve (обычные уравнения).
_NUM_STARTS = (-20.0, -10.0, -5.0, -2.0, -1.0, -0.5, 0.001,
               0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0)


class CalculatorError(Exception):
    """Ошибка, которую можно показать пользователю на дисплее."""


# ---------------------------------------------------------------------------
# Преобразование и форматирование значений
# ---------------------------------------------------------------------------

def _exact_to_number(node: sp.Basic) -> Number:
    """Точное sympy-значение -> python-число (int/float/complex).

    Целые остаются целыми; комплексное a+bi с ничтожной мнимой/вещественной
    частью сокращается до числа соответствующего типа.
    """
    if node is sp.zoo or node == sp.nan:
        raise CalculatorError("Деление на ноль")
    if node in (sp.oo, -sp.oo):
        raise CalculatorError("Результат бесконечен")
    if getattr(node, "free_symbols", None):
        names = ", ".join(sorted(str(s) for s in node.free_symbols))
        raise CalculatorError("Не заданы переменные: " + names)
    if getattr(node, "is_Integer", False):
        return int(node)
    value = complex(sp.N(node, 15))
    scale = max(1.0, abs(value.real), abs(value.imag))
    if abs(value.imag) <= 1e-11 * scale:
        return value.real
    if abs(value.real) <= 1e-11 * scale:
        return complex(0.0, value.imag)
    return value


def _fmt_float(v: float) -> str:
    """Число в стиле дисплея калькулятора (до 10 значащих цифр)."""
    if v != v:  # NaN
        return "Ошибка"
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    a = abs(v)
    if a != 0.0 and (a >= 1e12 or a < 1e-9):
        s = f"{v:.8e}"
        mant, exp = s.split("e")
        mant = mant.rstrip("0").rstrip(".")
        return f"{mant}e{int(exp)}"
    return f"{v:.10g}"


def _fmt_number(v: Number) -> str:
    """Строка для дисплея: 0.5, 1.414213562, 1+2i, 6.62607015e-34."""
    if isinstance(v, complex):
        re_txt = _fmt_float(v.real) if abs(v.real) > 1e-12 else ""
        im = v.imag
        if abs(im) <= 1e-12:
            return re_txt or "0"
        sign = "-" if im < 0 else ""
        im_txt = _fmt_float(abs(im))
        if im_txt == "1":
            im_txt = ""  # чистое i: "i", "-i"
        if re_txt and im_txt:
            return f"{re_txt}{'+' if im > 0 else '-'}{im_txt}i"
        if im_txt:
            return f"{sign}{im_txt}i"
        return sign + "i"
    return _fmt_float(v)


def _sort_roots(roots: Sequence[Number]) -> List[Number]:
    """Вещественные корни по возрастанию, затем комплексные (re, im)."""
    real = sorted(v for v in roots if not isinstance(v, complex))
    comp = sorted((v for v in roots if isinstance(v, complex)),
                  key=lambda z: (z.real, z.imag))
    return real + comp


def _preprocess(text: str) -> str:
    """Кнопочные/типографские символы -> синтаксис SymPy."""
    for bad, good in (("×", "*"), ("·", "*"), ("÷", "/"), ("−", "-"),
                      ("–", "-"), ("π", "pi"), ("²", "**2"), ("³", "**3"),
                      ("⁴", "**4"), ("√", "sqrt(")):
        text = text.replace(bad, good)
    text = text.replace("^", "**")
    # Дописываем недостающие закрывающие скобки (как на Casio при «=»).
    opened = text.count("(") - text.count(")")
    if opened > 0:
        text += ")" * opened
    return text.strip()


# ---------------------------------------------------------------------------
# Движок
# ---------------------------------------------------------------------------

class CalculatorEngine:
    """Ядро калькулятора: вычисления, уравнения, память, режим углов."""

    def __init__(self, angle_mode: str = "DEG") -> None:
        self.set_angle_mode(angle_mode)
        self.memory: float = 0.0
        self.ans: Number = 0.0

    # --- режим углов -------------------------------------------------------
    def set_angle_mode(self, mode: str) -> None:
        if mode not in ANGLE_MODES:
            raise ValueError(f"Неизвестный режим углов: {mode!r}")
        self.angle_mode = mode

    # --- пространство имён парсера -----------------------------------------
    def _namespace(self) -> Dict[str, Any]:
        ns: Dict[str, Any] = {
            "x": sp.Symbol("x"), "Ans": self.ans, "ans": self.ans,
            "sqrt": sp.sqrt, "abs": sp.Abs, "exp": sp.exp,
        }
        deg = self.angle_mode == "DEG"
        to_rad = (lambda a: sp.pi * a / 180) if deg else (lambda a: a)
        to_deg = (lambda a: a * 180 / sp.pi) if deg else (lambda a: a)
        ns["sin"], ns["cos"], ns["tan"] = (
            (lambda a: sp.sin(to_rad(a))),
            (lambda a: sp.cos(to_rad(a))),
            (lambda a: sp.tan(to_rad(a))),
        )
        ns["asin"], ns["acos"], ns["atan"] = (
            (lambda a: to_deg(sp.asin(a))),
            (lambda a: to_deg(sp.acos(a))),
            (lambda a: to_deg(sp.atan(a))),
        )
        ns["arcsin"], ns["arccos"], ns["arctan"] = ns["asin"], ns["acos"], ns["atan"]
        ns["sinh"], ns["cosh"], ns["tanh"] = sp.sinh, sp.cosh, sp.tanh
        ns["ln"] = sp.log
        ns["log"] = self._logb
        ns["log10"] = lambda a: sp.log(a, 10)
        ns["log2"] = lambda a: sp.log(a, 2)
        ns["mod"] = self._mod
        ns.update(K.ALLOWED)  # константы: PI, C, h, G, ...
        return ns

    @staticmethod
    def _logb(a, b=None):
        """log(x) — натуральный; log(x, b) — по произвольному основанию."""
        if b is None:
            return sp.log(a)
        if sp.simplify(b - 1) == 0:
            raise CalculatorError("Основание логарифма не может быть 1")
        return sp.log(a) / sp.log(b)

    @staticmethod
    def _mod(a, b):
        """Остаток: mod(a, b). Для a>0 совпадает с привычным делением."""
        return sp.Mod(a, b)

    def _parse(self, text: str, ns: Dict[str, Any]) -> sp.Basic:
        cleaned = _preprocess(text)
        if not cleaned:
            raise CalculatorError("Пустой ввод")
        try:
            return parse_expr(cleaned, local_dict=ns,
                              transformations=_TRANSFORMS, evaluate=True)
        except CalculatorError:
            raise
        except Exception:
            raise CalculatorError("Ошибка ввода")

    # --- основное вычисление ------------------------------------------------
    def calculate(self, expr: str) -> Result:
        """Вычислить выражение. Вернуть (текст для дисплея, значение)."""
        text = (expr or "").strip()
        if not text:
            raise CalculatorError("Пустой ввод")
        if "=" in text:
            return self._solve_and_format(text)
        ns = self._namespace()
        node = self._parse(text, ns)
        try:
            node = sp.simplify(node)
        except Exception:
            pass  # неудачный simplify не мешает численному результату
        value = _exact_to_number(node)
        self.ans = value
        return _fmt_number(value), value

    # --- память --------------------------------------------------------------
    def mem_add(self, value: Number) -> None:
        self.memory += float(value)

    def mem_sub(self, value: Number) -> None:
        self.memory -= float(value)

    def mem_recall(self) -> float:
        return self.memory

    def mem_clear(self) -> None:
        self.memory = 0.0

    # --- полиномы -------------------------------------------------------------
    def solve_polynomial(self, coeffs, var: str = "x") -> List[Number]:
        """Корни полинома. ``coeffs`` — коэффициенты ПО УБЫВАНИЮ степеней:
        [1, -6, 11, -6] -> x^3 - 6x^2 + 11x - 6. Либо строка-выражение."""
        if isinstance(coeffs, str):
            ns = {"x": sp.Symbol(var), "Ans": self.ans}
            self._trig_into(ns)
            node = self._parse(coeffs, ns)
            poly_expr = sp.expand(node)
        else:
            values = list(coeffs)
            if len(values) < 2:
                raise CalculatorError("Нужно минимум два коэффициента")
            poly_expr = sp.Poly([sp.Rational(c) for c in values],
                                sp.Symbol(var)).as_expr()
        try:
            roots = sp.solve(poly_expr, sp.Symbol(var))
        except Exception:
            raise CalculatorError("Не удалось решить полином")
        if not roots:
            raise CalculatorError("Корней не найдено")
        return _sort_roots([_exact_to_number(r) for r in roots])

    def _trig_into(self, ns: Dict[str, Any]) -> None:
        """Тригонометрия/логарифмы без физических констант (для уравнений)."""
        ns.update({k: v for k, v in self._namespace().items()
                   if k not in K.ALLOWED and k not in ("pi", "e")})
        ns["pi"], ns["e"] = sp.pi, sp.E

    # --- общие уравнения --------------------------------------------------------
    def solve_equation(self, eq: str, var: str = "x") -> List[Number]:
        """Решить уравнение вида «LHS = RHS» (если RHS нет — считать 0)."""
        eq = (eq or "").strip()
        if eq.count("=") > 1:
            raise CalculatorError("Нужно ровно одно «=» (например: 2^x = 8)")
        if "=" in eq:
            lhs, rhs = (part.strip() for part in eq.split("="))
        else:
            lhs, rhs = eq, "0"  # без правой части считаем RHS = 0
        sym = sp.Symbol(var)
        ns = {var: sym, "Ans": self.ans, "sqrt": sp.sqrt, "exp": sp.exp,
              "abs": sp.Abs, "ln": sp.log, "log": self._logb}
        self._trig_into(ns)
        try:
            f = self._parse(lhs, ns) - self._parse(rhs, ns)
        except CalculatorError as exc:
            raise CalculatorError(f"Ошибка ввода: {exc}") from exc
        f = sp.expand(f)
        if sp.simplify(f) == 0:
            raise CalculatorError("Бесконечно много решений")
        if sym not in f.free_symbols:
            raise CalculatorError("Нет решений")
        try:
            solutions = sp.solve(f, sym)
        except Exception:
            solutions = []
        if not solutions:
            solutions = self._numeric_roots(f, sym)
            if not solutions:
                raise CalculatorError("Корней не найдено")
        return _sort_roots([_exact_to_number(s) for s in solutions])

    def _numeric_roots(self, f: sp.Basic, sym: sp.Symbol) -> List[float]:
        """Численный запасной поиск (nsolve с разных стартов), только вещественные."""
        found: List[float] = []
        for start in _NUM_STARTS:
            try:
                r = sp.nsolve(f, sym, start, tol=1e-14, maxsteps=60)
            except Exception:
                continue
            xv = float(sp.N(r))
            if not math.isfinite(xv):
                continue
            try:
                residue = abs(float(sp.N(f.subs(sym, xv))))
            except (TypeError, ValueError):
                continue
            if residue > 1e-6:
                continue
            if not any(abs(xv - r0) <= 1e-7 * max(1.0, abs(xv)) for r0 in found):
                found.append(xv)
        return found

    # --- служебное ---------------------------------------------------------------
    def _solve_and_format(self, text: str) -> Result:
        roots = self.solve_equation(text)
        self.ans = roots[0] if roots else 0.0
        pretty = ", ".join(_fmt_number(r) for r in roots)
        return f"x = {pretty}", (roots[0] if roots else 0.0)
