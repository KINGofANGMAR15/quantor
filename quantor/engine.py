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
import re
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

# Знаки неравенств: <= раньше <, >= раньше > (иначе поймаем половину знака)
INEQ_RE = re.compile(r"<=|>=|≤|≥|<|>")

#: Буквы-константы, которые нельзя считать переменными неравенства
_INEQ_CONSTANT_LETTERS = {"e", "E"}


def _guess_var(text: str) -> str:
    """Переменная неравенства по тексту: одиночные латинские буквы.
    Приоритет x, y, z, t, …; одиночные буквы-константы (e, E) пропускаем."""
    letters = re.findall(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])", text)
    letters = [ch for ch in letters if ch not in _INEQ_CONSTANT_LETTERS]
    for pref in ("x", "y", "z", "t", "u", "v", "w", "n"):
        if pref in letters:
            return pref
    return letters[0] if letters else "x"


def _bound_str(bound: sp.Basic) -> str:
    """Граница интервала: '√2', '3/2', '-1.324717957'."""
    if bound in (sp.oo, -sp.oo):
        return ""
    return format_exact(bound) or _fmt_number(_exact_to_number(bound))


def _dedupe_roots(roots: List[sp.Basic]) -> List[sp.Basic]:
    """Убрать кратные корни, сохранив порядок."""
    out: List[sp.Basic] = []
    for r in roots:
        if not any(sp.simplify(r - o) == 0 for o in out):
            out.append(r)
    return out


def _sample_f(f: sp.Basic, sym: sp.Symbol, lo: sp.Basic, hi: sp.Basic) -> float:
    """Знак f в пробной точке интервала (lo, hi)."""
    if lo == -sp.oo and hi == sp.oo:
        x0 = 0.0
    elif lo == -sp.oo:
        x0 = float(sp.N(hi)) - 1.0
    elif hi == sp.oo:
        x0 = float(sp.N(lo)) + 1.0
    else:
        x0 = (float(sp.N(lo)) + float(sp.N(hi))) / 2.0
    return float(sp.N(f.subs(sym, x0)))


def _poly_ineq_set(f: sp.Basic, sym: sp.Symbol, op: str) -> sp.Set:
    """Решение «f(sym) <один из <,<=,>,>=» 0 через корни и знаки интервалов."""
    strict = op in ("<", ">")
    want_pos = op in (">", ">=")
    roots = _dedupe_roots(list(sp.Poly(f, sym).real_roots()))

    points: List[sp.Basic] = []
    try:
        roots = sorted(roots, key=lambda r: float(sp.N(r)))
    except (TypeError, ValueError):
        pass

    intervals: List[sp.Interval] = []
    points = [-sp.oo] + roots + [sp.oo]
    for lo, hi in zip(points, points[1:]):
        if lo == hi:            # вырожденный интервал между дубликатами
            continue
        val = _sample_f(f, sym, lo, hi)
        if (val > 0) == want_pos:
            intervals.append(sp.Interval(
                lo, hi,
                left_open=(lo in (-sp.oo, sp.oo) or strict),
                right_open=(hi in (-sp.oo, sp.oo) or strict)))

    parts: List[sp.Set] = list(intervals)
    if not strict:
        covered = sp.Union(*intervals) if intervals else sp.S.EmptySet
        extra = [r for r in roots if not bool(covered.contains(r))]
        if extra:
            parts.append(sp.FiniteSet(*extra))
    if not parts:
        return sp.S.EmptySet
    return sp.Union(*parts)


def _format_interval(iv: sp.Interval, var: str) -> str:
    """Interval(-2, 2, True, True) -> '-2 < x < 2' (знаки ≤/≥ для границ)."""
    low, high = _bound_str(iv.start), _bound_str(iv.end)
    l_open, r_open = iv.left_open, iv.right_open
    if iv.start in (-sp.oo, sp.oo):
        op = "<" if r_open else "≤"     # x слева от конца интервала
        return f"{var} {op} {high}"
    if iv.end in (-sp.oo, sp.oo):
        op = ">" if l_open else "≥"     # x справа от начала интервала
        return f"{var} {op} {low}"
    l_op = "<" if l_open else "≤"
    r_op = "<" if r_open else "≤"
    return f"{low} {l_op} {var} {r_op} {high}"


def _format_solution_set(sol: sp.Set, var: str = "x") -> List[str]:
    """Множество решений -> текстовые части: ['x < -2', 'x > 2']."""
    if sol == sp.S.EmptySet:
        return ["решений нет (∅)"]
    if sol == sp.S.Reals:
        return [f"{var} — любое действительное число"]
    if isinstance(sol, sp.FiniteSet):
        pts = [_bound_str(p) for p in sol.args]
        return [f"{var} = {pts[0]}"] if len(pts) == 1 else \
            [f"{var} ∈ {{{', '.join(pts)}}}"]
    if isinstance(sol, sp.Interval):
        return [_format_interval(sol, var)]
    if isinstance(sol, sp.Union):
        parts: List[str] = []
        try:
            items = sorted(sol.args, key=lambda s: s.start)
        except (AttributeError, TypeError):
            items = list(sol.args)
        for s in items:
            parts.extend(_format_solution_set(s, var))
        return [" или ".join(parts)] if len(parts) > 1 else parts
    if isinstance(sol, sp.Complement):
        universe, exc = sol.args
        if isinstance(exc, sp.FiniteSet):
            pts = ", ".join(_bound_str(p) for p in exc.args)
            return [f"{var} ≠ {pts}"] if len(exc.args) == 1 else \
                [f"{var} ∉ {{{pts}}}"]
    # непонятное множество — показываем как есть
    return [sp.sstr(sol)]


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


# ---------------------------------------------------------------------------
# Точный «табличный» вид: 1/2, √3/2, (√6 - √2)/4, 2π (как S⇔D у Casio)
# ---------------------------------------------------------------------------

#: Максимальный размер «красивой» записи, иначе — десятичная дробь
_EXACT_MAX_LEN = 48


def _is_nice(node: sp.Basic) -> bool:
    """Выражение состоит только из целых, дробей, √(целое), π и e?"""
    if node.is_Rational:
        return abs(node.p) <= 10**12 and node.q <= 10**6
    if node is sp.pi or node is sp.E:
        return True
    if isinstance(node, sp.Pow):
        return (node.exp == sp.Rational(1, 2) and node.base.is_Integer
                and 2 <= node.base <= 10**6)
    if isinstance(node, (sp.Mul, sp.Add)):
        if len(node.args) > 8:
            return False
        return all(_is_nice(a) for a in node.args)
    return False


def _rational_str(r: sp.Rational) -> str:
    return str(r.p) if r.q == 1 else f"{r.p}/{r.q}"


def _split_mul(node: sp.Mul):
    """Mul -> (рациональный коэффициент, [нерациональные множители])."""
    coeff = sp.Rational(1)
    factors = []
    for f in node.args:
        if f.is_Rational:
            coeff *= f
        else:
            factors.append(f)
    return coeff, factors


def _combine(coeff: sp.Rational, body: str) -> str:
    """Коэффициент × тело: 1·√3 -> √3, -1·√2 -> -√2, 1/2·√3 -> √3/2."""
    if body == "":
        return _rational_str(coeff)
    if coeff == 1:
        return body
    if coeff == -1:
        return "-" + body
    p, q = coeff.p, coeff.q
    if q == 1:
        return f"{p}{body}"
    num = body if abs(p) == 1 else f"{abs(p)}{body}"
    return f"-{num}/{q}" if p < 0 else f"{num}/{q}"


def _join_terms(terms) -> str:
    """[(строка)] -> 'a + b - c' со знаком внутри каждого слагаемого."""
    out = ""
    for i, t in enumerate(terms):
        if i == 0:
            out = t
        elif t.startswith("-"):
            out += " - " + t[1:]
        else:
            out += " + " + t
    return out


def _pretty_add(node: sp.Add) -> str:
    """Сумма: по возможности общий знаменатель — (√6 - √2)/4, (1 + √5)/4."""
    terms = []  # (коэффициент, тело: '√2', 'π', '' для константы)
    for t in node.args:
        if t.is_Rational:
            if t != 0:
                terms.append((t, ""))
        elif isinstance(t, sp.Mul):
            c, factors = _split_mul(t)
            terms.append((c, "".join(_pretty_exact(f) for f in factors)))
        else:
            terms.append((sp.Rational(1), _pretty_exact(t)))
    terms.sort(key=lambda cb: (cb[0].p < 0, cb[1] == "",
                               tuple(-ord(ch) for ch in cb[1])))

    dens = {c.q for c, _ in terms if c != 0}
    if len(dens) == 1 and dens != {1}:
        q = dens.pop()
        inner_parts = []
        for c, body in terms:
            p = c.p
            if p == 0:
                continue
            if body == "":
                inner_parts.append(str(p))
            elif p == 1:
                inner_parts.append(body)
            elif p == -1:
                inner_parts.append("-" + body)
            else:
                inner_parts.append(f"{p}{body}")
        return f"({_join_terms(inner_parts)})/{q}"
    parts = []
    for c, body in terms:
        if body == "":
            if c != 0:
                parts.append(_rational_str(c))
        elif c != 0:
            parts.append(_combine(c, body))
    return _join_terms(parts)


def _pretty_exact(node: sp.Basic) -> str:
    if node is sp.pi:
        return "π"
    if node is sp.E:
        return "e"
    if node.is_Rational:
        return _rational_str(node)
    if isinstance(node, sp.Pow):  # √k (проверено в _is_nice)
        return f"√{node.base}"
    if isinstance(node, sp.Mul):
        coeff, factors = _split_mul(node)
        return _combine(coeff, "".join(_pretty_exact(f) for f in factors))
    if isinstance(node, sp.Add):
        return _pretty_add(node)
    return str(node)


def format_exact(node: sp.Basic) -> "str | None":
    """Точный вид выражения ('1/2', '√3/2', '(√6 - √2)/4', '2π') или None.

    None — если результат не выражается такой «табличной» записью
    (есть десятичные дроби, неизвестные переменные и т.п.).
    """
    if not isinstance(node, sp.Basic) or getattr(node, "free_symbols", None):
        return None
    if not _is_nice(node):
        return None
    try:
        text = _pretty_exact(node)
    except Exception:
        return None
    return text if 0 < len(text) <= _EXACT_MAX_LEN else None


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
        if INEQ_RE.search(text):
            return self._solve_ineq_and_format(text)
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
        # Точный вид (1/2, √3/2, 2π), если он существует; иначе десятичный
        return format_exact(node) or _fmt_number(value), value

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

    # --- неравенства -----------------------------------------------------------
    def solve_inequality_set(self, text: str, var: str = None) -> sp.Set:
        """Решить неравенство степени 1–4. Возвращает SymPy-множество.

        Примеры: ``x^2-4<0``, ``2x-1>=3``, ``x^4-5x^2+4<=0``.
        Правая часть может отсутствовать («x^2>1» -> «x^2>0»? нет:
        «x<0» означает «x < 0»). Знаки: <, >, <=, >=, ≤, ≥.
        """
        text = (text or "").strip()
        m = INEQ_RE.search(text)
        if not m:
            raise CalculatorError(
                "Не найден знак неравенства (поддерживаются <, >, <=, >=, ≤, ≥)")
        op_txt = {"≤": "<=", "≥": ">="}.get(m.group(), m.group())
        lhs, rhs = text[:m.start()].strip(), text[m.end():].strip()
        if not lhs or not rhs:
            raise CalculatorError(
                "Пустая часть неравенства (например: x^2-4<0, 2x-1>=3)")

        var = var or _guess_var(text)
        sym = sp.Symbol(var)
        ns = {var: sym, "Ans": self.ans, "sqrt": sp.sqrt, "exp": sp.exp,
              "abs": sp.Abs, "ln": sp.log, "log": self._logb}
        self._trig_into(ns)

        try:
            left = self._parse_side(lhs, ns)
            right = self._parse_side(rhs, ns)
        except CalculatorError as exc:
            raise CalculatorError(f"Ошибка ввода: {exc}") from exc
        f = sp.expand(left - right)
        # десятичные коэффициенты -> точные рациональные (0.5 -> 1/2)
        f = f.xreplace({a: sp.Rational(str(a)) for a in f.atoms(sp.Float)})

        others = f.free_symbols - {sym}
        if others:
            raise CalculatorError(
                "Неравенство решается по одной переменной ("
                + ", ".join(sorted(str(s) for s in others)) + ")")

        rel_ops = {"<": sp.Lt, "<=": sp.Le, ">": sp.Gt, ">=": sp.Ge}
        rel = rel_ops[op_txt](f, 0)

        if sym not in f.free_symbols:
            # «0 < 1» — верно всегда; «2 < 1» — никогда
            if rel:
                return sp.S.Reals
            return sp.S.EmptySet
        if f == 0:
            return sp.S.Reals if rel else sp.S.EmptySet

        try:
            deg = sp.Poly(f, sym).degree()
        except sp.PolynomialError:
            raise CalculatorError(
                "Поддерживаются неравенства с полиномом степени 1–4")
        if deg > 4:
            raise CalculatorError("Поддерживаются неравенства степени 1–4")

        try:
            return _poly_ineq_set(f, sym, op_txt)
        except sp.PolynomialError:
            raise CalculatorError(
                "Поддерживаются неравенства с полиномом степени 1–4")
        except Exception:
            raise CalculatorError("Не удалось решить неравенство")

    def _parse_side(self, side: str, ns: Dict[str, Any]) -> sp.Basic:
        """Разбор части неравенства: обычный ввод, при ошибке — LaTeX."""
        try:
            return self._parse(side, ns)
        except CalculatorError:
            from .latex_input import parse_input  # локальный импорт (циклы)
            return parse_input(side, self)

    def solve_inequality(self, text: str, var: str = None) -> List[str]:
        """Решить неравенство. Возвращает список частей решения:
        ['x < -2', 'x > 2'] или ['решений нет']."""
        sol = self.solve_inequality_set(text, var)
        return _format_solution_set(sol, var or _guess_var(text))

    def _solve_ineq_and_format(self, text: str) -> Result:
        parts = self.solve_inequality(text)
        return " или ".join(parts), None

    # --- служебное ---------------------------------------------------------------
    def _solve_and_format(self, text: str) -> Result:
        roots = self.solve_equation(text)
        self.ans = roots[0] if roots else 0.0
        pretty = ", ".join(_fmt_number(r) for r in roots)
        return f"x = {pretty}", (roots[0] if roots else 0.0)
