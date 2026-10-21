"""Ядро построения графиков «Квантора» (без GUI).

Возможности:
- явные функции y = f(x): разбор одной функции на строку (обычный или
  LaTeX), компиляция через sympy.lambdify + numpy, векторная выборка;
- неявные связи (x^2+y^2=25, x*y=4, x^2+y^2<16): сетка x×y и знак
  выражения «левая − правая» (контур на нуле / заливка области);
- асимптоты: вертикальные (полюсы, в т.ч. периодические у tan/cot),
  горизонтальные и наклонные (через пределы при x -> ±inf).

Режим DEG/RAD: концепция «сырой» тригонометрии + таблица lambdify
(в DEG sin получает градусы, asin возвращает градусы).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp
from sympy.core.relational import Relational

from .engine import CalculatorEngine, CalculatorError, format_exact, _fmt_number, _exact_to_number
from .latex_input import LatexParseError

__all__ = [
    "PlotError", "PlotCurve", "ImplicitMesh", "Asymptotes", "PlotResult",
    "MAX_FUNCTIONS", "parse_lines", "parse_functions", "plot_data",
    "default_range", "sample_curve", "implicit_mesh", "analyze_asymptotes",
]

#: Максимум линий на одном графике
MAX_FUNCTIONS = 6

#: Число точек выборки по x (достаточно для гладких кривых)
SAMPLES = 1200

#: Палитра линий (под светлую тему калькулятора)
PLOT_COLORS = ("#1a5fb4", "#c01c28", "#26a269", "#e66100", "#813d9c", "#986a00")

#: Предел «разумного» значения: выше — выброс асимптоты
_Y_HARD_LIMIT = 1e8

#: Лимит числа прочерченных вертикальных асимптот периодической функции
_MAX_POLES_DRAWN = 60


class PlotError(Exception):
    """Ошибка построения графика — показывается пользователю."""


@dataclass
class PlotCurve:
    """Явная кривая: выборка по x и y с NaN-масками в особых точках."""
    label: str
    xs: np.ndarray
    ys: np.ndarray
    color: str


@dataclass
class ImplicitMesh:
    """Неявное множество: сетка значений E(x,y) = левая − правая."""
    label: str
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray            # NaN там, где не вещественно
    kind: str                # 'eq' | 'lt' | 'le' | 'gt' | 'ge'
    color: str


@dataclass
class Asymptotes:
    """Асимптоты одной явной функции: подписи + данные для рисования."""
    texts: List[str] = field(default_factory=list)         # 'x = π/2 + k·π'
    v_positions: List[float] = field(default_factory=list) # x-координаты полюсов
    h_values: List[float] = field(default_factory=list)    # y = const
    o_lines: List[Tuple[float, float]] = field(default_factory=list)  # y = k*x + b


@dataclass
class PlotResult:
    """Данные для отрисовки."""
    curves: List[PlotCurve] = field(default_factory=list)
    meshes: List[ImplicitMesh] = field(default_factory=list)
    xlabel: str = "x"
    ylabel: str = "y"
    xlim: Tuple[float, float] = (-10.0, 10.0)   # фактический диапазон выборки
    ylim: Tuple[float, float] = (-1.0, 1.0)
    #: Подписи асимптот (все функции, без повторов)
    asymptote_texts: List[str] = field(default_factory=list)
    #: Данные для рисования асимптот (объединение всех функций)
    v_positions: List[float] = field(default_factory=list)
    h_values: List[float] = field(default_factory=list)
    o_lines: List[Tuple[float, float]] = field(default_factory=list)
    #: Сколько точек скрыто особыми точками (для статуса)
    breaks: int = 0


# --------------------------------------------------------------------------- разбор

_REL_OPS = ("<=", ">=", "<", ">")


def _symbol_ns(engine: CalculatorEngine, var: str,
               extra_var: Optional[str] = None) -> Dict[str, object]:
    """Неймспейс с «сырой» тригонометрией: конверсия углов — в lambdify-таблице."""
    sym_x, sym_y = sp.Symbol(var), None
    if extra_var:
        sym_y = sp.Symbol(extra_var)
    ns: Dict[str, object] = {var: sym_x, "Ans": engine.ans,
                             "sqrt": sp.sqrt, "exp": sp.exp, "abs": sp.Abs,
                             "ln": sp.log, "log": engine._logb,
                             "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                             "cot": sp.cot,
                             "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
                             "arcsin": sp.asin, "arccos": sp.acos,
                             "arctan": sp.atan,
                             "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
                             "pi": sp.pi, "e": sp.E}
    if sym_y is not None:
        ns[extra_var] = sym_y
    return ns


def _parse_line(line: str, engine: CalculatorEngine, var: str,
                allow_second_var: bool) -> Tuple[sp.Basic, Optional[str]]:
    """Разобрать строку. Возвращает (выражение, оператор или None).

    Оператор: '=' или знак неравенства — значит строка неявная.
    """
    for op in (_REL_OPS + ("=",)):
        # знак внутри функций типа log(8,2) не бывает; ищем первый сверху
        idx = line.find(op)
        if idx > 0:
            lhs, rhs = line[:idx].strip(), line[idx + len(op):].strip()
            if not lhs or not rhs:
                break
            ns = _symbol_ns(engine, var, "y" if allow_second_var else None)
            left = engine._parse_side(lhs, ns)
            right = engine._parse_side(rhs, ns)
            return left - right, op
    # обычная функция: знаков сравнения быть не должно
    if any(ch in line for ch in "<>"):
        raise PlotError("Введите функцию без знаков <, >, например x^2-4")
    ns = _symbol_ns(engine, var)
    return engine._parse_side(line, ns), None


def parse_lines(text: str, engine: CalculatorEngine, var: str = "x",
                limit: int = MAX_FUNCTIONS) -> List[dict]:
    """Разобрать многострочный ввод -> список спецификаций линий.

    Спецификация: {'kind': 'func'|'eq'|'lt'|'le'|'gt'|'ge',
                   'label': str, 'expr': sympy.Basic}
    kind 'func' — явная y=f(var); остальные — неявные множества (x, y).
    """
    raw = [(re.sub(r"^\s*y\s*=", "", ln, flags=re.I).strip())
           for ln in (text or "").splitlines()]
    lines = [ln for ln in raw if ln]
    if not lines:
        raise PlotError("Введите хотя бы одну функцию")
    if len(lines) > limit:
        raise PlotError(f"Максимум {limit} линий на графике (сейчас {len(lines)})")

    kinds = {"=": "eq", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}
    sym_x, sym_y = sp.Symbol(var), sp.Symbol("y")
    out: List[dict] = []
    for i, line in enumerate(lines, 1):
        try:
            expr, op = _parse_line(line, engine, var,
                                   allow_second_var=op_has_y(line))
        except PlotError:
            raise
        except (CalculatorError, LatexParseError) as exc:
            raise PlotError(f"Строка {i}: не удалось разобрать выражение") from exc
        if op is None:
            extra = expr.free_symbols - {sym_x}
            if extra:
                # двух переменных без знака равенства быть не может
                raise PlotError(
                    f"Строка {i}: функция должна зависеть только от «{var}» "
                    + f"(задействовано ещё: {', '.join(sorted(str(s) for s in extra))})")
            if isinstance(expr, Relational):
                raise PlotError(
                    f"Строка {i}: введите функцию без знаков <, >, =, например x^2-4")
            out.append({"kind": "func", "label": _label(str(expr)), "expr": expr})
            continue
        # ----- неявная связь: переменные x, y -----
        allowed = {sym_x, sym_y}
        unknown = expr.free_symbols - allowed
        if unknown:
            raise PlotError(
                f"Строка {i}: неявные графики — по переменным {var} и y "
                f"(лишние: {', '.join(sorted(str(s) for s in unknown))})")
        if op == "=" and not expr.free_symbols:
            raise PlotError(f"Строка {i}: тождество (левая и правая части равны)")
        if expr == 0:
            raise PlotError(f"Строка {i}: тождество (левая и правая части равны)")
        out.append({"kind": kinds[op], "label": _label(line), "expr": expr})
    return out


def op_has_y(line: str) -> bool:
    """Есть ли в строке «y» как отдельный токен (грубая проверка до разбора)."""
    import re
    return bool(re.search(r"(?<![A-Za-z])y(?![A-Za-z])", line))


def _label(s: str) -> str:
    return s.replace("**", "^").replace("*", "·")[:32]


def parse_functions(text: str, engine: CalculatorEngine, var: str = "x",
                    limit: int = MAX_FUNCTIONS) -> List[sp.Expr]:
    """Только явные функции (совместимость со старым API)."""
    specs = parse_lines(text, engine, var, limit)
    funcs = []
    for s in specs:
        if s["kind"] != "func":
            raise PlotError(f"«{s['label']}» — неявная связь, а не функция y=f({var})")
        funcs.append(s["expr"])
    return funcs


def _has_trig(exprs: Sequence[sp.Basic]) -> bool:
    """Обычная (не гиперболическая) тригонометрия среди выражений."""
    _trig = (sp.sin, sp.cos, sp.tan, sp.cot, sp.asin, sp.acos, sp.atan)
    for e in exprs:
        if any(a.func in _trig for a in e.atoms(sp.Function)):
            return True
    return False


def default_range(engine: CalculatorEngine, exprs: Sequence[sp.Basic]
                  ) -> Tuple[float, float]:
    """[-10; 10]; тригонометрия в DEG -> [-360; 360]."""
    if engine.angle_mode == "DEG" and _has_trig(exprs):
        return (-360.0, 360.0)
    return (-10.0, 10.0)


# ------------------------------------------------------------------- вычисление

def _lambdify_table(engine: CalculatorEngine) -> Dict[str, Callable]:
    """Lambdify-таблица: градусы в DEG-режиме; log по произвольному основанию."""
    base = {
        "log": lambda a, b=None: np.log(a) if b is None else np.log(a) / np.log(b),
        "cot": lambda a: 1.0 / np.tan(a),
    }
    if engine.angle_mode != "DEG":
        return base
    d2r, r2d = np.pi / 180.0, 180.0 / np.pi
    tab = dict(base)
    tab.update({
        "sin": lambda a: np.sin(d2r * a),
        "cos": lambda a: np.cos(d2r * a),
        "tan": lambda a: np.tan(d2r * a),
        "cot": lambda a: 1.0 / np.tan(d2r * a),
        "asin": lambda a: r2d * np.arcsin(a),
        "acos": lambda a: r2d * np.arccos(a),
        "atan": lambda a: r2d * np.arctan(a),
    })
    return tab


def _lambdify(args: Tuple[sp.Symbol, ...], expr: sp.Basic,
              engine: CalculatorEngine) -> Callable:
    """Скомпилировать выражение в векторную функцию от сетки."""
    no_vars = all(s not in expr.free_symbols for s in args)
    if no_vars:
        val = complex(sp.N(expr))
        c = val.real if abs(val.imag) < 1e-12 else np.nan
        return lambda *xs: np.full_like(np.broadcast_arrays(
            *[np.asarray(a, dtype=float) for a in xs])[0], c)
    return sp.lambdify(args, expr, modules=[_lambdify_table(engine), "numpy"])


def _break_mask(ys: np.ndarray) -> np.ndarray:
    """True там, где кривую надо оборвать (скачок знака с большой величиной)."""
    fin = np.isfinite(ys)
    mask = np.zeros(len(ys), dtype=bool)
    if fin.sum() < 3:
        return mask
    with np.errstate(all="ignore"):
        med = float(np.nanmedian(np.abs(ys[fin])))
    thresh = max(1.0, 50.0 * med)
    idx = np.where(fin[:-1] & fin[1:])[0]
    for i in idx:
        a, b = ys[i], ys[i + 1]
        if (a > 0) != (b > 0) and (abs(a) > thresh or abs(b) > thresh):
            mask[i] = mask[i + 1] = True
    return mask


def sample_curve(expr: sp.Expr, sym: sp.Symbol, engine: CalculatorEngine,
                 x0: float, x1: float, n: int = SAMPLES) -> np.ndarray:
    """Выборка y(x); NaN в особых точках и вне области определения."""
    fun = _lambdify((sym,), expr, engine)
    if x1 <= x0:
        x0, x1 = x1, x0
    xs = np.linspace(float(x0), float(x1), int(n))
    with np.errstate(all="ignore"):
        ys = fun(xs)
    ys = np.asarray(ys)
    if ys.shape != xs.shape:
        ys = np.broadcast_to(ys, xs.shape)
    if np.iscomplexobj(ys):
        big = np.abs(np.imag(ys)) > 1e-9 * np.maximum(1.0, np.abs(np.real(ys)))
        ys = np.where(big, np.nan, np.real(ys))
    ys = ys.astype(float)
    ys = np.where(np.isfinite(ys) & (np.abs(ys) < _Y_HARD_LIMIT), ys, np.nan)
    if np.isnan(ys).all():
        return ys
    ys[_break_mask(ys)] = np.nan
    return ys


def implicit_mesh(expr: sp.Basic, sym_x: sp.Symbol, sym_y: sp.Symbol,
                  engine: CalculatorEngine, xr: Tuple[float, float],
                  yr: Tuple[float, float], n: int = 300
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Сетка E(x, y) на прямоугольнике (x0..x1) × (y0..y1); NaN вне ℝ."""
    fun = _lambdify((sym_x, sym_y), expr, engine)
    xs = np.linspace(float(xr[0]), float(xr[1]), n)
    ys = np.linspace(float(yr[0]), float(yr[1]), n)
    with np.errstate(all="ignore"):
        X, Y = np.meshgrid(xs, ys[::-1])
        Z = fun(X, Y)
    Z = np.asarray(Z)
    if Z.shape != X.shape:
        Z = np.broadcast_to(Z, X.shape)
    if np.iscomplexobj(Z):
        big = np.abs(np.imag(Z)) > 1e-9 * np.maximum(1.0, np.abs(np.real(Z)))
        Z = np.where(big, np.nan, np.real(Z))
    return X, Y, Z.astype(float)


def _ylimits(ys_list: Sequence[np.ndarray], y0: Optional[float],
             y1: Optional[float]) -> Tuple[float, float]:
    """Заданные границы или авто по перцентилям (выбросы асимптот не рулят)."""
    if y0 is not None and y1 is not None and y0 < y1:
        return (float(y0), float(y1))
    arrays = [y[np.isfinite(y)] for y in ys_list]
    finite = np.concatenate(arrays) if arrays else np.array([])
    if finite.size == 0:
        return (-1.0, 1.0)
    lo, hi = np.percentile(finite, [1.5, 98.5])
    span = hi - lo
    if span < 1e-9:
        span = max(0.5, abs(lo) * 0.1)
    return (lo - 0.08 * span, hi + 0.08 * span)


# --------------------------------------------------------------------- асимптоты

def _fmt_num(v) -> str:
    """Точный вид числа, если возможно; иначе компактный десятичный."""
    try:
        if isinstance(v, sp.Basic) and not v.free_symbols:
            s = format_exact(v)
            if s:
                return s
        return _fmt_number(_exact_to_number(v))
    except Exception:
        return str(v)


def _linear_arg(arg: sp.Expr, sym: sp.Symbol) -> Optional[Tuple[sp.Expr, sp.Expr]]:
    """arg = a + b·sym -> (a, b); None если нелинейно."""
    b = sp.diff(arg, sym)
    if b.is_number and b != 0:
        a = sp.simplify(arg - b * sym)
        return (a, b)
    return None


def _vertical_poles(expr: sp.Expr, sym: sp.Symbol) -> List[Tuple[sp.Basic, str]]:
    """Вертикальные асимптоты: [(точное значение, подпись)] без периодических."""
    out: List[Tuple[sp.Basic, str]] = []
    try:
        den = sp.denom(sp.cancel(sp.together(expr)))
    except Exception:
        den = sp.Integer(1)
    if den.has(sym):
        try:
            for v in sp.solve(sp.Eq(den, 0), sym):
                if v.is_real and v.is_finite:
                    if _is_pole(expr, sym, v):
                        out.append((sp.simplify(v), f"x = {_fmt_num(v)}"))
        except Exception:
            pass
    try:
        sset = sp.singularities(expr, sym)
        if isinstance(sset, sp.FiniteSet):
            for v in sset:
                if v.is_real and v.is_finite and _is_pole(expr, sym, v):
                    if all(v != o for o, _ in out):
                        out.append((sp.simplify(v), f"x = {_fmt_num(v)}"))
    except Exception:
        pass
    return out


def _is_pole(expr: sp.Expr, sym: sp.Symbol, v: sp.Basic) -> bool:
    """В точке v функция уходит в ±inf хотя бы с одной стороны?"""
    for direction in ("+", "-"):
        try:
            L = sp.limit(expr, sym, v, direction)
            if L in (sp.oo, -sp.oo):
                return True
        except Exception:
            continue
    return False


def _periodic_poles(expr: sp.Expr, sym: sp.Symbol, x0: float, x1: float
                    ) -> List[Tuple[sp.Expr, sp.Expr, List[float]]]:
    """tan/cot: полюсы как (фаза, шаг, список позиций в диапазоне)."""
    out: List[Tuple[sp.Expr, sp.Expr, List[float]]] = []
    phase_shift = {sp.tan: sp.pi / 2, sp.cot: sp.Integer(0)}
    for t in expr.atoms(sp.Function):
        if t.func not in phase_shift:
            continue
        lin = _linear_arg(t.args[0], sym)
        if lin is None:
            continue
        a, b = lin
        # a + b·x = A + π·k  <=>  cos(arg) = 0 (tan) или sin(arg) = 0 (cot)
        A = phase_shift[t.func]
        step = sp.pi / sp.Abs(b)
        phase = sp.simplify((A - a) / b)
        step_f = float(sp.N(step))
        phase_f = float(sp.N(phase))
        if step_f < 1e-15:
            continue
        # повторяющиеся tan/cot с той же фазой и шагом не дублируем
        key = math.fmod(phase_f, step_f)
        if any(abs(key - math.fmod(float(sp.N(ph)), float(sp.N(st)))) < 1e-12
               and abs(step_f - float(sp.N(st))) < 1e-12 for ph, st, _ in out):
            continue
        lo, hi = min(x0, x1), max(x0, x1)
        kmin = math.ceil((lo - phase_f) / step_f)
        kmax = math.floor((hi - phase_f) / step_f)
        positions = [float(phase_f + k * step_f)
                     for k in range(kmin, kmax + 1)][:_MAX_POLES_DRAWN]
        if positions:
            out.append((phase, step, positions))
    return out


def _numeric_poles(xs: np.ndarray, ys: np.ndarray) -> List[float]:
    """Кластеры NaN с конечными значениями по обе стороны -> вероятные полюсы."""
    fin = np.isfinite(ys)
    poles: List[float] = []
    i = 0
    n = len(ys)
    while i < n:
        if not fin[i]:
            j = i
            while j < n and not fin[j]:
                j += 1
            if i > 0 and j < n and fin[i - 1] and fin[j]:
                poles.append(float((xs[i - 1] + xs[j]) / 2))
            i = j
        else:
            i += 1
    return poles


def _deg_factor(expr: sp.Expr, deg: bool) -> sp.Basic:
    """В DEG режиме ось x в градусах: радианные фазы умножаются на 180/π."""
    if deg and _has_trig([expr]):
        return sp.Integer(180) / sp.pi
    return sp.Integer(1)


def analyze_asymptotes(expr: sp.Expr, sym: sp.Symbol, xs: np.ndarray,
                       ys: np.ndarray, deg: bool = False) -> Asymptotes:
    """Вертикальные/горизонтальные/наклонные асимптоты одной функции."""
    info = Asymptotes()
    x0, x1 = float(xs[0]), float(xs[-1])
    f = _deg_factor(expr, deg)

    # --- вертикальные ---
    for v, text in _vertical_poles(expr, sym):
        try:
            xf = float(sp.N(v))
        except (TypeError, ValueError):
            continue
        if x0 <= xf <= x1:
            info.texts.append(text)
            info.v_positions.append(xf)
    for phase, step, positions in _periodic_poles(expr, sym, x0 / f, x1 / f):
        if not positions:
            continue
        phase_txt = _fmt_num(phase * f)
        step_txt = _fmt_num(step * f)
        if phase_txt in ("0", "-0", "−0"):
            info.texts.append(f"x = k·{step_txt}")
        else:
            info.texts.append(f"x = {phase_txt} + k·{step_txt}")
        info.v_positions.extend(p * float(f) for p in positions)

    # числовой резерв: кластеры NaN внутри обзора
    span = x1 - x0
    for p in _numeric_poles(xs, ys):
        if abs(p) < span * 2e-6:
            p = 0.0
        if all(abs(p - v) > span / 400 for v in info.v_positions):
            info.v_positions.append(p)
            info.texts.append(f"x ≈ {_fmt_num(sp.Float(p, 12))}")

    # --- горизонтальные и наклонные (по обе стороны) ---
    seen_h: List[sp.Basic] = []
    seen_o: List[Tuple[float, float]] = []
    for direction in (sp.oo, -sp.oo):
        try:
            Lh = sp.limit(expr, sym, direction)
            if Lh.is_finite and Lh not in seen_h:
                seen_h.append(Lh)
                info.h_values.append(float(sp.N(Lh)))
                tag = "" if direction is sp.oo else " (при x → −∞)"
                info.texts.append(f"y = {_fmt_num(Lh)}{tag}")
            k = sp.limit(expr / sym, sym, direction)
            if k.is_finite and k != 0:
                # прямая y = k·x + b сама себе наклонная — не считаем асимптотой
                try:
                    pdeg = sp.degree(expr, sym)
                except Exception:
                    pdeg = None
                if pdeg is not None and pdeg <= 1:
                    continue
                b = sp.limit(expr - k * sym, sym, direction)
                if b.is_finite:
                    kf, bf = float(sp.N(k)), float(sp.N(b))
                    if not any(abs(kf - k0) < 1e-9 and abs(bf - b0) < 1e-9
                               for k0, b0 in seen_o):
                        seen_o.append((kf, bf))
                        info.o_lines.append((kf, bf))
                        note = "" if direction is sp.oo else " (x → −∞)"
                        info.texts.append(f"y = {_oblique_text(k, b)}{note}")
        except Exception:
            continue
    return info


def _oblique_text(k, b) -> str:
    """Текст 'y = x + 1', 'y = −x/2 − 3' и т.п."""
    kf = _fmt_num(k)
    if kf == "1":
        kpart = "x"
    elif kf == "-1":
        kpart = "−x"
    else:
        kpart = f"{kf}·x"
    bf = _fmt_num(b)
    if bf == "0":
        return kpart
    return kpart + (f" + {bf}" if not str(bf).startswith("-")
                    else f" − {str(bf)[1:]}")


# ------------------------------------------------------------------------ сразу

def plot_data(text: str, engine: CalculatorEngine, var: str = "x",
              x0: Optional[float] = None, x1: Optional[float] = None,
              y0: Optional[float] = None, y1: Optional[float] = None,
              n: int = SAMPLES, mesh_n: int = 300) -> PlotResult:
    """Полный цикл: разбор -> выборка/сетка -> данные для отрисовки."""
    specs = parse_lines(text, engine, var)
    sym = sp.Symbol(var)
    sym_x, sym_y = sp.Symbol(var), sp.Symbol("y")
    if x0 is None or x1 is None:
        fexprs = [s["expr"] for s in specs if s["kind"] == "func"]
        x0, x1 = default_range(engine, fexprs)
        if not fexprs:
            x0, x1 = -10.0, 10.0
    if x1 <= x0:
        x0, x1 = float(x1), float(x0)

    has_trig = _has_trig([s["expr"] for s in specs])
    unit = "градусы" if engine.angle_mode == "DEG" else "радианы"
    result = PlotResult(xlabel=f"{var} ({unit})" if has_trig else var)

    mesh_xlim = (float(x0), float(x1))
    mesh_ylim: Optional[Tuple[float, float]] = None
    if y0 is not None and y1 is not None and y0 < y1:
        mesh_ylim = (float(y0), float(y1))
    elif any(s["kind"] != "func" for s in specs):
        span = mesh_xlim[1] - mesh_xlim[0]
        mesh_ylim = (-span / 2.0, span / 2.0)      # по умолчанию — высота по x

    color_i = 0
    for s in specs:
        kind = s["kind"]
        if kind == "func":
            ys = sample_curve(s["expr"], sym, engine, x0, x1, n)
            xs = np.linspace(float(x0), float(x1), n)
            info = analyze_asymptotes(s["expr"], sym, xs, ys,
                                      deg=engine.angle_mode == "DEG")
            result.curves.append(PlotCurve(s["label"], xs, ys,
                                           PLOT_COLORS[color_i % len(PLOT_COLORS)]))
            for t in info.texts:
                if t not in result.asymptote_texts:
                    result.asymptote_texts.append(t)
            result.v_positions.extend(info.v_positions)
            result.h_values.extend(info.h_values)
            result.o_lines.extend(info.o_lines)
        else:
            X, Y, Z = implicit_mesh(s["expr"], sym_x, sym_y, engine,
                                    mesh_xlim, mesh_ylim or (-10.0, 10.0), mesh_n)
            result.meshes.append(ImplicitMesh(s["label"], X, Y, Z, kind,
                                              PLOT_COLORS[color_i % len(PLOT_COLORS)]))
        color_i += 1

    result.v_positions = result.v_positions[:200]
    result.xlim = (float(x0), float(x1))
    if y0 is not None and y1 is not None and y0 < y1:
        result.ylim = (float(y0), float(y1))
    elif result.meshes and mesh_ylim is not None:
        result.ylim = mesh_ylim
    else:
        result.ylim = _ylimits([c.ys for c in result.curves], y0, y1)
    result.breaks = int(sum(int(np.isnan(c.ys).sum()) for c in result.curves))
    return result
