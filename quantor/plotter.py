"""Ядро построения графиков «Квантора» (без GUI).

Скорость: одна компиляция sympy-выражения в числовую функцию
(sympy.lambdify + numpy) на функцию, затем векторный вызов на массиве
точек. Особые точки (асимптоты, комплексные значения) маскируются NaN,
чтобы matplotlib не «склеивал» линии через разрывы.

Режим DEG/RAD: в DEG тригонометрия получает/возвращает градусы
(синус от градусов, арксинус в градусах) — как на дисплее калькулятора.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp
from sympy.core.relational import Relational

from .engine import CalculatorEngine, CalculatorError

__all__ = [
    "PlotError", "PlotCurve", "PlotResult", "MAX_FUNCTIONS",
    "parse_functions", "plot_data", "default_range",
]

#: Максимум функций на одном графике
MAX_FUNCTIONS = 6

#: Число точек выборки по x (достаточно для гладких кривых)
SAMPLES = 1200

#: Палитра линий (со светло-плаtnой темой калькулятора)
PLOT_COLORS = ("#1a5fb4", "#c01c28", "#26a269", "#e66100", "#813d9c", "#986a00")

#: Радианы "разумного" масштаба: значения больше этого считаем выбросом асимптоты
_Y_HARD_LIMIT = 1e8


class PlotError(Exception):
    """Ошибка построения графика — показывается пользователю."""


@dataclass
class PlotCurve:
    """Одна кривая: выборка по x и y с NaN-масками в особых точках."""
    label: str
    xs: np.ndarray
    ys: np.ndarray
    color: str


@dataclass
class PlotResult:
    """Данные для отрисовки: список кривых, подписи осей, границы по y."""
    curves: List[PlotCurve] = field(default_factory=list)
    xlabel: str = "x"
    ylabel: str = "y"
    #: Границы оси y (авто по перцентилям или заданные)
    ylim: Tuple[float, float] = (-1.0, 1.0)
    #: Сколько точек скрыто особыми точками (для статуса в окне)
    breaks: int = 0


# --------------------------------------------------------------------------- разбор

def parse_functions(text: str, engine: CalculatorEngine, var: str = "x",
                    limit: int = MAX_FUNCTIONS) -> List[sp.Expr]:
    """Список функций: одна функция на строку, пустые строки пропускаются."""
    lines = [ln.strip().replace("y=", "").replace("Y=", "")
             for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        raise PlotError("Введите хотя бы одну функцию")
    if len(lines) > limit:
        raise PlotError(f"Максимум {limit} функций на графике (сейчас {len(lines)})")

    sym = sp.Symbol(var)
    # Тригонометрия «сырая» (sp.sin), конверсия углов — в _deg_table при
    # вычислении: иначе DEG-обёртка движка вшивает pi*x/180 в выражение,
    # и градусная таблица lambdify сработала бы дважды.
    ns: Dict[str, object] = {var: sym, "Ans": engine.ans,
                             "sqrt": sp.sqrt, "exp": sp.exp, "abs": sp.Abs,
                             "ln": sp.log, "log": engine._logb,
                             "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                             "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
                             "arcsin": sp.asin, "arccos": sp.acos,
                             "arctan": sp.atan,
                             "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
                             "pi": sp.pi, "e": sp.E}

    out: List[sp.Expr] = []
    for i, line in enumerate(lines, 1):
        try:
            expr = engine._parse_side(line, ns)
        except CalculatorError as exc:
            raise PlotError(f"Строка {i}: {exc}") from exc
        if isinstance(expr, Relational):
            raise PlotError(
                f"Строка {i}: введите функцию без знаков <, >, =, например x^2-4")
        extra = expr.free_symbols - {sym}
        if extra:
            raise PlotError(
                f"Строка {i}: функция должна зависеть только от «{var}» "
                + f"(задействовано ещё: {', '.join(sorted(str(s) for s in extra))})")
        out.append(sp.simplify(expr) if expr.free_symbols else expr)
    return out


def _has_trig(exprs: Sequence[sp.Basic]) -> bool:
    """Есть ли среди функций обычный тригонометрия (не гиперболический)."""
    _trig = (sp.sin, sp.cos, sp.tan, sp.asin, sp.acos, sp.atan)
    for e in exprs:
        if any(a.func in _trig for a in e.atoms(sp.Function)):
            return True
    return False


def default_range(engine: CalculatorEngine, exprs: Sequence[sp.Basic]
                  ) -> Tuple[float, float]:
    """Диапазон по умолчанию: [-10; 10], тригонометрия в DEG -> [-360; 360]."""
    if engine.angle_mode == "DEG" and _has_trig(exprs):
        return (-360.0, 360.0)
    return (-10.0, 10.0)


# ------------------------------------------------------------------- вычисление

def _deg_table() -> Dict[str, Callable]:
    """Lambdify-таблица тригонометрии в градусах (поверх numpy)."""
    d2r, r2d = np.pi / 180.0, 180.0 / np.pi
    return {
        "sin": lambda a: np.sin(d2r * a),
        "cos": lambda a: np.cos(d2r * a),
        "tan": lambda a: np.tan(d2r * a),
        "asin": lambda a: r2d * np.arcsin(a),
        "acos": lambda a: r2d * np.arccos(a),
        "atan": lambda a: r2d * np.arctan(a),
        # log(x, b) калькулятора — по произвольному основанию; numpy.log ждёт 1 arg
        "log": lambda a, b=None: np.log(a) if b is None else np.log(a) / np.log(b),
    }


def _lambdify(expr: sp.Expr, sym: sp.Symbol, engine: CalculatorEngine
             ) -> Callable[[np.ndarray], np.ndarray]:
    """Скомпилировать символическое выражение в векторную функцию."""
    funcs = _deg_table() if engine.angle_mode == "DEG" else {
        "log": lambda a, b=None: np.log(a) if b is None else np.log(a) / np.log(b),
    }
    if sym not in expr.free_symbols:
        val = complex(sp.N(expr))
        if abs(val.imag) < 1e-12:        # константа: горизонтальная линия
            c = val.real
            return lambda xs: np.full_like(np.asarray(xs, dtype=float), c)
    return sp.lambdify(sym, expr, modules=[funcs, "numpy"])


def _break_mask(ys: np.ndarray) -> np.ndarray:
    """True там, где кривую надо оборвать (скачок знака с большой величиной).

    Классический случай «1/x» и «tan(x)»: слева от точки значения уходят
    в -inf, справа в +inf — соединительная линия выглядела бы стеной через
    весь график. Малые по модулю пересечения нуля этим не задеваются.
    """
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
    """Выборка y(x) по равномерной сетке [x0..x1]; NaN в особых точках."""
    fun = _lambdify(expr, sym, engine)
    if x1 <= x0:
        x0, x1 = x1, x0
    xs = np.linspace(float(x0), float(x1), int(n))
    with np.errstate(all="ignore"):
        ys = fun(xs)
    ys = np.asarray(ys)
    if ys.shape != xs.shape:
        ys = np.broadcast_to(ys, xs.shape)
    if np.iscomplexobj(ys):              # мнимое — вне области определения
        big = np.abs(np.imag(ys)) > 1e-9 * np.maximum(1.0, np.abs(np.real(ys)))
        ys = np.where(big, np.nan, np.real(ys))
    ys = ys.astype(float)
    ys = np.where(np.isfinite(ys) & (np.abs(ys) < _Y_HARD_LIMIT), ys, np.nan)
    if np.isnan(ys).all():
        return ys
    ys[_break_mask(ys)] = np.nan
    return ys


def _ylimits(ys_list: Sequence[np.ndarray], y0: Optional[float],
             y1: Optional[float]) -> Tuple[float, float]:
    """Границы по y: заданные пользователем или авто по перцентилям."""
    if y0 is not None and y1 is not None and y0 < y1:
        return (float(y0), float(y1))
    finite = np.concatenate([y[np.isfinite(y)] for y in ys_list]) \
        if ys_list else np.array([])
    if finite.size == 0:
        return (-1.0, 1.0)
    lo, hi = np.percentile(finite, [1.5, 98.5])
    span = hi - lo
    if span < 1e-9:
        span = max(0.5, abs(lo) * 0.1)
    return (lo - 0.08 * span, hi + 0.08 * span)


# ------------------------------------------------------------------------ сразу

def plot_data(text: str, engine: CalculatorEngine, var: str = "x",
              x0: Optional[float] = None, x1: Optional[float] = None,
              y0: Optional[float] = None, y1: Optional[float] = None,
              n: int = SAMPLES) -> PlotResult:
    """Полный цикл: разбор -> выборка -> данные для отрисовки."""
    exprs = parse_functions(text, engine, var)
    if x0 is None or x1 is None:
        x0, x1 = default_range(engine, exprs)

    sym = sp.Symbol(var)
    unit = "градусы" if engine.angle_mode == "DEG" else "радианы"
    result = PlotResult(xlabel=f"{var} ({unit})" if _has_trig(exprs) else var)
    for i, expr in enumerate(exprs):
        ys = sample_curve(expr, sym, engine, x0, x1, n)
        label = str(expr).replace("**", "^")[:32]
        result.curves.append(PlotCurve(label, np.linspace(x0, x1, n), ys,
                                       PLOT_COLORS[i % len(PLOT_COLORS)]))
    result.ylim = _ylimits([c.ys for c in result.curves], y0, y1)
    result.breaks = int(sum(int(np.isnan(c.ys).sum()) for c in result.curves))
    return result
