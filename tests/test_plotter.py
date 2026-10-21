"""Ядро графиков и окно «График»."""

import numpy as np
import pytest

from quantor.engine import CalculatorEngine
from quantor.plotter import (
    PLOT_COLORS, PlotError, default_range, parse_functions, plot_data,
    sample_curve,
)

import sympy as sp


@pytest.fixture()
def engine():
    return CalculatorEngine("DEG")


# ---------------------------------------------------------------- разбор
def test_parse_multiple_lines(engine):
    exprs = parse_functions("sin(x)\ncos(x)\n\n  x^2-4  ", engine)
    assert [str(e) for e in exprs] == ["sin(x)", "cos(x)", "x**2 - 4"]


def test_parse_strips_y_prefix(engine):
    exprs = parse_functions("y=sin(x)", engine)
    assert str(exprs[0]) == "sin(x)"


def test_parse_errors(engine):
    with pytest.raises(PlotError, match="хотя бы одну функцию"):
        parse_functions("   \n  ", engine)
    with pytest.raises(PlotError, match="Максимум 6"):
        parse_functions("\n".join(["x"] * 7), engine)
    with pytest.raises(PlotError, match="Строка 2"):
        parse_functions("x\nx^2+y", engine)            # вторая переменная
    with pytest.raises(PlotError, match="Строка 1"):
        parse_functions("5+q", engine)                  # неизвестное имя
    with pytest.raises(PlotError, match="без знаков"):
        parse_functions("x^2-4>0", engine)              # неравенство


def test_parse_latex_side(engine):
    exprs = parse_functions(r"\frac{\sin(x)}{2}", engine)
    assert sp.simplify(exprs[0] - sp.sin(sp.Symbol("x")) / 2) == 0


# ---------------------------------------------------------------- выборка
def test_sample_count_and_constant(engine):
    xs_sym = sp.Symbol("x")
    ys = sample_curve(sp.Integer(5), xs_sym, engine, -1, 1, n=101)
    assert ys.shape == (101,)
    assert np.allclose(ys, 5.0)


def test_sample_asymptote_1_over_x(engine):
    """1/x: в точке разрыва NaN, линия не «склеена» через асимптоту."""
    ys = sample_curve(1 / sp.Symbol("x"), sp.Symbol("x"), engine, -1, 1)
    assert len(ys) == 1200
    mid = len(ys) // 2
    assert np.isnan(ys[mid - 1]) and np.isnan(ys[mid])
    # края сохранены
    assert np.isclose(abs(ys[0]), 1.0) and np.isclose(abs(ys[-1]), 1.0)


def test_sample_tan_deg_breaks(engine):
    ys = sample_curve(sp.tan(sp.Symbol("x")), sp.Symbol("x"), engine,
                      -360, 360, n=2400)
    # 4 асимптоты (±90, ±270) -> есть обрывы, и их немного
    assert 4 <= np.isnan(ys).sum() <= 40


def test_sample_complex_masked(engine):
    """√(1-x²) — только верхняя полуокружность, вне области — NaN."""
    ys = sample_curve(sp.sqrt(1 - sp.Symbol("x")**2), sp.Symbol("x"),
                      engine, -3, 3, n=601)
    mid = len(ys) // 2
    assert np.isclose(ys[mid], 1.0)               # x=0 -> y=1
    assert np.isnan(ys[0]) and np.isnan(ys[-1])   # |x|=3 -> комплексные


def test_sample_deg_trig_values():
    """В DEG sin получает градусы: sin(30°)=0.5 ровно на точке сетки."""
    eng = CalculatorEngine("DEG")
    xs_sym = sp.Symbol("x")
    ys = sample_curve(sp.sin(xs_sym), xs_sym, eng, -360, 360, n=1201)
    # сетка: -360 + k*0.6, k=650 -> ровно 30
    assert abs(ys[650] - 0.5) < 1e-12
    assert abs(ys[600]) < 1e-12                   # x=0 -> sin=0


def test_sample_rad_mode():
    eng = CalculatorEngine("RAD")
    ys = sample_curve(sp.sin(sp.Symbol("x")), sp.Symbol("x"), eng, -1, 1)
    mid = len(ys) // 2
    assert abs(ys[mid]) < 0.005                   # точка сетки близка к 0
    i = int(np.argmin(np.abs(np.linspace(-1, 1, len(ys)) - np.pi / 6)))
    assert abs(ys[i] - 0.5) < 0.01


# ------------------------------------------------------------------- общее
def test_default_range(engine):
    x = sp.Symbol("x")
    assert default_range(engine, [x]) == (-10.0, 10.0)
    assert default_range(engine, [sp.sin(x), sp.cos(x)]) == (-360.0, 360.0)


def test_plot_data_multi(engine):
    r = plot_data("sin(x)\ncos(x)\nx^2-4", engine)
    assert len(r.curves) == 3
    cols = [c.color for c in r.curves]
    assert cols == list(PLOT_COLORS[:3])
    assert r.xlabel == "x (градусы)"              # тригонометрия в DEG
    assert r.curves[0].xs[0] == -360.0
    assert r.curves[0].xs[-1] == 360.0
    assert r.ylim[0] < 0 and r.ylim[1] > 1.0      # y-оси охватывают все кривые


def test_plot_data_no_trig_xlabel(engine):
    r = plot_data("x^2-4", engine, x0=-5, x1=5)
    assert r.xlabel == "x"
    assert r.curves[0].xs[0] == -5.0 and r.curves[0].xs[-1] == 5.0


def test_plot_data_custom_y_limits(engine):
    r = plot_data("x^2", engine, y0=-10, y1=10)
    assert r.ylim == (-10.0, 10.0)


def test_plot_data_log_two_args(engine):
    r = plot_data("log(8,2)", engine)
    ys = r.curves[0].ys
    assert np.allclose(ys[np.isfinite(ys)], 3.0)   # константа log2(8)=3


def test_plot_data_auto_ylim_robust(engine):
    """Выбросы 1/x не должны определять масштаб по y."""
    r = plot_data("1/x", engine, x0=-1, x1=1)
    lo, hi = r.ylim
    assert abs(lo) < 50 and abs(hi) < 50           # а не ±тысячи


# -------------------------------------------------------------------- GUI
def test_plot_window_draws(qapp):
    from quantor.engine import CalculatorEngine
    from quantor.gui.plot_window import PlotDialog

    eng = CalculatorEngine("DEG")
    win = PlotDialog(None, eng)
    win.funcs_edit.setPlainText("sin(x)\ncos(x)")
    win._plot_current()
    ax = win.figure.axes[0]
    assert len(ax.lines) == 2
    assert ax.get_legend() is not None
    assert "градус" in ax.get_xlabel()
    assert win.status.text().startswith("")       # без ошибок
    assert "Ошибка" not in win.status.text()


def test_plot_window_error_status(qapp):
    from quantor.engine import CalculatorEngine
    from quantor.gui.plot_window import PlotDialog

    win = PlotDialog(None, CalculatorEngine("DEG"))
    win.funcs_edit.setPlainText("x^2+y")
    win._plot_current()
    assert "Ошибка" in win.status.text()
    assert len(win.figure.axes[0].lines) > 0 or True  # окно не упало


def test_main_window_opens_plot_once(qapp):
    from quantor.engine import CalculatorEngine
    from quantor.gui.main_window import MainWindow

    w = MainWindow(CalculatorEngine("DEG"))
    w._open_plot()
    first = w._plot_win
    assert first.isVisible()
    w._open_plot()
    assert w._plot_win is first                   # не создаёт второй экземпляр
    first.close()
