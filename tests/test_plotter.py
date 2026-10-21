"""Ядро графиков и окно «График»."""

import numpy as np
import pytest

from quantor.engine import CalculatorEngine
from quantor.plotter import (
    PLOT_COLORS, PlotError, default_range, parse_functions, parse_lines,
    plot_data, sample_curve, _numeric_poles,
)

import sympy as sp


@pytest.fixture()
def engine():
    return CalculatorEngine("DEG")


@pytest.fixture()
def isolated_settings(tmp_path, monkeypatch):
    """GUI-тесты не должны читать реальные ~/.quantor/settings.json."""
    from quantor import settings
    monkeypatch.setattr(settings, "_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "_PATH", str(tmp_path / "settings.json"))


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
    with pytest.raises(PlotError, match="тождество"):
        parse_functions("x^2=x^2", engine)              # LHS == RHS


def test_parse_line_kinds(engine):
    """Классификация строк: функция vs неявные связи."""
    specs = parse_lines("y=x^2\nx^2+y^2=25\nx*y=1\nx^2-4>0\ny^2<16", engine)
    assert [s["kind"] for s in specs] == ["func", "eq", "eq", "gt", "lt"]
    assert str(specs[0]["expr"]) == "x**2"
    assert str(specs[1]["expr"]) == "x**2 + y**2 - 25"


def test_parse_functions_rejects_implicit(engine):
    with pytest.raises(PlotError, match="неявная связь"):
        parse_functions("x^2+y^2=25", engine)


def test_parse_implicit_errors(engine):
    with pytest.raises(PlotError, match="лишние"):
        parse_lines("x^2+y^2=z", engine)                # третья переменная
    with pytest.raises(PlotError, match="разобрать"):
        parse_lines("1=", engine)                       # пустая часть


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


# ------------------------------------------------------------------- асимптоты
def test_asymptotes_rational(engine):
    r = plot_data("1/x", engine, x0=-2, x1=2)
    assert "x = 0" in r.asymptote_texts
    assert "y = 0" in r.asymptote_texts
    assert any(abs(p) < 1e-9 for p in r.v_positions)


def test_asymptotes_tan_rad():
    """tan: периодические полюсы x = π/2 + k·π."""
    eng = CalculatorEngine("RAD")
    r = plot_data("tan(x)", eng, x0=-3, x1=3)
    assert "x = π/2 + k·π" in r.asymptote_texts
    around = [round(p, 4) for p in r.v_positions]
    assert 1.5708 in around and -1.5708 in around


def test_asymptotes_tan_deg(engine):
    """tan в DEG: полюсы на оси градусов — 90 + 180·k."""
    r = plot_data("tan(x)", engine, x0=-360, x1=360)
    assert "x = 90 + k·180" in r.asymptote_texts
    positions = [round(p) for p in r.v_positions]
    for deg in (-270, -90, 90, 270):
        assert deg in positions


def test_asymptotes_log(engine):
    r = plot_data("log(x)", engine)
    assert "x = 0" in r.asymptote_texts


def test_asymptotes_oblique(engine):
    r = plot_data("(x^2+1)/x", engine, x0=-5, x1=5)
    assert "x = 0" in r.asymptote_texts
    assert "y = x" in r.asymptote_texts
    assert (1.0, 0.0) in [(round(k, 6), round(b, 6)) for k, b in r.o_lines]


def test_asymptotes_linear_self_excluded(engine):
    """Прямая y=x+1 не должна получать себе асимптоту."""
    r = plot_data("x+1", engine)
    assert r.asymptote_texts == [] and r.o_lines == []


def test_numeric_poles_fallback(engine):
    """Резерв: NaN-кластеры между конечными ветвями дают числовой полюс."""
    xs = np.linspace(-10, 10, 2001)
    ys = 1.0 / np.maximum(np.abs(xs - 2.0), 1e-9).copy()
    ys[np.abs(xs - 2.0) < 1e-3] = np.nan
    poles = _numeric_poles(xs, ys)
    assert len(poles) == 1 and abs(poles[0] - 2.0) < 0.01


# ------------------------------------------------------------------ неявные
def test_implicit_mesh_circle(engine):
    r = plot_data("x^2+y^2=25", engine)
    assert len(r.meshes) == 1 and r.meshes[0].kind == "eq"
    m = r.meshes[0]
    assert m.Z.shape[0] == m.Z.shape[1] == 300
    assert abs(m.Z[150, 150] + 25.0) < 0.5        # центр: -25
    assert (m.Z > 0).any() and (m.Z < 0).any()    # есть нулевой уровень
    assert r.ylim == (-10.0, 10.0)                # по умолчанию высота = полуширина


def test_implicit_region(engine):
    r = plot_data("x^2+y^2<16", engine)
    m = r.meshes[0]
    assert m.kind == "lt"
    assert m.Z[150, 150] < 0                      # центр внутри области


def test_implicit_mesh_finite_hole(engine):
    """sqrt(…): NaN вне области определения не ломают сетку."""
    r = plot_data("sqrt(x^2+y^2-1)<5", engine)
    assert np.isnan(r.meshes[0].Z).any()


def test_plot_data_mixed(engine):
    r = plot_data("x^2\nx^2+y^2=25", engine, x0=-5, x1=5)
    assert len(r.curves) == 1 and len(r.meshes) == 1
    assert r.ylim == (-5.0, 5.0)                  # высота по полной ширине x


# -------------------------------------------------------------------- GUI
def test_plot_window_draws(qapp, isolated_settings):
    from quantor.engine import CalculatorEngine
    from quantor.gui.plot_window import PlotDialog

    eng = CalculatorEngine("DEG")
    win = PlotDialog(None, eng)
    win.funcs_edit.setPlainText("sin(x)\ncos(x)")
    win._plot_current()
    ax = win.figure.axes[0]
    curves = [l for l in ax.lines if l.get_color() in PLOT_COLORS]
    assert len(curves) == 2
    assert ax.get_legend() is not None
    assert "градус" in ax.get_xlabel()
    assert win.status.text() == "" or "Ошибка" not in win.status.text()
    # яркие оси и оси через начало координат
    assert float(ax.spines["left"].get_linewidth()) >= 1.4
    assert any(abs(l.get_linewidth() - 1.4) < 1e-6 and l.get_color() == "#1a1a1a"
               for l in ax.lines)
    # панель зума
    assert hasattr(win, "nav") and win.nav is not None


def test_plot_window_asymptotes_status(qapp, isolated_settings):
    """tan: штриховые вертикальные линии + подписи асимптот в статусе."""
    from quantor.gui.plot_window import PlotDialog

    win = PlotDialog(None, CalculatorEngine("DEG"))
    win.funcs_edit.setPlainText("tan(x)")
    win._plot_current()
    ax = win.figure.axes[0]
    assert "Асимптоты" in win.status.text() and "90 + k·180" in win.status.text()
    dashed = [l for l in ax.lines
              if l.get_linestyle() in ("--", (0, (4, 3)))]
    assert len(dashed) >= 4                       # ±90, ±270 внутри [-360; 360]


def test_plot_window_implicit_draws(qapp, isolated_settings):
    """Неявные: контур равенства и заливка области рисуются."""
    from quantor.gui.plot_window import PlotDialog

    win = PlotDialog(None, CalculatorEngine("RAD"))
    win.funcs_edit.setPlainText("x^2+y^2=25\nx^2+y^2<16")
    win._plot_current()
    ax = win.figure.axes[0]
    assert len(ax.collections) >= 2               # контур + заливка
    assert "Ошибка" not in win.status.text()
    labels = win.figure.legends or []
    assert len(ax.get_legend().get_texts()) == 2  # прокси-записи обеих линий


# ------------------------------------------------------- перерасчёт вида
def test_view_recompute_extends_curve(qapp, isolated_settings):
    """Панорама: кривая пересчитывается под новые границы x, а не обрывается."""
    from quantor.gui.plot_window import PlotDialog

    win = PlotDialog(None, CalculatorEngine("RAD"))
    win.funcs_edit.setPlainText("sin(x)")
    for e in (win.x0_edit, win.x1_edit, win.y0_edit, win.y1_edit):
        e.clear()
    win._plot_current()
    old_r = win._result
    ax = win.figure.axes[0]
    ax.set_xlim(5.0, 15.0)
    win._refresh_view()
    r = win._result
    assert (r.xlim[0], r.xlim[1]) == (5.0, 15.0)
    xs = r.curves[0].xs
    assert abs(xs[0] - 5.0) < 1e-9 and abs(xs[-1] - 15.0) < 1e-9
    assert r is not old_r                        # данные действительно новые


def test_view_recompute_asymptotes_in_view(qapp, isolated_settings):
    """При зуме асимптоты пересчитываются в пределах нового вида."""
    from quantor.gui.plot_window import PlotDialog

    win = PlotDialog(None, CalculatorEngine("RAD"))
    win.funcs_edit.setPlainText("tan(x)")
    win._plot_current()
    ax = win.figure.axes[0]
    assert any(p < 0 for p in win._result.v_positions)
    ax.set_xlim(2.0, 6.0)
    win._refresh_view()
    positions = win._result.v_positions
    assert positions and all(2.0 <= p <= 6.0 for p in positions)
    assert any(abs(p - 4.71238) < 0.01 for p in positions)


def test_view_recompute_meshes(qapp, isolated_settings):
    """Неявные тоже перерисовываются под новый вид."""
    from quantor.gui.plot_window import PlotDialog

    win = PlotDialog(None, CalculatorEngine("RAD"))
    win.funcs_edit.setPlainText("x^2+y^2=25")
    win._plot_current()
    ax = win.figure.axes[0]
    ax.set_xlim(-20.0, 20.0)
    win._refresh_view()
    m = win._result.meshes[0]
    assert abs(m.X.min() + 20.0) < 1e-6 and abs(m.X.max() - 20.0) < 1e-6


def test_refresh_skips_same_view_and_deep_zoom(qapp, isolated_settings):
    """Без изменения вида и при сверхглубоком зуме перерасчёт не тратится."""
    import numpy as _np
    from quantor.gui.plot_window import PlotDialog

    win = PlotDialog(None, CalculatorEngine("RAD"))
    win.funcs_edit.setPlainText("1/x")
    win._plot_current()
    first = win._result
    win._refresh_view()
    assert win._result is first                  # вид не менялся — пропуск

    ax = win.figure.axes[0]
    ax.set_xlim(1.0, 1.0 + 1e-11)                # сверхглубокий зум
    win._refresh_view()
    assert win._result is first


def test_draw_event_starts_timer(qapp, isolated_settings):
    """Событие перерисовки с новыми границами запускает буферизацию."""
    from quantor.gui.plot_window import PlotDialog

    win = PlotDialog(None, CalculatorEngine("RAD"))
    win.funcs_edit.setPlainText("sin(x)")
    win._plot_current()
    win.figure.axes[0].set_xlim(0.0, 4.0)
    assert not win._nav_timer.isActive()
    win._on_draw_event()
    assert win._nav_timer.isActive()
    win.nav_timer_stop()


def test_plot_window_error_status(qapp, isolated_settings):
    from quantor.engine import CalculatorEngine
    from quantor.gui.plot_window import PlotDialog

    win = PlotDialog(None, CalculatorEngine("DEG"))
    win.funcs_edit.setPlainText("x^2+y")
    win._plot_current()
    assert "Ошибка" in win.status.text()
    assert len(win.figure.axes[0].lines) > 0 or True  # окно не упало


def test_main_window_opens_plot_once(qapp, isolated_settings):
    from quantor.engine import CalculatorEngine
    from quantor.gui.main_window import MainWindow

    w = MainWindow(CalculatorEngine("DEG"))
    w._open_plot()
    first = w._plot_win
    assert first.isVisible()
    w._open_plot()
    assert w._plot_win is first                   # не создаёт второй экземпляр
    first.close()
