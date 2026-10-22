"""Неравенства степени 1–4: ядро и диалог «Квантора»."""

import pytest

from quantor.engine import CalculatorEngine, CalculatorError


@pytest.fixture()
def engine():
    return CalculatorEngine("DEG")


# ------------------------------------------------------------------------- ядро
@pytest.mark.parametrize("text,expected", [
    ("x^2-4<0", ["-2 < x < 2"]),
    ("x^2>4", ["x < -2 или x > 2"]),
    ("2x-1>3", ["x > 2"]),
    ("x^4-5x^2+4<=0", ["-2 ≤ x ≤ -1 или 1 ≤ x ≤ 2"]),
    ("x^2+x+1>0", ["x — любое действительное число"]),
    ("x^2<=0", ["x = 0"]),
    ("0.5x<2", ["x < 4"]),
    ("x<0", ["x < 0"]),
    ("(x-1)(x+2)>0", ["x < -2 или x > 1"]),
    ("x^2>=x^2", ["x — любое действительное число"]),
    ("0<1", ["x — любое действительное число"]),
    ("2<1", ["решений нет (∅)"]),
    ("x^2<2", ["-√2 < x < √2"]),
    ("x^3-6x^2+11x-6>0", ["1 < x < 2 или x > 3"]),
    ("-x^2+1>=0", ["-1 ≤ x ≤ 1"]),
    ("(x-1)^2<=0", ["x = 1"]),           # вырожденное решение — точка
    ("(x-1)^2>0", ["x < 1 или x > 1"]),
    ("-x^2>=0", ["x = 0"]),
    ("x^3>0", ["x > 0"]),
    ("x^4-3x^2+2<0", ["-√2 < x < -1 или 1 < x < √2"]),
    ("y^2-1<=0", ["-1 ≤ y ≤ 1"]),        # переменная не только x
    ("t+1>0", ["t > -1"]),
])
def test_solve_inequality(engine, text, expected):
    assert engine.solve_inequality(text) == expected


def test_quartic_numeric_bounds(engine):
    # Корни квартики не берутся в радикалах — числовые границы
    parts = engine.solve_inequality("x^4-x-1<0")
    assert parts and "<" in parts[0]
    lo, hi = [float(tok) for tok in
              parts[0].replace("< x <", "|").split("|")]
    assert abs(lo - (-0.7244919590)) < 1e-8
    assert abs(hi - 1.2207440848) < 1e-8


def test_unicode_signs(engine):
    assert engine.solve_inequality("x^2−4≤0") == ["-2 ≤ x ≤ 2"]
    assert engine.solve_inequality("x^2−4≥0") == ["x ≤ -2 или x ≥ 2"]


@pytest.mark.parametrize("text,error", [
    ("2^x>8", "полиномом"),          # не полином
    ("1/x<2", "полиномом"),          # не полином
    ("x^5+1<0", "степени 1–4"),      # степень слишком высока
    ("x^2+y<0", "одной переменной"),
    ("x<", "Пустая часть"),
    ("x^2-4<", "Пустая часть"),
    ("x^2-4=0", "знак неравенства"),
])
def test_inequality_errors(engine, text, error):
    with pytest.raises(CalculatorError) as exc:
        engine.solve_inequality(text)
    assert error in str(exc.value)


def test_inequality_in_equation_dialog_hint(engine):
    """Неравенство в окне уравнений -> понятная подсказка, а не TypeError."""
    with pytest.raises(CalculatorError) as exc:
        engine.solve_equation("x^2-4>0")
    assert "Неравенства" in str(exc.value)
    with pytest.raises(CalculatorError) as exc:
        engine.solve_equation("x^2-4>=0")
    assert "Неравенства" in str(exc.value)


def test_calculate_routes_inequality_and_keeps_ans(engine):
    engine.calculate("2*2")                 # ans = 4 (float 4.0)
    shown, value = engine.calculate("x^2-4<0")
    assert shown == "-2 < x < 2"
    assert value is None
    assert abs(engine.ans - 4) < 1e-12
    shown, value = engine.calculate("x^2>4")
    assert shown == "x < -2 или x > 2"


def test_latex_sides_fallback(engine):
    # LaTeX-операнды: parse латексом, знак — обычный текст
    parts = engine.solve_inequality(r"\frac{x}{2} - 1 <= 0")
    assert parts == ["x ≤ 2"]


def test_latex_inequality_signs(engine):
    r"""LaTeX-знаки неравенств: \leq, \geq (и \le/\leqslant)."""
    assert engine.solve_inequality(r"\frac{x - 1}{2} + 1 \leq 0") == ["x ≤ -1"]
    assert engine.solve_inequality(r"x^2 - 4 \geq 0") == ["x ≤ -2 или x ≥ 2"]
    assert engine.solve_inequality(r"x + 1 \leqslant 3") == ["x ≤ 2"]
    # LaTeX-знаки не бьют по \left при нормализации (\left не «\le …»)
    r = engine.solve_inequality(r"\frac{x}{2} - 1 \ge 0")
    assert r == ["x ≥ 2"]


# ------------------------------------------------------------------------ GUI
def test_inequality_dialog(qapp):
    from quantor.gui.dialogs import InequalityDialog

    engine = CalculatorEngine("DEG")
    inserted: list = []
    dlg = InequalityDialog(None, engine, inserted.append)

    dlg.solve()
    items = [dlg.list.item(i).text() for i in range(dlg.list.count())]
    assert items == ["-2 < x < 2"]

    dlg.ineq_edit.setText("x^2>4")
    dlg.solve()
    items = [dlg.list.item(i).text() for i in range(dlg.list.count())]
    assert items == ["x < -2 или x > 2"]

    dlg.ineq_edit.setText("x^5<0")
    dlg.solve()
    assert dlg.status.text() and "1–4" in dlg.status.text()

    # «Вставить в дисплей»: неравенство уходит в строку ввода
    dlg.ineq_edit.setText("x^2-4<=0")
    dlg._insert_ineq()
    assert inserted == ["x^2-4<=0"]


def test_main_window_inequality_line(qapp):
    from quantor.engine import CalculatorEngine
    from quantor.gui.main_window import MainWindow

    w = MainWindow(CalculatorEngine("DEG"))

    w.input.setText("x^2-4<0")
    w._evaluate()
    assert w.result.text() == "= -2 < x < 2"

    w.input.setText("x^2+x+1>0")
    w._evaluate()
    assert "любое" in w.result.text()

    # LaTeX с неравенством: знак обычным текстом, части — LaTeX
    w.input.setText(r"\frac{x}{2} - 1 <= 0")
    w._evaluate()
    assert "x ≤ 2" in w.result.text()
    assert w.result.property("err") is not True
