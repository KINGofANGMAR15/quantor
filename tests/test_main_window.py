"""Поведение главного окна: маршрутизация ввода, память, отображение."""

import pytest

from quantor.engine import CalculatorEngine
from quantor.gui.main_window import MainWindow


@pytest.fixture()
def window(qapp, isolated_settings):
    w = MainWindow(CalculatorEngine("DEG"))
    yield w
    w.close()


@pytest.fixture()
def isolated_settings(tmp_path, monkeypatch):
    """GUI-тесты не должны трогать реальные ~/.quantor/settings.json."""
    from quantor import settings
    monkeypatch.setattr(settings, "_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "_PATH", str(tmp_path / "settings.json"))


# ------------------------------------------------- \int-ввод вне шаблонов
def test_int_query_without_dx_is_friendly_error(window):
    r"""«\int x» (без dx): без «Внутренней ошибки».

    Раньше evaluate_query возвращал None и _show_calculus(None) падал
    с AttributeError. Теперь общий путь либо решает, либо даёт понятный
    отказ — parse_latex понимает \int и показывает символьный интеграл.
    """
    window.input.setText(r"\int x")
    window._evaluate()
    assert "Внутренняя ошибка" not in window.result.text()
    assert window.result.text().strip()


def test_int_limits_without_braces_compute(window):
    r"""\int_0^\pi — теперь считается (пределы-атомы), а не «Внутренняя ошибка»."""
    window.input.setText(r"\int_0^\pi \sin(x) dx")
    window._evaluate()
    assert "Внутренняя ошибка" not in window.result.text()
    assert window.result.property("err") is not True
    assert "точно: 2" in window.result.text()


def test_int_nested_brace_limits_compute(window):
    r"""\int_{\frac{\pi}{2}}^{\pi} sin(x) dx = 1 — вложенные скобки в пределе."""
    window.input.setText(r"\int_{\frac{\pi}{2}}^{\pi} \sin(x) dx")
    window._evaluate()
    assert window.result.property("err") is not True
    assert "точно: 1" in window.result.text()


# ------------------------------------------------------------- память
def test_mem_add_complex_shows_clear_error(window):
    """M+ при комплексном результате — понятный отказ, а не TypeError в слоте."""
    window.input.setText("i")
    window._evaluate()
    assert window.result.property("err") is not True
    window._mem_add()
    assert window.result.property("err") is True
    assert "Комплексное" in window.result.text()
    # память не изменилась и калькулятор жив
    window.input.clear()
    window.input.setText("2+2")
    window._evaluate()
    assert window.result.text() == "= 4"


def test_mem_add_real_still_works(window):
    window.input.setText("2+3")
    window._evaluate()
    window.input.clear()
    window._mem_add()          # _last_value = 5 при пустом вводе
    window._mem_recall()
    assert window.input.text() == "5"


# ------------------------------------------------------------- отображение
def test_inequality_prefix_no_duplicate_for_any_var(window):
    """Точечное решение по y: «y = 0», а не «= y = 0»."""
    window.input.setText("y^2<=0")
    window._evaluate()
    assert window.result.text() == "y = 0"
    window.input.setText("x^2<=0")
    window._evaluate()
    assert window.result.text() == "x = 0"
