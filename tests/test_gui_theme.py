"""Тесты централизованной темы «Квантора» (quantor.gui.theme).

Проверяется: две палитры одинаковой структуры, применение к QApplication,
переключение из главного окна с перерендером формул и сохранением выбора,
рендер LaTeX в цвете темы.
"""

import json

import pytest

from quantor.gui import theme


@pytest.fixture()
def isolated_settings(tmp_path, monkeypatch):
    from quantor import settings
    monkeypatch.setattr(settings, "_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "_PATH", str(tmp_path / "settings.json"))


@pytest.fixture()
def restore_dark_theme(qapp):
    """Смена темы в тесте не протекает в остальные тесты."""
    yield
    theme.apply_app_theme(qapp, "dark")


CORE_KEYS = ("window", "panel", "surface", "border", "text", "text_dim",
             "accent", "accent_text", "accent_hover", "accent_press",
             "error", "error_bg", "error_border", "selection_bg",
             "placeholder")


def test_palettes_complete_and_distinct():
    for t in (theme.DARK, theme.LIGHT):
        missing = [k for k in CORE_KEYS if k not in t]
        assert not missing, f"не заданы ключи палитры: {missing}"
    assert theme.DARK["window"] != theme.LIGHT["window"]
    # акценты различаются: на светлом фоне нужен более тёмный бирюзовый
    assert theme.DARK["accent"] != theme.LIGHT["accent"]


def test_stylesheet_uses_palette(qapp):
    dark = theme.stylesheet(theme.DARK)
    light = theme.stylesheet(theme.LIGHT)
    assert theme.DARK["accent"] in dark        # палитра попадает в QSS
    assert theme.LIGHT["accent"] in light
    assert dark != light                         # один QSS, два словаря


def test_apply_app_theme(qapp, restore_dark_theme):
    t = theme.apply_app_theme(qapp, "light")
    assert t is theme.LIGHT
    assert theme.current() is theme.LIGHT
    pal = qapp.palette()
    assert (pal.color(pal.ColorRole.Window).name().upper()
            == theme.LIGHT["window"])
    t = theme.apply_app_theme(qapp, "dark")
    assert t is theme.DARK
    # неизвестное имя -> тёмная, а не падение
    assert theme.apply_app_theme(qapp, "no-such-theme") is theme.DARK


def test_view_menu_and_switch(qapp, isolated_settings, restore_dark_theme):
    from quantor.engine import CalculatorEngine
    from quantor.gui.main_window import MainWindow
    from quantor import settings

    w = MainWindow(CalculatorEngine("DEG"))
    assert set(w._theme_actions) == {"dark", "light"}
    assert w._theme_actions["dark"].isChecked()

    w._set_theme("light")
    assert theme.current() is theme.LIGHT
    assert not w._theme_actions["dark"].isChecked()
    assert w._theme_actions["light"].isChecked()
    saved = json.load(open(settings._PATH, encoding="utf-8"))
    assert saved["theme"] == "light"

    w._set_theme("dark")
    assert theme.current() is theme.DARK


def test_theme_renders_formula_in_theme_color(qapp):
    from quantor.latex_input import render_latex
    for t in (theme.DARK, theme.LIGHT):
        pix = render_latex(r"\frac{1}{2}", color=t["text"])
        assert pix is not None and not pix.isNull()


def test_initial_theme_name_from_settings(isolated_settings):
    from quantor import settings
    from quantor.gui import initial_theme_name

    assert initial_theme_name() == "dark"          # по умолчанию
    settings.save_settings({"theme": "light"})
    assert initial_theme_name() == "light"
    settings.save_settings({"theme": "neon"})
    assert initial_theme_name() == "dark"          # мусорное значение -> тёмная
