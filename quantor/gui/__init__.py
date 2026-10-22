"""GUI «Квантора» на PySide6 (Qt 6)."""

from __future__ import annotations

from ..engine import CalculatorEngine
from ..settings import load_settings
from .theme import DEFAULT_THEME, THEMES

__all__ = ["run_app", "initial_theme_name"]


def initial_theme_name() -> str:
    """Тема из настроек; тёмная — по умолчанию и при мусорном значении."""
    name = load_settings().get("theme")
    return name if name in THEMES else DEFAULT_THEME


def run_app(engine: CalculatorEngine) -> int:
    """Создать QApplication, применить тему, показать окно, войти в цикл."""
    from PySide6.QtWidgets import QApplication

    from . import theme
    from .main_window import MainWindow

    app = QApplication(["Квантор"])
    app.setApplicationDisplayName("Квантор")
    app.setStyle("Fusion")
    theme.apply_app_theme(app, initial_theme_name())  # до создания окна

    window = MainWindow(engine)
    window.show()
    return app.exec()
