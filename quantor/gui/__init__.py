"""GUI «Квантора» на PySide6 (Qt 6)."""

from __future__ import annotations

from ..engine import CalculatorEngine


def apply_light_theme(app) -> None:
    """Фиксированная светлая тема: не зависим от системной палитры Qt.

    Без этого на системах с тёмной темой (KDE/GTK dark) весь текст
    наследуется белым и сливается со светлыми фонами кнопок.
    """
    from PySide6.QtGui import QColor, QPalette

    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor("#f4f4f2"))
    pal.setColor(QPalette.ColorRole.WindowText, QColor("#1a1a1a"))
    pal.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#f0f0ee"))
    pal.setColor(QPalette.ColorRole.Text, QColor("#1a1a1a"))
    pal.setColor(QPalette.ColorRole.Button, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor("#1a1a1a"))
    pal.setColor(QPalette.ColorRole.Highlight, QColor("#4c8bf5"))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor("#ffffe1"))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor("#1a1a1a"))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor("#8a8a8a"))
    pal.setColor(QPalette.ColorRole.Link, QColor("#2a6fdb"))
    # неактивные виджеты — приглушённо-серые, но читаемые
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText,
                 QPalette.ColorRole.WindowText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor("#9a9a9a"))
    app.setPalette(pal)


def run_app(engine: CalculatorEngine) -> int:
    """Создать QApplication, показать главное окно, войти в цикл событий."""
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication
    from .main_window import MainWindow, STYLESHEET

    app = QApplication(["Квантор"])
    app.setApplicationDisplayName("Квантор")
    app.setStyle("Fusion")
    apply_light_theme(app)  # до создания окна: тема не зависит от системы
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(STYLESHEET)

    window = MainWindow(engine)
    window.show()
    return app.exec()
