"""Централизованная тема «Квантора»: палитры, шрифты, стили Qt и matplotlib.

Единственный источник цветов и отступов для всех окон (главное окно,
диалоги, график). Тёмная тема — основная, светлая — вторичная; обе
описаны таблицами одинаковой структуры, stylesheet Qt и настройки
matplotlib строятся из одного и того же словаря — копий стилей нет.
"""

from __future__ import annotations

from typing import Dict

from PySide6.QtGui import QColor, QPalette

# ---------------------------------------------------------------- шрифты
#: UI-шрифт: стек Windows первым, Linux-запасные после него.
FONT_UI = ["Segoe UI", "Noto Sans", "Cantarell", "Ubuntu", "DejaVu Sans"]
#: Моноширинный стек для ввода выражений (цифры и скобки выровнены).
FONT_MONO = ["Cascadia Code", "JetBrains Mono", "Consolas",
             "Noto Sans Mono", "DejaVu Sans Mono"]

#: Единая шкала радиусов и отступов (px)
RADIUS = 8
RADIUS_SMALL = 6
MARGIN = 12
SPACING = 9


# ---------------------------------------------------------------- палитры
DARK: Dict[str, str] = {
    "name": "dark",
    # поверхности
    "window": "#151A22",        # фон приложения
    "panel": "#1E2530",         # панели и карточки
    "surface": "#252E3A",       # обычные кнопки и поля
    "surface_hover": "#2C3644",
    "surface_press": "#333F4E",
    "border": "#354151",
    "border_hover": "#465464",
    # текст
    "text": "#E6EDF3",
    "text_dim": "#9AA8B8",
    "text_disabled": "#5C6875",
    "placeholder": "#66737F",
    # акцент (бирюзовый)
    "accent": "#55D6BE",
    "accent_hover": "#74E4CE",
    "accent_press": "#3FBFA9",
    "accent_text": "#0B241F",   # тёмный текст на акцентной заливке
    "accent_tint": "#20403C",   # бирюзовый, разведённый поверх panel
    "accent_border": "#2E5E55",
    # состояния
    "error": "#F97583",
    "error_hover": "#FF919E",
    "error_press": "#D95F6E",
    "error_bg": "#34242B",      # бледно-красная подложка
    "error_border": "#5C3742",
    "selection_bg": "#2E5E55",
    "selection_text": "#E6EDF3",
    "menu_sel": "#2A3542",
}

LIGHT: Dict[str, str] = {
    "name": "light",
    "window": "#F2F4F7",
    "panel": "#FFFFFF",
    "surface": "#FFFFFF",
    "surface_hover": "#E9EEF4",
    "surface_press": "#DDE4EC",
    "border": "#D4DAE2",
    "border_hover": "#BDC7D2",
    "text": "#1D242E",
    "text_dim": "#5F6C7A",
    "text_disabled": "#9AA4B0",
    "placeholder": "#98A2AE",
    # акцент затемнён: на светлом фоне бледный бирюзовый теряет контраст
    "accent": "#0E8F78",
    "accent_hover": "#0DA586",
    "accent_press": "#0B7A66",
    "accent_text": "#FFFFFF",
    "accent_tint": "#E2F4EF",
    "accent_border": "#9AD4C6",
    "error": "#C23652",
    "error_hover": "#D34964",
    "error_press": "#A82B45",
    "error_bg": "#FBE7EA",
    "error_border": "#F0BFC6",
    "selection_bg": "#CDEDE5",
    "selection_text": "#0F2B25",
    "menu_sel": "#E7EDF4",
}

#: Токены окна графика (matplotlib): вычисляются из палитры.
#: Кривые из quantor.plotter.PLOT_COLORS остаются originale — они читаемы
#: на обоих фонах и не belong к теме GUI.
# pylint: disable=line-too-long
THEMES: Dict[str, Dict[str, str]] = {"dark": DARK, "light": LIGHT}
DEFAULT_THEME = "dark"


# ---------------------------------------------------------------- QPalette
def app_palette(t: Dict[str, str]) -> QPalette:
    """QPalette, согласованная со словарём темы (для нативных элементов)."""
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(t["window"]))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(t["text"]))
    pal.setColor(QPalette.ColorRole.Base, QColor(t["panel"]))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(t["surface"]))
    pal.setColor(QPalette.ColorRole.Text, QColor(t["text"]))
    pal.setColor(QPalette.ColorRole.Button, QColor(t["surface"]))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(t["text"]))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(t["selection_bg"]))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(t["selection_text"]))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(t["panel"]))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(t["text"]))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(t["placeholder"]))
    pal.setColor(QPalette.ColorRole.Link, QColor(t["accent"]))
    dim = QColor(t["text_disabled"])
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText,
                 QPalette.ColorRole.WindowText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, dim)
    return pal


# ---------------------------------------------------------------- QSS
def _btn(name: str, bg: str, fg: str, border: str, hover: str, press: str,
         weight: str | None = None, disabled_fg: str | None = None) -> str:
    """Один согласованный вариант кнопки: база/наведение/нажатие/фокус."""
    weight_line = f"\n    font-weight: {weight};" if weight else ""
    disabled = disabled_fg or fg
    return f"""
QPushButton#{name} {{
    background: {bg}; color: {fg}; border: 1px solid {border};
    border-radius: {RADIUS_SMALL}px;{weight_line}
}}
QPushButton#{name}:hover {{ background: {hover}; }}
QPushButton#{name}:pressed {{ background: {press}; }}
QPushButton#{name}:focus {{ border: 1.5px solid {border}; }}
QPushButton#{name}:disabled {{ background: {bg}; color: {disabled};
    border-color: {border}; }}
"""


def stylesheet(t: Dict[str, str]) -> str:
    """Полный QSS приложения, построенный из словаря темы.

    Светлая и тёмная темы — тот же stylesheet с другим словарём:
    двух независимых копий стилей не существует.
    """
    font_ui = ",".join(FONT_UI) + ",sans-serif"
    font_mono = ",".join(FONT_MONO) + ",monospace"
    dis = t["text_disabled"]
    base = f"""
QMainWindow, QDialog {{ background: {t['window']}; }}
QWidget {{ color: {t['text']}; font-family: {font_ui}; }}

QMenuBar {{ background: {t['window']}; color: {t['text']}; border: none; }}
QMenuBar::item {{ padding: 4px 10px; border-radius: {RADIUS_SMALL}px; }}
QMenuBar::item:selected {{ background: {t['surface_hover']}; }}
QMenu {{ background: {t['panel']}; color: {t['text']};
         border: 1px solid {t['border']}; border-radius: {RADIUS}px;
         padding: 4px; }}
QMenu::item {{ padding: 5px 22px 5px 12px; border-radius: {RADIUS_SMALL}px; }}
QMenu::item:selected {{ background: {t['menu_sel']}; }}
QMenu::separator {{ height: 1px; background: {t['border']}; margin: 4px 8px; }}
QToolTip {{ background: {t['panel']}; color: {t['text']};
            border: 1px solid {t['border']}; border-radius: {RADIUS_SMALL}px;
            padding: 4px 8px; }}

/* --- панели и области формул ------------------------------------------- */
QFrame#panel {{ background: {t['panel']};
    border: 1px solid {t['border']}; border-radius: {RADIUS}px; }}
QFrame#panel:focus {{ border-color: {t['accent_border']}; }}
QLabel#preview {{ background: transparent; color: {t['text_dim']}; }}
QLabel#result {{ font-size: 21px; font-weight: 600; color: {t['text']}; }}
QLabel#result[err="true"] {{ color: {t['error']}; }}
QLabel#cap {{ color: {t['text_dim']}; font-size: 11px; }}
QLabel#status {{ color: {t['text_dim']}; }}
QLabel#status[err="true"] {{ color: {t['error']}; }}
QLabel#badge {{ color: {t['error']}; background: {t['error_bg']};
    border: 1px solid {t['error_border']}; border-radius: {RADIUS_SMALL}px;
    padding: 4px 10px; }}

/* --- поля ввода ---------------------------------------------------------- */
QLineEdit {{ background: {t['surface']}; color: {t['text']};
    border: 1px solid {t['border']}; border-radius: {RADIUS_SMALL}px;
    padding: 6px 10px;
    selection-background-color: {t['selection_bg']};
    selection-color: {t['selection_text']}; }}
QLineEdit:hover {{ border-color: {t['border_hover']}; }}
QLineEdit:focus {{ border: 1.5px solid {t['accent']}; }}
QLineEdit:disabled {{ color: {dis}; background: {t['panel']}; }}
QLineEdit#expr {{
    font-family: {font_mono}; font-size: 16px; padding: 10px 14px;
    border-radius: {RADIUS}px;
}}
QLineEdit#expr:focus {{ border: 1.5px solid {t['accent']}; }}
QPlainTextEdit {{ background: {t['surface']}; color: {t['text']};
    border: 1px solid {t['border']}; border-radius: {RADIUS_SMALL}px;
    padding: 4px;
    selection-background-color: {t['selection_bg']};
    selection-color: {t['selection_text']}; }}
QPlainTextEdit:focus {{ border: 1.5px solid {t['accent']}; }}

/* --- списки, переключатели ------------------------------------------------ */
QListWidget {{ background: {t['panel']}; color: {t['text']};
    border: 1px solid {t['border']}; border-radius: {RADIUS}px; padding: 4px; }}
QListWidget::item {{ padding: 5px 8px; border-radius: {RADIUS_SMALL}px; }}
QListWidget::item:selected {{ background: {t['selection_bg']};
    color: {t['selection_text']}; }}
QListWidget::item:hover:!selected {{ background: {t['surface_hover']}; }}
QRadioButton, QCheckBox {{ color: {t['text']}; spacing: 7px; }}

/* --- прочее ---- ------------------------------------------------------- */
QStatusBar {{ background: {t['window']}; color: {t['text_dim']};
    border: none; }}
QToolBar {{ background: {t['panel']}; border: none; spacing: 2px; }}
QToolButton {{ background: transparent; color: {t['text_dim']};
    border: none; border-radius: {RADIUS_SMALL}px; padding: 3px; }}
QToolButton:hover {{ background: {t['surface_hover']}; color: {t['text']}; }}
QToolButton:pressed {{ background: {t['surface_press']}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {t['border']};
    border-radius: 4px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {t['border_hover']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}

/* --- базовая кнопка и согласованные варианты ---------------------------- */
QPushButton {{
    background: {t['surface']}; color: {t['text']};
    border: 1px solid {t['border']}; border-radius: {RADIUS_SMALL}px;
    padding: {6}px {4}px; font-size: 13px; min-height: 28px;
}}
QPushButton:hover {{ background: {t['surface_hover']};
    border-color: {t['border_hover']}; }}
QPushButton:pressed {{ background: {t['surface_press']}; }}
QPushButton:focus {{ border: 1.5px solid {t['accent_border']}; }}
QPushButton:disabled {{
    background: {t['panel']}; color: {dis}; border-color: {t['border']};
}}
"""
    css = base + _btn("num", t['surface'], t['text'], t['border'],
                      t['surface_hover'], t['surface_press'], weight="600")
    css += _btn("sci", t['panel'], t['text_dim'], t['border'],
                t['surface_hover'], t['surface_press'])
    css += _btn("op", t['accent_tint'], t['accent'], t['accent_border'],
                t['surface_hover'], t['surface_press'], weight="600")
    css += _btn("calc", t['panel'], t['accent'], t['accent_border'],
                t['surface_hover'], t['surface_press'])
    css += _btn("mem", t['surface'], t['text_dim'], t['border'],
                t['surface_hover'], t['surface_press'])
    css += _btn("eq", t['accent'], t['accent_text'], t['accent'],
                t['accent_hover'], t['accent_press'], weight="600")
    css += _btn("clr", t['error_bg'], t['error'], t['error_border'],
                t['surface_hover'], t['surface_press'], disabled_fg=dis)
    return css


# ------------------------------------------------------------- matplotlib
def plot_style(t: Dict[str, str]) -> Dict[str, str]:
    """Токены окна графика. PNG экспортируется с ними же (WYSIWYG)."""
    return {
        "fig_face": t["panel"],        # фон вокруг осей
        "axes_face": t["window"],      # фон области построения
        "axis_color": t["text"],       # оси через начало координат
        "spine_minor": t["border"],    # верх/право (теневые рамки)
        "grid_color": t["border"],
        "grid_alpha": 0.55,
        "tick_color": t["border_hover"],
        "tick_label": t["text_dim"],
        "axis_label": t["text_dim"],
        "legend_text": t["text"],
        "legend_face": t["panel"],
        "legend_edge": t["border"],
        "asym_color": t["text_dim"],
    }


# ------------------------------------------------------------- текущая тема
_current: Dict[str, str] = DARK


def current() -> Dict[str, str]:
    """Активная палитра: для заборов цвета там, где QSS не дотягивается
    (matplotlib, предпросмотр формул, PNG-экспорт)."""
    return _current


def theme_names() -> list:
    return list(THEMES)


def current_theme_name() -> str:
    return _current["name"]


def apply_app_theme(app, name: str) -> Dict[str, str]:
    """Применить тему к QApplication: палитра, шрифт, общий stylesheet.

    Неизвестное имя -> тёмная. Возвращает активный словарь палитры.
    """
    global _current
    t = THEMES.get(name, DARK)
    _current = t
    from PySide6.QtGui import QFont

    app.setPalette(app_palette(t))
    font = QFont()
    font.setFamilies(FONT_UI)
    font.setPointSize(10)
    app.setFont(font)
    app.setStyleSheet(stylesheet(t))
    return t
