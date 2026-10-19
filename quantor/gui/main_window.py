"""Главное окно «Квантора» на PySide6 (Qt 6).

Структура сверху вниз: панель живого предпросмотра LaTeX -> строка ввода ->
результат (текст + предпросмотр формулы) -> сетка кнопок -> статус (режим, M).
Вся математика — в engine.CalculatorEngine и calculus.CalculusEngine.
"""

from __future__ import annotations

from typing import Optional

import sympy as sp
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from .. import constants as K
from ..calculus import CalculusEngine, is_calculus_query
from ..engine import (
    ANGLE_MODES, CalculatorEngine, CalculatorError, _fmt_number,
    _exact_to_number, format_exact,
)
from ..latex_input import (
    LatexParseError, is_latex, parse_input, render_latex, to_latex,
)
from ..settings import load_settings, save_settings
from .dialogs import EquationDialog, HistoryDialog

# --- палитра (светлая) ---------------------------------------------------------
BG = "#f4f4f2"
FG = "#1a1a1a"
ERR = "#b3261e"
OK = "#0b6623"
GRAY = "#8a8a8a"

STYLESHEET = f"""
QMainWindow, QDialog {{ background: {BG}; color: {FG}; }}
QWidget {{ color: {FG}; }}
QLineEdit#expr {{ font-family: 'Consolas','DejaVu Sans Mono',monospace;
                  font-size: 17px; padding: 8px; background: #ffffff;
                  color: {FG}; border: 1px solid #c9c9c4; border-radius: 4px; }}
QLabel#preview {{ background: #ffffff; border: 1px solid #e0e0db;
                  border-radius: 4px; color: {GRAY}; }}
QLabel#result {{ font-size: 17px; font-weight: bold; color: {FG}; }}
QLabel#result.err {{ color: {ERR}; }}
QPushButton {{ font-size: 13px; padding: 8px 2px; border: 1px solid #cfcfc9;
               border-radius: 4px; background: #ffffff; color: {FG}; }}
QPushButton:hover {{ background: #eef1f5; }}
QPushButton:pressed {{ background: #dfe4ea; }}
QPushButton:disabled {{ color: #9a9a9a; }}
QPushButton#sci {{ background: #e8f0fe; }}
QPushButton#op {{ background: #eef3e6; }}
QPushButton#mem {{ background: #eceae4; }}
QPushButton#eq {{ background: #c8e6c9; font-weight: bold; }}
QPushButton#clr {{ background: #ffcdd2; }}
QPushButton#calc {{ background: #ede7f6; font-weight: bold; }}
QLabel#mode {{ font-weight: bold; color: {FG}; }}
QLabel#memind {{ font-weight: bold; color: {OK}; }}
QMenuBar {{ background: {BG}; color: {FG}; }}
QMenuBar::item:selected {{ background: #e4e7ec; }}
QMenu {{ background: #ffffff; color: {FG}; }}
QMenu::item:selected {{ background: #4c8bf5; color: #ffffff; }}
QStatusBar {{ background: {BG}; color: {GRAY}; }}
QListWidget {{ background: #ffffff; color: {FG}; }}
QRadioButton {{ color: {FG}; }}
QToolTip {{ background: #ffffe1; color: {FG}; border: 1px solid #c9c9c4; }}
"""


class MainWindow(QMainWindow):
    """Главное окно калькулятора."""

    def __init__(self, engine: CalculatorEngine) -> None:
        super().__init__()
        self.engine = engine
        self.calculus = CalculusEngine(engine)
        self._history: list[tuple[str, str]] = []
        self._last_value: Optional[float] = None

        self.setWindowTitle("Квантор — инженерный калькулятор")
        self.setMinimumSize(660, 600)
        self._build_menu()
        self._build_ui()
        self._bind_keys()
        self._apply_settings()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(6)

        # Живой предпросмотр LaTeX
        self.preview = QLabel(" ")
        self.preview.setObjectName("preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(76)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Fixed)
        root.addWidget(self.preview)

        # Строка ввода
        self.input = QLineEdit()
        self.input.setObjectName("expr")
        self.input.setPlaceholderText(
            "выражение или LaTeX: sin(30) · \\frac{\\sin(x)}{2} · \\int_{0}^{\\pi} \\sin(x) dx")
        root.addWidget(self.input)

        # Результат: текст + предпросмотр формулы
        self.result = QLabel(" ")
        self.result.setObjectName("result")
        self.result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.result.setWordWrap(True)
        root.addWidget(self.result)

        self.result_preview = QLabel(" ")
        self.result_preview.setObjectName("preview")
        self.result_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_preview.setMinimumHeight(64)
        root.addWidget(self.result_preview)

        # Статус: режим углов + индикатор памяти
        status = QHBoxLayout()
        self.mode_btn = QPushButton(self.engine.angle_mode)
        self.mode_btn.setObjectName("mem")
        self.mode_btn.setFixedWidth(72)
        self.mode_btn.setToolTip("Переключить градусы/радианы")
        self.mode_btn.clicked.connect(self._toggle_mode)
        status.addWidget(self.mode_btn)
        self.mem_label = QLabel("")
        self.mem_label.setObjectName("memind")
        status.addWidget(self.mem_label)
        status.addStretch(1)
        root.addLayout(status)

        # Сетка кнопок
        grid_holder = QWidget()
        grid = QGridLayout(grid_holder)
        grid.setSpacing(4)
        rows = self._button_rows()
        styles = self._button_styles()
        for r, row in enumerate(rows):
            for c, (label, action) in enumerate(row):
                btn = QPushButton(label)
                btn.setObjectName(styles.get(label, "sci"))
                btn.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
                if callable(action):
                    btn.clicked.connect(action)
                else:
                    btn.clicked.connect(lambda _=False, t=action: self._insert(t))
                grid.addWidget(btn, r, c)
        for c in range(7):
            grid.setColumnStretch(c, 1)
        root.addWidget(grid_holder, 1)

        self.setCentralWidget(central)

        # Таймер дебаунса предпросмотра (< 100 мс отклик)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(80)
        self._preview_timer.timeout.connect(self._update_preview)
        self.input.textEdited.connect(lambda _: self._preview_timer.start())
        self.input.textChanged.connect(lambda _: self._preview_timer.start())

    def _button_rows(self):
        """Сетка кнопок: слева научные функции, справа цифровой блок."""
        return [
            [("DEG/RAD", self._toggle_mode), ("π", "π"), ("e", "e"),
             ("Ans", "Ans"), ("C", self._clear), ("⌫", self._backspace),
             ("Уравнение", self._open_equation)],
            [("sin", "sin("), ("cos", "cos("), ("tan", "tan("),
             ("7", "7"), ("8", "8"), ("9", "9"), ("÷", "÷")],
            [("asin", "asin("), ("acos", "acos("), ("atan", "atan("),
             ("4", "4"), ("5", "5"), ("6", "6"), ("×", "×")],
            [("ln", "ln("), ("log", "log("), ("log₁₀", "log10("),
             ("1", "1"), ("2", "2"), ("3", "3"), ("−", "−")],
            [("log₂", "log2("), ("x²", "**2"), ("xʸ", "^"),
             ("0", "0"), (".", "."), ("+", "+"), ("=", self._evaluate)],
            [("√", "√"), ("n!", "!"), ("|x|", "abs("),
             ("(", "("), (")", ")"), ("mod(", "mod("), ("M+", self._mem_add)],
            [("MR", self._mem_recall), ("M−", self._mem_sub), ("MC", self._mem_clear),
             ("1/x", "^(-1)"), ("10ˣ", "10^"), ("eˣ", "exp("),
             ("История", self._show_history)],
            [("∫ dx", "\\int "), ("∫ₐᵇ dx", "\\int_{0}^{1} "),
             ("d/dx", "\\frac{d}{dx} "), ("d²/dx²", "\\frac{d^2}{dx^2} "),
             ("d³/dx³", "\\frac{d^3}{dx^3} "), ("|ₓ₌ₐ", "\\big|_{x=0}"),
             ("⧉ LaTeX", self._copy_latex)],
        ]

    @staticmethod
    def _button_styles() -> dict:
        styles = {name: "mem" for name in
                  ("DEG/RAD", "M+", "M−", "MR", "MC", "История")}
        styles.update({"C": "clr", "⌫": "clr", "=": "eq", "Уравнение": "sci"})
        styles.update({name: "op" for name in ("÷", "×", "−", "+", "^", "!")})
        styles.update({name: "calc" for name in
                       ("∫ dx", "∫ₐᵇ dx", "d/dx", "d²/dx²", "d³/dx³", "|ₓ₌ₐ")})
        return styles

    def _bind_keys(self) -> None:
        self.input.returnPressed.connect(self._evaluate)
        seq_clear = QAction("Очистить", self)
        seq_clear.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        seq_clear.triggered.connect(self._clear)
        self.addAction(seq_clear)

        seq_eq = QAction("Решить уравнение", self)
        seq_eq.setShortcut(QKeySequence("Ctrl+E"))
        seq_eq.triggered.connect(self._open_equation)
        self.addAction(seq_eq)

        seq_copy = QAction("Копировать LaTeX", self)
        seq_copy.setShortcut(QKeySequence("Ctrl+Shift+C"))
        seq_copy.triggered.connect(self._copy_latex)
        self.addAction(seq_copy)

    # ------------------------------------------------------------------ меню
    def _build_menu(self) -> None:
        menubar = self.menuBar()

        mode_menu = menubar.addMenu("Углы")
        group = QActionGroup(self)
        for mode in ANGLE_MODES:
            act = QAction(mode, self)
            act.setCheckable(True)
            act.setChecked(mode == self.engine.angle_mode)
            act.triggered.connect(lambda _=False, m=mode: self._set_mode(m))
            group.addAction(act)
            mode_menu.addAction(act)

        const_menu = menubar.addMenu("Константы")
        for name, (value, label, unit) in sorted(K.CONSTANTS.items()):
            act = QAction(f"{name} = {value:g} — {label} ({unit})", self)
            act.triggered.connect(lambda _=False, n=name: self._insert(n))
            const_menu.addAction(act)

        hist = QAction("История", self)
        hist.triggered.connect(self._show_history)
        menubar.addAction(hist)
        about = QAction("О программе", self)
        about.triggered.connect(self._show_about)
        menubar.addAction(about)

    # ------------------------------------------------------------------ ввод
    def _insert(self, text: str) -> None:
        self.input.insert(text)
        self.input.setFocus()

    def _insert_at_cursor(self, text: str) -> None:
        self._insert(text)

    def _clear(self) -> None:
        self.input.clear()
        self.result.setText(" ")
        self.result.setProperty("err", False)
        self._style_refresh(self.result)
        self.result_preview.clear()
        self._update_preview()

    def _backspace(self) -> None:
        self.input.backspace()
        self.input.setFocus()

    def _style_refresh(self, label: QLabel) -> None:
        label.style().unpolish(label)
        label.style().polish(label)

    # ------------------------------------------------------------------ вычисления
    def _evaluate(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        try:
            if is_calculus_query(text):
                self._show_calculus(self.calculus.evaluate_query(text))
            elif is_latex(text):
                self._show_symbolic(parse_input(text, self.engine))
            else:
                shown, value = self.engine.calculate(text)
                self._show_plain(shown, value)
        except (LatexParseError, CalculatorError, ValueError) as exc:
            self._show_error(str(exc))
        except Exception as exc:  # последнее защитное сито — приложение не падает
            self._show_error(f"Внутренняя ошибка: {exc.__class__.__name__}")

    def _show_plain(self, shown: str, value) -> None:
        self._last_value = value
        self._history.append((self.input.text().strip(), shown))
        self.result.setText(f"= {shown}")
        self.result.setProperty("err", False)
        self._style_refresh(self.result)
        self._set_result_preview(to_latex(sp.sympify(value)))
        self.input.setText(shown)
        self.input.setCursorPosition(len(shown))

    def _show_symbolic(self, expr: sp.Basic) -> None:
        latex = to_latex(expr)
        if expr.free_symbols:
            shown = str(expr).replace("**", "^")
            value = None
        else:
            try:
                value = _exact_to_number(expr)
            except CalculatorError as exc:
                self._show_error(str(exc))
                return
            shown = self._display_text(expr, value)
        self._last_value = value
        self._history.append((self.input.text().strip(), shown))
        self.result.setText(f"= {shown}")
        self.result.setProperty("err", False)
        self._style_refresh(self.result)
        self._set_result_preview(latex)
        if value is not None:
            self.engine.ans = value

    @staticmethod
    def _display_text(expr: sp.Basic, value) -> str:
        """Точный вид (1/2, √3/2, π/4), иначе — десятичный."""
        exact = format_exact(expr)
        if exact is not None:
            return exact
        return _fmt_number(value)

    def _show_calculus(self, res) -> None:
        self._last_value = res.value
        self._history.append((self.input.text().strip(), "; ".join(res.lines)))
        self.result.setText("\n".join(res.lines))
        self.result.setProperty("err", False)
        self._style_refresh(self.result)
        self._set_result_preview(res.latex)
        if res.value is not None:
            self.engine.ans = res.value

    def _show_error(self, message: str) -> None:
        self.result.setText(f"Ошибка: {message}")
        self.result.setProperty("err", True)
        self._style_refresh(self.result)
        self.result_preview.clear()

    # ------------------------------------------------------------------ предпросмотр
    def _update_preview(self) -> None:
        text = self.input.text().strip()
        if not text:
            self.preview.clear()
            self.preview.setText(" ")
            return
        latex = text if is_latex(text) else None
        if latex is None:
            try:  # обычный ввод тоже показываем формулой
                latex = to_latex(parse_input(text, self.engine))
            except Exception:
                latex = None
        source = latex if latex is not None else text
        pixmap = render_latex(source)
        if pixmap is not None and not pixmap.isNull():
            self.preview.setPixmap(pixmap)
            self.preview.setText("")
        else:
            self.preview.setPixmap(self._empty_pixmap())
            self.preview.setText("предпросмотр недоступен — проверьте ввод")

    def _set_result_preview(self, latex: str) -> None:
        pixmap = render_latex(latex)
        if pixmap is not None and not pixmap.isNull():
            self.result_preview.setPixmap(pixmap)
            self.result_preview.setText("")
        else:
            self.result_preview.setPixmap(self._empty_pixmap())
            self.result_preview.setText("")

    @staticmethod
    def _empty_pixmap():
        from PySide6.QtGui import QPixmap
        return QPixmap()

    # ------------------------------------------------------------------ LaTeX
    def _current_latex(self) -> str:
        text = self.input.text().strip()
        if not text:
            return ""
        if is_calculus_query(text):
            return text  # запрос уже в LaTeX
        try:
            return to_latex(parse_input(text, self.engine))
        except Exception:
            return text

    def _copy_latex(self) -> None:
        latex = self._current_latex()
        if latex:
            QGuiApplication.clipboard().setText(latex)
            self.statusBar().showMessage("LaTeX скопирован в буфер обмена", 2500)

    # ------------------------------------------------------------------ память
    def _current_value(self) -> float:
        """Число с дисплея: результат последнего '=' или пересчёт ввода."""
        if self._last_value is not None and not self.input.text().strip():
            return float(self._last_value)
        text = self.input.text().strip()
        if not text:
            return float(self._last_value or 0.0)
        if is_calculus_query(text):
            res = self.calculus.evaluate_query(text)
            if res.value is None:
                raise ValueError("Результат не является числом")
            return float(res.value)
        if is_latex(text):
            expr = parse_input(text, self.engine)
            if expr.free_symbols:
                raise ValueError("В выражении есть неизвестные")
            return float(sp.N(expr))
        _shown, value = self.engine.calculate(text)
        return float(value)

    def _mem_add(self) -> None:
        try:
            self.engine.mem_add(self._current_value())
        except (CalculatorError, LatexParseError, ValueError) as exc:
            self._show_error(str(exc))
            return
        self._update_mem()

    def _mem_sub(self) -> None:
        try:
            self.engine.mem_sub(self._current_value())
        except (CalculatorError, LatexParseError, ValueError) as exc:
            self._show_error(str(exc))
            return
        self._update_mem()

    def _mem_recall(self) -> None:
        self._insert(_fmt_number(self.engine.mem_recall()))

    def _mem_clear(self) -> None:
        self.engine.mem_clear()
        self._update_mem()

    def _update_mem(self) -> None:
        self.mem_label.setText("M" if abs(self.engine.mem_recall()) > 1e-300 else "")

    # ------------------------------------------------------------------ окна
    def _toggle_mode(self) -> None:
        self._set_mode("RAD" if self.engine.angle_mode == "DEG" else "DEG")

    def _set_mode(self, mode: str) -> None:
        self.engine.set_angle_mode(mode)
        self.mode_btn.setText(mode)
        for act in self.menuBar().actions()[0].menu().actions():
            act.setChecked(act.text() == mode)
        self._update_preview()

    def _open_equation(self) -> None:
        EquationDialog(self, self.engine, self.calculus, self._insert_at_cursor).exec()

    def _show_history(self) -> None:
        HistoryDialog(self, self._history, self._insert_at_cursor).exec()

    def _show_about(self) -> None:
        QMessageBox.information(
            self, "О программе",
            "Квантор — офлайн инженерный калькулятор.\n\n"
            "Ядро: SymPy + mpmath. GUI: PySide6 (Qt 6).\n"
            "LaTeX-предпросмотр: matplotlib mathtext.\n"
            "Все вычисления выполняются локально, сеть не используется.\n\n"
            "Горячие клавиши: Enter — вычислить, Esc — очистить,\n"
            "Ctrl+E — уравнения, Ctrl+Shift+C — копировать LaTeX.")

    # ------------------------------------------------------------------ настройки
    def _apply_settings(self) -> None:
        from PySide6.QtCore import QByteArray
        settings = load_settings()
        mode = settings.get("angle_mode")
        if mode in ANGLE_MODES:
            self._set_mode(mode)
        geo = settings.get("qt_geometry")
        if isinstance(geo, str):
            try:
                self.restoreGeometry(QByteArray.fromHex(geo.encode("ascii")))
            except Exception:
                pass

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt-имя)
        settings = load_settings()
        settings["angle_mode"] = self.engine.angle_mode
        settings["qt_geometry"] = bytes(self.saveGeometry().toHex()).decode("ascii")
        save_settings(settings)
        super().closeEvent(event)
