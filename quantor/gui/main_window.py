"""Главное окно «Квантора» на PySide6 (Qt 6).

Структура сверху вниз: заголовочная строка (режим углов, память) ->
панель живого предпросмотра LaTeX -> строка ввода -> панель результата
(текст + формула) -> клавиатура калькулятора внизу фиксированной высоты.
Оба предпросмотра — прозрачный matplotlib, цвет формулы берётся из темы.
Вся математика — в engine.CalculatorEngine и calculus.CalculusEngine.
Стили и палитры — в gui.theme, отдельного «второго списка цветов» нет.
"""

from __future__ import annotations

from typing import Optional

import sympy as sp
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from .. import constants as K
from ..calculus import CalculusEngine, is_calculus_query
from ..engine import (
    ANGLE_MODES, CalculatorEngine, CalculatorError, INEQ_RE,
    _fmt_number, _exact_to_number, _format_solution_set, _guess_var,
    format_exact, number_to_sympy,
)
from ..latex_input import (
    LatexParseError, is_latex, parse_input, render_latex, to_latex,
)
from ..settings import load_settings, save_settings
from . import theme
from .dialogs import EquationDialog, HistoryDialog, InequalityDialog
from .plot_window import PlotDialog


class MainWindow(QMainWindow):
    """Главное окно калькулятора."""

    def __init__(self, engine: CalculatorEngine) -> None:
        super().__init__()
        self.engine = engine
        self.calculus = CalculusEngine(engine)
        self._history: list[tuple[str, str]] = []
        self._last_value: Optional[float] = None
        self._result_latex = ""          # формула результата — для смены темы

        self.setWindowTitle("Квантор — инженерный калькулятор")
        self.setMinimumSize(680, 640)
        self._build_menu()
        self._build_ui()
        self._bind_keys()
        self._apply_settings()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("central")
        root = QVBoxLayout(central)
        root.setContentsMargins(theme.MARGIN, theme.MARGIN, theme.MARGIN,
                                theme.MARGIN)
        root.setSpacing(theme.SPACING)

        # --- заголовок: режим углов, индикатор памяти ------------------------
        header = QWidget()
        header.setObjectName("header")
        head = QHBoxLayout(header)
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(theme.SPACING)
        self.mode_btn = QPushButton(self.engine.angle_mode)
        self.mode_btn.setObjectName("calc")          # акцентная кнопка режима
        self.mode_btn.setToolTip("Переключить градусы/радианы")
        self.mode_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mode_btn.clicked.connect(self._toggle_mode)
        head.addWidget(self.mode_btn)
        self.mem_label = QLabel("")
        self.mem_label.setObjectName("memind")
        head.addWidget(self.mem_label)
        head.addStretch(1)
        root.addWidget(header)

        # --- панель живого предпросмотра LaTeX -------------------------------
        preview_frame = QFrame()
        preview_frame.setObjectName("panel")
        pv = QVBoxLayout(preview_frame)
        pv.setContentsMargins(10, 8, 10, 8)
        self.preview = QLabel(" ")
        self.preview.setObjectName("preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(96)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Expanding)
        pv.addWidget(self.preview)
        root.addWidget(preview_frame, 1)

        # --- строка ввода (главный элемент) ----------------------------------
        self.input = QLineEdit()
        self.input.setObjectName("expr")
        self.input.setPlaceholderText(
            "выражение или LaTeX: sin(30) · \\frac{\\sin(x)}{2} · "
            "\\int_{0}^{\\pi} \\sin(x) dx")
        root.addWidget(self.input)

        # --- панель результата: текст + математическое представление ---------
        result_frame = QFrame()
        result_frame.setObjectName("panel")
        rl = QVBoxLayout(result_frame)
        rl.setContentsMargins(14, 10, 14, 10)
        rl.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(theme.SPACING)
        self.result = QLabel(" ")
        self.result.setObjectName("result")
        self.result.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.result.setWordWrap(True)
        self.result.setMinimumHeight(38)
        self.result.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Preferred)
        row.addWidget(self.result, 1)
        self.result_preview = QLabel(" ")
        self.result_preview.setObjectName("preview")
        self.result_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_preview.setMinimumHeight(64)
        self.result_preview.setSizePolicy(QSizePolicy.Policy.Expanding,
                                          QSizePolicy.Policy.Expanding)
        row.addWidget(self.result_preview, 0, Qt.AlignmentFlag.AlignVCenter)
        rl.addLayout(row, 1)
        root.addWidget(result_frame)

        # --- клавиатура калькулятора -----------------------------------------
        grid_holder = QWidget()
        grid = QGridLayout(grid_holder)
        grid.setSpacing(6)
        grid.setContentsMargins(0, 0, 0, 0)
        for r, row_items in enumerate(self._button_rows()):
            for c, (label, action) in enumerate(row_items):
                btn = QPushButton(label)
                btn.setObjectName(self._button_styles().get(label, "sci"))
                btn.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
                btn.setMinimumHeight(34)
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                if callable(action):
                    btn.clicked.connect(action)
                else:
                    btn.clicked.connect(
                        lambda _=False, t=action: self._insert(t))
                grid.addWidget(btn, r, c)
        for c in range(7):
            grid.setColumnStretch(c, 1)
        root.addWidget(grid_holder)

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
            [("MR", self._mem_recall), ("M−", self._mem_sub),
             ("MC", self._mem_clear),
             ("1/x", "^(-1)"), ("10ˣ", "10^"), ("eˣ", "exp("),
             ("История", self._show_history)],
            [("∫ dx", "\\int "), ("∫ₐᵇ dx", "\\int_{0}^{1} "),
             ("d/dx", "\\frac{d}{dx} "), ("d²/dx²", "\\frac{d^2}{dx^2} "),
             ("d³/dx³", "\\frac{d^3}{dx^3} "), ("|ₓ₌ₐ", "\\big|_{x=0}"),
             ("⧉ LaTeX", self._copy_latex)],
        ]

    @staticmethod
    def _button_styles() -> dict:
        """Роль каждой кнопки: рендерится единым QSS из gui.theme."""
        styles = {name: "mem" for name in
                  ("DEG/RAD", "M+", "M−", "MR", "MC", "История")}
        styles.update({name: "num" for name in
                       ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".")})
        styles.update({"C": "clr", "⌫": "clr", "=": "eq",
                       "Уравнение": "sci", "⧉ LaTeX": "sci"})
        styles.update({name: "op" for name in
                       ("÷", "×", "−", "+", "^", "!", "**2", "^(-1)",
                        "10^", "exp(", "√", "abs(", "mod(")})
        styles.update({name: "calc" for name in
                       ("∫ dx", "∫ₐᵇ dx", "d/dx", "d²/dx²", "d³/dx³",
                        "|ₓ₌ₐ")})
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

        seq_ineq = QAction("Решить неравенство", self)
        seq_ineq.setShortcut(QKeySequence("Ctrl+I"))
        seq_ineq.triggered.connect(self._open_inequality)
        self.addAction(seq_ineq)

        seq_plot = QAction("Построить график", self)
        seq_plot.setShortcut(QKeySequence("Ctrl+G"))
        seq_plot.triggered.connect(self._open_plot)
        self.addAction(seq_plot)

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

        view_menu = menubar.addMenu("Вид")
        view_group = QActionGroup(self)
        self._theme_actions = {}
        for tname, title in (("dark", "Тёмная тема"),
                             ("light", "Светлая тема")):
            act = QAction(title, self)
            act.setCheckable(True)
            act.setChecked(theme.current_theme_name() == tname)
            act.triggered.connect(
                lambda _=False, n=tname: self._set_theme(n))
            view_group.addAction(act)
            view_menu.addAction(act)
            self._theme_actions[tname] = act

        hist = QAction("История", self)
        hist.triggered.connect(self._show_history)
        menubar.addAction(hist)

        ineq = QAction("Неравенства", self)
        ineq.triggered.connect(self._open_inequality)
        menubar.addAction(ineq)

        plot = QAction("График", self)
        plot.triggered.connect(self._open_plot)
        menubar.addAction(plot)
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
        self._set_result_preview("")
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
                res = self.calculus.evaluate_query(text)
                if res is not None:
                    self._show_calculus(res)
                    return
                # похоже на матанализ, но шаблону не соответствует
                # (\int без dx, непривычная запись пределов) — идём дальше:
                # общий путь даст понятную ошибку разбора
            if INEQ_RE.search(text):
                self._show_inequality(text)
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
        self._set_result_preview(
            to_latex(number_to_sympy(value)) if value is not None else "")
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

    def _show_inequality(self, text: str) -> None:
        """Неравенство прямо в строке ввода: x^2-4<0 -> «-2 < x < 2»."""
        var = _guess_var(text)
        try:
            sol = self.engine.solve_inequality_set(text, var)
        except CalculatorError as exc:
            self._show_error(str(exc))
            return
        except Exception as exc:
            self._show_error(f"Внутренняя ошибка: {exc.__class__.__name__}")
            return
        shown = " или ".join(_format_solution_set(sol, var))
        # без дубля переменной: «x = 5», а не «= x = 5» (для любой переменной)
        prefix = "" if shown.startswith(f"{var} =") else "= "
        self._last_value = None  # Ans не трогаем: число здесь не при чём
        self._history.append((text, shown))
        self.result.setText(f"{prefix}{shown}")
        self.result.setProperty("err", False)
        self._style_refresh(self.result)
        self._set_result_preview(to_latex(sol))

    def _show_error(self, message: str) -> None:
        # «Ошибка ввода» без дубля префикса
        body = message if message.startswith("Ошибка") else f"Ошибка: {message}"
        self.result.setText(body)
        self.result.setProperty("err", True)
        self._style_refresh(self.result)
        self._set_result_preview("")

    # ------------------------------------------------------------------ предпросмотр
    def _theme_color(self) -> str:
        return theme.current()["text"]

    def _update_preview(self) -> None:
        text = self.input.text().strip()
        if not text:
            self.preview.clear()
            self.preview.setText(" ")
            return
        latex = text if is_latex(text) else None
        if latex is None:
            try:  # обычный ввод показываем формулой через безопасный парсер
                expr = self.engine._parse(text, self.engine._namespace())
                latex = to_latex(expr)
            except Exception:
                latex = None
        source = latex if latex is not None else text
        pixmap = render_latex(source, fontsize=18, color=self._theme_color())
        if pixmap is not None and not pixmap.isNull():
            self.preview.setPixmap(pixmap)
            self.preview.setText("")
        else:
            self.preview.setPixmap(self._empty_pixmap())
            self.preview.setText("предпросмотр недоступен — проверьте ввод")

    def _set_result_preview(self, latex: str) -> None:
        self._result_latex = latex or ""
        pixmap = render_latex(latex, color=self._theme_color()) if latex else None
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
            return self._mem_value(self._last_value)
        text = self.input.text().strip()
        if not text:
            return self._mem_value(self._last_value or 0.0)
        if is_calculus_query(text):
            res = self.calculus.evaluate_query(text)
            if res.value is None:
                raise ValueError("Результат не является числом")
            return self._mem_value(res.value)
        if is_latex(text):
            expr = parse_input(text, self.engine)
            if expr.free_symbols:
                raise ValueError("В выражении есть неизвестные")
            return self._mem_value(sp.N(expr))
        _shown, value = self.engine.calculate(text)
        return self._mem_value(value)

    @staticmethod
    def _mem_value(value) -> float:
        """Число для ячеек памяти (M+/M−): комплексные не сохраняются."""
        if isinstance(value, complex):
            raise ValueError("Комплексное число нельзя сохранить в память")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Результат не является числом") from exc

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
        self.mem_label.setText("M" if abs(self.engine.mem_recall()) > 1e-300
                               else "")

    # ------------------------------------------------------------------ окна
    def _toggle_mode(self) -> None:
        self._set_mode("RAD" if self.engine.angle_mode == "DEG" else "DEG")

    def _set_mode(self, mode: str) -> None:
        self.engine.set_angle_mode(mode)
        self.mode_btn.setText(mode)
        for act in self.menuBar().actions()[0].menu().actions():
            act.setChecked(act.text() == mode)
        self._update_preview()

    # ------------------------------------------------------------------ темы
    def _set_theme(self, name: str) -> None:
        """Переключить тему: палитра+QSS, перерендер формул, сохранение."""
        app = QGuiApplication.instance()
        if app is None:
            return
        theme.apply_app_theme(app, name)
        settings = load_settings()
        settings["theme"] = name
        save_settings(settings)
        for tname, act in self._theme_actions.items():
            act.setChecked(tname == name)
        # формулы отрендерены в старом цвете — перерисовать
        self._update_preview()
        self._set_result_preview(self._result_latex)
        self._update_mem()

    def _open_equation(self) -> None:
        EquationDialog(self, self.engine, self.calculus,
                       self._insert_at_cursor).exec()

    def _open_inequality(self) -> None:
        InequalityDialog(self, self.engine, self._insert_at_cursor).exec()

    def _open_plot(self) -> None:
        """Немодальное окно графика: одно на приложение, фокус при повторе."""
        win = getattr(self, "_plot_win", None)
        if win is not None and win.isVisible():
            win.raise_()
            win.activateWindow()
            return
        self._plot_win = PlotDialog(self, self.engine)
        self._plot_win.show()

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
            "Ctrl+E — уравнения, Ctrl+I — неравенства, Ctrl+G — график,\n"
            "Ctrl+Shift+C — копировать LaTeX.")

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
        settings["theme"] = theme.current_theme_name()
        settings["qt_geometry"] = bytes(
            self.saveGeometry().toHex()).decode("ascii")
        save_settings(settings)
        super().closeEvent(event)
