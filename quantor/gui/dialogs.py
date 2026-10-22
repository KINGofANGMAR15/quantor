"""Диалоги «Квантора» (PySide6): уравнения/полиномы, неравенства, история.

Стилевые цвета — только через gui.theme (objectName + общий QSS),
inline-палитра запрещена.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QPushButton, QRadioButton, QVBoxLayout,
)

from ..calculus import CalculusEngine
from ..engine import CalculatorEngine, CalculatorError, _fmt_number


class EquationDialog(QDialog):
    """Решение уравнений (текстом) и полиномов (по коэффициентам)."""

    def __init__(self, parent, engine: CalculatorEngine,
                 calculus: CalculusEngine,
                 insert_callback: Callable[[str], None]) -> None:
        super().__init__(parent)
        self.engine = engine
        self.calculus = calculus
        self._insert = insert_callback
        self.setWindowTitle("Уравнения — Квантор")
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 14)
        layout.setSpacing(10)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(10)

        self.eq_edit = QLineEdit("x^3-6x^2+11x-6=0")
        self.eq_edit.setToolTip(
            "Обычная запись: 2^x=8, x^2-4=0. LaTeX тоже: \\log_2 x = 10, "
            "\\frac{x-1}{2} = 5")
        form.addRow("Уравнение (текст или LaTeX, 2^x=8, \\log_2 x=10):",
                    self.eq_edit)

        self.mode_eq = QRadioButton("Уравнение (текстом)")
        self.mode_poly = QRadioButton("Полином, коэффициенты по убыванию степеней:")
        self.mode_eq.setChecked(True)
        self.coeff_edit = QLineEdit("1, -6, 11, -6")
        self.coeff_edit.setEnabled(False)
        self.mode_poly.toggled.connect(self.coeff_edit.setEnabled)
        form.addRow(self.mode_eq)
        form.addRow(self.mode_poly, self.coeff_edit)

        self.var_edit = QLineEdit("x")
        form.addRow("Переменная:", self.var_edit)
        layout.addLayout(form)

        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(self._insert_selected)
        layout.addWidget(self.list, 1)

        self.status = QLabel("")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        buttons = QHBoxLayout()
        solve = QPushButton("Решить")
        solve.setObjectName("eq")
        solve.setDefault(True)
        solve.setCursor(Qt.CursorShape.PointingHandCursor)
        solve.clicked.connect(self.solve)
        insert = QPushButton("Вставить в дисплей")
        insert.clicked.connect(self._insert_selected)
        close = QPushButton("Закрыть")
        close.clicked.connect(self.reject)
        buttons.addWidget(solve)
        buttons.addWidget(insert)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)

        self.eq_edit.returnPressed.connect(self.solve)

    def solve(self) -> None:
        self.list.clear()
        self.status.setText("")
        self.status.setProperty("err", False)
        self._refresh_status()
        var = self.var_edit.text().strip() or "x"
        try:
            if self.mode_poly.isChecked():
                raw = self.coeff_edit.text().replace(",", " ").split()
                coeffs = [float(tok) for tok in raw]
                roots = self.engine.solve_polynomial(coeffs, var)
            else:
                roots = self.engine.solve_equation(self.eq_edit.text(), var)
        except (CalculatorError, ValueError) as exc:
            self.status.setText(f"Ошибка: {exc}")
            self.status.setProperty("err", True)
            self._refresh_status()
            return
        if not roots:
            self.status.setText("Корней не найдено")
            return
        for i, root in enumerate(roots, 1):
            self.list.addItem(f"x{i} = {_fmt_number(root)}")
        # Первая строка выбрана сразу: «Вставить в дисплей» работает без клика
        self.list.setCurrentRow(0)

    def _refresh_status(self) -> None:
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    def _insert_selected(self, *_args) -> None:
        item = self.list.currentItem() or (
            self.list.item(0) if self.list.count() else None)
        if item is None:
            return
        self._insert(item.text().split("=", 1)[1].strip())


class InequalityDialog(QDialog):
    """Решение неравенств степени 1–4: вывод множества решений."""

    def __init__(self, parent, engine: CalculatorEngine,
                 insert_callback: Callable[[str], None]) -> None:
        super().__init__(parent)
        self.engine = engine
        self._insert = insert_callback
        self.setWindowTitle("Неравенства — Квантор")
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 14)
        layout.setSpacing(10)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(10)

        self.ineq_edit = QLineEdit("x^2-4<0")
        form.addRow("Неравенство (x^2-4<0, 2x-1>=3):", self.ineq_edit)

        self.var_edit = QLineEdit("x")
        form.addRow("Переменная:", self.var_edit)
        layout.addLayout(form)

        self.list = QListWidget()
        layout.addWidget(self.list, 1)

        self.status = QLabel("")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        hint = QLabel(
            "Поддерживаются полиномы степени 1–4: x^2-4<0, "
            "x^3-6x^2+11x-6>=0, 0.5x<2.\n"
            "Знаки: <, >, <=, >=, ≤, ≥, а также \\leq и \\ge.\n"
            "Правую часть можно опустить (x<0); части можно писать LaTeX'ом: "
            "\\frac{x-1}{2} <= 0.")
        hint.setObjectName("cap")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        buttons = QHBoxLayout()
        solve = QPushButton("Решить")
        solve.setObjectName("eq")
        solve.setDefault(True)
        solve.clicked.connect(self.solve)
        insert = QPushButton("Вставить в дисплей")
        insert.setToolTip("Вставить неравенство в строку ввода калькулятора")
        insert.clicked.connect(self._insert_ineq)
        close = QPushButton("Закрыть")
        close.clicked.connect(self.reject)
        buttons.addWidget(solve)
        buttons.addWidget(insert)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)

        self.ineq_edit.returnPressed.connect(self.solve)

    def solve(self) -> None:
        self.list.clear()
        self.status.setText("")
        self.status.setProperty("err", False)
        self._refresh_status()
        var = self.var_edit.text().strip() or "x"
        try:
            parts = self.engine.solve_inequality(self.ineq_edit.text(), var)
        except (CalculatorError, ValueError) as exc:
            self.status.setText(f"Ошибка: {exc}")
            self.status.setProperty("err", True)
            self._refresh_status()
            return
        for part in parts:
            self.list.addItem(part)

    def _refresh_status(self) -> None:
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    def _insert_ineq(self, *_args) -> None:
        text = self.ineq_edit.text().strip()
        if text:
            self._insert(text)


class HistoryDialog(QDialog):
    """История вычислений: двойной щелчок вставляет выражение."""

    def __init__(self, parent, history: List[Tuple[str, str]],
                 insert_callback: Callable[[str], None]) -> None:
        super().__init__(parent)
        self._history = history
        self._insert = insert_callback
        self.setWindowTitle("История — Квантор")
        self.resize(540, 400)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 14)
        layout.setSpacing(10)
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(self._recall)
        for expr, result in history:
            self.list.addItem(f"{expr}  =  {result}")
        if self.list.count():
            self.list.setCurrentRow(0)
        layout.addWidget(self.list, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.reject)
        recall = QPushButton("Вставить выражение")
        recall.clicked.connect(self._recall)
        buttons.addButton(recall, QDialogButtonBox.ButtonRole.ActionRole)
        layout.addWidget(buttons)

    def _recall(self, *_args) -> None:
        item = self.list.currentItem() or (
            self.list.item(0) if self.list.count() else None)
        if item is None:
            return
        expr = item.text().split("  =  ", 1)[0]
        self._insert(expr)
