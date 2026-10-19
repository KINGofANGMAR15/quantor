"""Диалоги «Квантора» (PySide6): уравнения/полиномы и история."""

from __future__ import annotations

from typing import Callable, List, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QPushButton, QRadioButton, QVBoxLayout,
)

from ..calculus import CalculusEngine
from ..engine import CalculatorEngine, CalculatorError, _fmt_number
from ..latex_input import LatexParseError


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
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)

        self.eq_edit = QLineEdit("x^3-6x^2+11x-6=0")
        form.addRow("Уравнение (2^x=8, x^2-4=0):", self.eq_edit)

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
        self.status.setStyleSheet("color: #b3261e;")
        layout.addWidget(self.status)

        buttons = QHBoxLayout()
        solve = QPushButton("Решить")
        solve.setDefault(True)
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
        var = self.var_edit.text().strip() or "x"
        try:
            if self.mode_poly.isChecked():
                raw = self.coeff_edit.text().replace(",", " ").split()
                coeffs = [float(tok) for tok in raw]
                roots = (self.engine if var == "x" else self.engine)\
                    .solve_polynomial(coeffs, var)
            else:
                roots = self.engine.solve_equation(self.eq_edit.text(), var)
        except (CalculatorError, ValueError) as exc:
            self.status.setText(f"Ошибка: {exc}")
            return
        if not roots:
            self.status.setText("Корней не найдено")
            return
        for i, root in enumerate(roots, 1):
            self.list.addItem(f"x{i} = {_fmt_number(root)}")

    def _insert_selected(self, *_args) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        self._insert(item.text().split("=", 1)[1].strip())


class HistoryDialog(QDialog):
    """История вычислений: двойной щелчок вставляет выражение."""

    def __init__(self, parent, history: List[Tuple[str, str]],
                 insert_callback: Callable[[str], None]) -> None:
        super().__init__(parent)
        self._history = history
        self._insert = insert_callback
        self.setWindowTitle("История — Квантор")
        self.resize(520, 380)

        layout = QVBoxLayout(self)
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(self._recall)
        for expr, result in history:
            self.list.addItem(f"{expr}  =  {result}")
        layout.addWidget(self.list, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.reject)
        recall = QPushButton("Вставить выражение")
        recall.clicked.connect(self._recall)
        buttons.addButton(recall, QDialogButtonBox.ButtonRole.ActionRole)
        layout.addWidget(buttons)

    def _recall(self, *_args) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        expr = item.text().split("  =  ", 1)[0]
        self._insert(expr)
