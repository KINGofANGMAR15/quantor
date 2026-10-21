"""Окно «График» — PySide6 + matplotlib (бэкенд qtagg).

Немодальное: можно вернуться к калькулятору, посчитать новое значение
и перестроить. Ввод: одна функция на строку (каждая — отдельная кривая).
"""

from __future__ import annotations

from typing import Optional

import matplotlib
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QVBoxLayout,
)

# Бэкенд выбирается до каких-либо отрисовок; PySide6 уже загружен,
# так что matplotlib привяжется именно к нему.
matplotlib.use("qtagg", force=True)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure                             # noqa: E402

from ..engine import CalculatorEngine, CalculatorError
from ..plotter import MAX_FUNCTIONS, PlotError, plot_data

_FUNC_PLACEHOLDER = "sin(x)\ncos(x)\nx^2-4"


class PlotDialog(QDialog):
    """Окно построения графиков функций одной переменной."""

    def __init__(self, parent, engine: CalculatorEngine) -> None:
        super().__init__(parent)
        self.engine = engine
        self.setWindowTitle("График — Квантор")
        self.setMinimumSize(720, 520)
        self.setModal(False)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)

        self.funcs_edit = QPlainTextEdit()
        self.funcs_edit.setPlaceholderText(
            _FUNC_PLACEHOLDER + "\n\nодна функция на строку, до "
            f"{MAX_FUNCTIONS} штук")
        self.funcs_edit.setFixedHeight(76)
        form.addRow("Функции:", self.funcs_edit)

        self.var_edit = QLineEdit("x")
        self.x0_edit = QLineEdit()
        self.x1_edit = QLineEdit()
        for e in (self.x0_edit, self.x1_edit):
            e.setPlaceholderText("авто")
        self.y0_edit = QLineEdit()
        self.y1_edit = QLineEdit()
        rng = QHBoxLayout()
        rng.addWidget(QLabel("x от"))
        rng.addWidget(self.x0_edit)
        rng.addWidget(QLabel("до"))
        rng.addWidget(self.x1_edit)
        rng.addSpacing(12)
        rng.addWidget(QLabel("y от"))
        rng.addWidget(self.y0_edit)
        rng.addWidget(QLabel("до"))
        rng.addWidget(self.y1_edit)
        rng.addStretch(1)
        form.addRow("Диапазон:", rng)
        form.addRow("Переменная:", self.var_edit)
        layout.addLayout(form)

        # --- холст matplotlib -------------------------------------------------
        self.figure = Figure(figsize=(7, 4.4), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setObjectName("plot_canvas")
        layout.addWidget(self.canvas, 1)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        buttons = QHBoxLayout()
        plot_btn = QPushButton("Построить")
        plot_btn.setObjectName("eq")
        plot_btn.setDefault(True)
        plot_btn.clicked.connect(self._plot_current)
        save_btn = QPushButton("Сохранить PNG…")
        save_btn.clicked.connect(self._save_png)
        reset_btn = QPushButton("Сброс")
        reset_btn.clicked.connect(self._reset)
        close = QPushButton("Закрыть")
        close.clicked.connect(self.close)
        buttons.addWidget(plot_btn)
        buttons.addWidget(save_btn)
        buttons.addWidget(reset_btn)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)

        self._load_state()
        self._plot_current()

    # ------------------------------------------------------------ построение
    def _opt_float(self, edit: QLineEdit) -> Optional[float]:
        text = edit.text().strip().replace(",", ".").replace("−", "-")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _plot_current(self) -> None:
        self.status.setText("")
        var = (self.var_edit.text().strip() or "x")[:8]
        try:
            result = plot_data(
                self.funcs_edit.toPlainText(), self.engine, var,
                x0=self._opt_float(self.x0_edit),
                x1=self._opt_float(self.x1_edit),
                y0=self._opt_float(self.y0_edit),
                y1=self._opt_float(self.y1_edit),
            )
        except (PlotError, CalculatorError, ValueError) as exc:
            self.status.setText(f"Ошибка: {exc}")
            return
        except Exception as exc:  # защита: окно не должно падать
            self.status.setText(f"Внутренняя ошибка: {exc.__class__.__name__}")
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        for c in result.curves:
            ax.plot(c.xs, c.ys, color=c.color, linewidth=1.6, label=c.label)
        ax.set_xlabel(result.xlabel)
        ax.set_ylabel("y")
        ax.set_ylim(*result.ylim)
        ax.grid(True, alpha=0.35, linewidth=0.6)
        if len(result.curves) > 1:
            ax.legend(loc="upper right", fontsize=8)
        self.canvas.draw_idle()

        note = ""
        if result.breaks:
            note = (f" · скрыто особых точек: {result.breaks} "
                    "(асимптоты/вне области определения)")
        self.status.setText(note)
        self._save_state()

    # ------------------------------------------------------------ прочее
    def _save_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить график", "quantor_plot.png",
            "Изображение PNG (*.png)")
        if path:
            if not path.lower().endswith(".png"):
                path += ".png"
            try:
                self.figure.savefig(path, dpi=150, facecolor="white")
                self.status.setText(f"Сохранено: {path}")
            except OSError as exc:
                self.status.setText(f"Не удалось сохранить: {exc}")

    def _reset(self) -> None:
        self.funcs_edit.setPlainText("")
        for e in (self.x0_edit, self.x1_edit, self.y0_edit, self.y1_edit):
            e.clear()
        self.figure.clear()
        self.figure.add_subplot(111).grid(True, alpha=0.35)
        self.canvas.draw_idle()
        self.status.setText("")
        self._save_state()

    # ------------------------------------------------------------ настройки
    def _save_state(self) -> None:
        from ..settings import load_settings, save_settings
        s = load_settings()
        s["plot_functions"] = self.funcs_edit.toPlainText()
        s["plot_var"] = self.var_edit.text()
        s["plot_x0"] = self.x0_edit.text()
        s["plot_x1"] = self.x1_edit.text()
        s["plot_y0"] = self.y0_edit.text()
        s["plot_y1"] = self.y1_edit.text()
        save_settings(s)

    def _load_state(self) -> None:
        from ..settings import load_settings
        s = load_settings()
        funcs = s.get("plot_functions")
        if isinstance(funcs, str):
            self.funcs_edit.setPlainText(funcs)
        else:
            self.funcs_edit.setPlainText(_FUNC_PLACEHOLDER)
        self.var_edit.setText(str(s.get("plot_var", "x")) or "x")
        for key, edit in (("plot_x0", self.x0_edit), ("plot_x1", self.x1_edit),
                          ("plot_y0", self.y0_edit), ("plot_y1", self.y1_edit)):
            val = s.get(key)
            if isinstance(val, str):
                edit.setText(val)
