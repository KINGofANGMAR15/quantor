"""Окно «График» — PySide6 + matplotlib (бэкенд qtagg).

Немодальное: можно вернуться к калькулятору, посчитать новое значение
и перестроить. Ввод: одна запись на строку:
- функция y=f(x) — кривая;
- неявная связь (x^2+y^2=25, x*y=1, y^2<x) — контур/область.
Есть зум/панорама (панель инструментов), автоматически вычисляются
и подписываются асимптоты, оси x и y выделены ярко.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import matplotlib
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QPlainTextEdit, QPushButton, QVBoxLayout,
)

# Бэкенд выбирается до каких-либо отрисовок; PySide6 уже загружен,
# так что matplotlib привяжется именно к нему.
matplotlib.use("qtagg", force=True)
from matplotlib.backends.backend_qtagg import (                      # noqa: E402
    FigureCanvasQTAgg, NavigationToolbar2QT,
)
from matplotlib.figure import Figure                                 # noqa: E402
from matplotlib.lines import Line2D                                  # noqa: E402
from matplotlib.patches import Patch                                 # noqa: E402

from ..engine import CalculatorEngine, CalculatorError
from ..plotter import MAX_FUNCTIONS, PlotError, plot_data
from . import theme

_FUNC_PLACEHOLDER = "sin(x)\nx^2+y^2=25\nx^2+y^2<16"

_MASK_KINDS = {"lt": lambda z: z < 0, "le": lambda z: z <= 0,
               "gt": lambda z: z > 0, "ge": lambda z: z >= 0}

#: Пауза перед перерасчётом данных под новый вид, мс
NAV_DEBOUNCE_MS = 250


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

        self.logx_cb = QCheckBox("лог. X")
        self.logx_cb.setToolTip(
            "Логарифмическая шкала по X: границы без положительной части\n"
            "поднимаются до 10⁻⁴·x₁")
        self.logy_cb = QCheckBox("лог. Y")
        self.logy_cb.setToolTip("Логарифмическая шкала по Y")
        scale_row = QHBoxLayout()
        scale_row.addWidget(self.logx_cb)
        scale_row.addWidget(self.logy_cb)
        scale_row.addStretch(1)
        form.addRow("Шкала:", scale_row)

        form.addRow("Переменная:", self.var_edit)
        layout.addLayout(form)

        # --- холст matplotlib -------------------------------------------------
        self.figure = Figure(figsize=(7, 4.4), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setObjectName("plot_canvas")
        layout.addWidget(self.canvas, 1)

        # панель инструментов: зум, панорама, «домой»
        self.nav = NavigationToolbar2QT(self.canvas, self)
        self.nav.setObjectName("plot_toolbar")
        layout.addWidget(self.nav)

        self.status = QLabel("")
        self.status.setObjectName("status")
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

        # Пересчёт содержимого при перемещении вида (панорама/зум): после буферизации
        # паузы перестраиваем данные под текущие границы — иначе за пределами
        # исходного диапазона кривая просто обрывается.
        self._result = None          # type: Optional[Any]
        self._text = ""              # разобранный текст для перерасчёта
        self._var = "x"
        self._in_render = False
        self._nav_timer = QTimer(self)
        self._nav_timer.setSingleShot(True)
        self._nav_timer.setInterval(NAV_DEBOUNCE_MS)
        self._nav_timer.timeout.connect(self._refresh_view)
        self.canvas.mpl_connect("draw_event", self._on_draw_event)
        self.logx_cb.toggled.connect(self._plot_current)
        self.logy_cb.toggled.connect(self._plot_current)

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
        logx = self.logx_cb.isChecked()
        requested_x0 = self._opt_float(self.x0_edit)
        try:
            result = plot_data(
                self.funcs_edit.toPlainText(), self.engine, var,
                x0=self._opt_float(self.x0_edit),
                x1=self._opt_float(self.x1_edit),
                y0=self._opt_float(self.y0_edit),
                y1=self._opt_float(self.y1_edit),
                logx=logx, logy=self.logy_cb.isChecked(),
            )
        except (PlotError, CalculatorError, ValueError) as exc:
            self.status.setText(f"Ошибка: {exc}")
            self._result = None
            return
        except Exception as exc:  # защита: окно не должно падать
            self.status.setText(f"Внутренняя ошибка: {exc.__class__.__name__}")
            self._result = None
            return
        self._text = self.funcs_edit.toPlainText()
        self._var = var
        self.nav_timer_stop()
        self._render(result, save=True)
        if logx and (requested_x0 is None or requested_x0 <= 0):
            self.status.setText(self.status.text()
                                + f"\nЛог-шкала X: диапазон от "
                                  f"{result.xlim[0]:g} (лог-ось не содержит 0)")

    # ------------------------------------------------------------- рисование
    def nav_timer_stop(self) -> None:
        if self._nav_timer.isActive():
            self._nav_timer.stop()

    def _on_draw_event(self, event=None) -> None:
        """Если границы вида изменились — отложенно пересчитать содержимое."""
        if self._in_render or self._result is None:
            return
        if not self.figure.axes:
            return
        ax = self.figure.axes[0]
        lo, hi = ax.get_xlim()
        vlo, vhi = ax.get_ylim()
        x0, x1 = self._result.xlim
        y0, y1 = self._result.ylim
        same = (abs(lo - x0) < 1e-12 and abs(hi - x1) < 1e-12
                and abs(vlo - y0) < 1e-12 and abs(vhi - y1) < 1e-12)
        if not same:
            self._nav_timer.start()

    def _refresh_view(self) -> None:
        """Построить данные под текущие границы окна (после панорамы/зума)."""
        self.nav_timer_stop()
        result = self._result
        if result is None or not self.figure.axes:
            return
        ax = self.figure.axes[0]
        lo, hi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        vals = (float(lo), float(hi), float(ylo), float(yhi))
        if not all(np.isfinite(vals)) or hi <= lo or yhi <= ylo:
            return
        if (hi - lo) < 1e-9:                      # слишком глубокий зум
            return
        if (abs(lo - result.xlim[0]) < 1e-12
                and abs(hi - result.xlim[1]) < 1e-12
                and abs(ylo - result.ylim[0]) < 1e-12
                and abs(yhi - result.ylim[1]) < 1e-12):
            return                                # вид не менялся
        try:
            fresh = plot_data(self._text, self.engine, self._var,
                              x0=lo, x1=hi, y0=ylo, y1=yhi,
                              logx=self.logx_cb.isChecked(),
                              logy=self.logy_cb.isChecked())
        except Exception:
            return                                # остаётся старая картинка
        # границы могли слегка округлиться — фиксируем вид заново
        fresh.xlim = (float(lo), float(hi))
        fresh.ylim = (float(ylo), float(yhi))
        self._result = fresh                      # до отрисовки: событие draw
        self._render(fresh, save=False)

    def _render(self, result, save: bool) -> None:
        self._result = result
        self._in_render = True
        st = theme.plot_style(theme.current())
        try:
            self.figure.clear()
            self.figure.set_facecolor(st["fig_face"])
            ax = self.figure.add_subplot(111)
            ax.set_facecolor(st["axes_face"])
            if self.logx_cb.isChecked():          # до отрисовки данных
                ax.set_xscale("log")
            if self.logy_cb.isChecked():
                ax.set_yscale("log")
            self._draw_curves(ax, result)
            self._draw_implicit(ax, result)
            self._draw_asymptotes(ax, result)
            ax.set_xlabel(result.xlabel)
            ax.set_ylabel("y")
            ax.set_xlim(*result.xlim)
            ax.set_ylim(*result.ylim)
            ax.grid(True, alpha=st["grid_alpha"], linewidth=0.6,
                    color=st["grid_color"])
            self._style_axes(ax)
            self._legend_handles(ax, result)
            self.canvas.draw_idle()
        finally:
            self._in_render = False

        notes: List[str] = []
        if result.breaks:
            notes.append(f"Скрыто особых точек: {result.breaks} "
                         "(асимптоты/вне области определения)")
        if result.asymptote_texts:
            notes.append("Асимптоты: " + "; ".join(result.asymptote_texts))
        self.status.setText("\n".join(notes))
        if save:
            self._save_state()
    @staticmethod
    def _draw_curves(ax, result) -> None:
        for c in result.curves:
            ax.plot(c.xs, c.ys, color=c.color, linewidth=1.6, zorder=2.4)

    @staticmethod
    def _draw_implicit(ax, result) -> None:
        """Неявные: равенство — контур E=0; неравенство — заливка области."""
        for m in result.meshes:
            try:
                if m.kind == "eq":
                    ax.contour(m.X, m.Y, m.Z, levels=[0.0], colors=[m.color],
                               linewidths=1.8, zorder=2.3)
                else:
                    mask = _MASK_KINDS[m.kind](m.Z)
                    filled = np.where(mask, 1.0, np.nan)
                    ax.contourf(m.X, m.Y, filled, levels=[0.5, 1.1],
                                colors=[m.color], alpha=0.22, zorder=1.5)
                    ax.contour(m.X, m.Y, m.Z, levels=[0.0], colors=[m.color],
                               linewidths=1.2, zorder=2.3)
            except Exception:
                # пустая область/весь NaN — не даём окну упасть
                continue

    @staticmethod
    def _draw_asymptotes(ax, result) -> None:
        """Штриховые линии по найденным асимптотам."""
        st = theme.plot_style(theme.current())
        color = st["asym_color"]
        kw = dict(color=color, ls=(0, (4, 3)), lw=1.0, alpha=0.85, zorder=1.8)
        lo, hi = ax.get_xlim()
        for xp in result.v_positions:
            if lo < xp < hi:
                ax.axvline(xp, **kw)
        for yp in result.h_values:
            ax.axhline(yp, **kw)
        for k, b in result.o_lines:
            ax.plot([lo, hi], [k * lo + b, k * hi + b], **kw)

    @staticmethod
    def _style_axes(ax) -> None:
        """Яркие оси: толстые левая/нижняя рамки + оси через начало координат.
        Цвета — из активной темы gui.theme."""
        st = theme.plot_style(theme.current())
        for side in ("top", "right"):
            ax.spines[side].set_color(st["spine_minor"])
            ax.spines[side].set_linewidth(0.8)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(st["axis_color"])
            ax.spines[side].set_linewidth(1.8)
        ax.tick_params(colors=st["tick_color"], labelcolor=st["tick_label"])
        ax.xaxis.label.set_color(st["axis_label"])
        ax.yaxis.label.set_color(st["axis_label"])
        lo, hi = ax.get_xlim()
        if lo <= 0 <= hi:
            ax.axvline(0, color=st["axis_color"], lw=1.4, zorder=1.9)
        lo, hi = ax.get_ylim()
        if lo <= 0 <= hi:
            ax.axhline(0, color=st["axis_color"], lw=1.4, zorder=1.9)

    @staticmethod
    def _legend_handles(ax, result) -> None:
        """Легенда по прокси-элементам (сам контур может ещё не иметь линий)."""
        st = theme.plot_style(theme.current())
        handles = []
        for c in result.curves:
            handles.append(Line2D([], [], color=c.color, lw=1.6, label=c.label))
        for m in result.meshes:
            if m.kind == "eq":
                handles.append(Line2D([], [], color=m.color, lw=1.8,
                                      label=m.label))
            else:
                handles.append(Patch(facecolor=m.color, alpha=0.30,
                                     edgecolor=m.color, label=m.label))
        if len(handles) > 1:
            legend = ax.legend(handles=handles, loc="upper right", fontsize=8)
            legend.get_frame().set_facecolor(st["legend_face"])
            legend.get_frame().set_edgecolor(st["legend_edge"])
            for text in legend.get_texts():
                text.set_color(st["legend_text"])

    # ------------------------------------------------------------ прочее
    def _save_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить график", "quantor_plot.png",
            "Изображение PNG (*.png)")
        if path:
            if not path.lower().endswith(".png"):
                path += ".png"
            try:
                # WYSIWYG: экспортируем с теми же цветами темы, что на экране
                self.figure.savefig(
                    path, dpi=150, facecolor=self.figure.get_facecolor())
                self.status.setText(f"Сохранено: {path}")
            except OSError as exc:
                self.status.setText(f"Не удалось сохранить: {exc}")

    def _reset(self) -> None:
        self.nav_timer_stop()
        self._result = None
        self.funcs_edit.setPlainText("")
        for e in (self.x0_edit, self.x1_edit, self.y0_edit, self.y1_edit):
            e.clear()
        for cb in (self.logx_cb, self.logy_cb):
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        st = theme.plot_style(theme.current())
        self.figure.clear()
        self.figure.set_facecolor(st["fig_face"])
        self.figure.add_subplot(111).set_facecolor(st["axes_face"])
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
        s["plot_logx"] = self.logx_cb.isChecked()
        s["plot_logy"] = self.logy_cb.isChecked()
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
        # переключение чекбоксов влечёт _plot_current: при загрузке без сигналов
        for key, cb in (("plot_logx", self.logx_cb),
                        ("plot_logy", self.logy_cb)):
            cb.blockSignals(True)
            cb.setChecked(bool(s.get(key)))
            cb.blockSignals(False)
