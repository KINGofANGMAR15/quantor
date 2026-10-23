"""LaTeX-ввод и рендеринг формул «Квантора».

* parse_input(text) — автоопределение формата (LaTeX или обычное выражение)
  и разбор в SymPy-выражение.
* render_latex(tex) — рендеринг LaTeX в QPixmap через matplotlib mathtext
  (без TeX Live/MiKTeX, без sympy.preview()).
"""

from __future__ import annotations

import io
import re
from typing import Any, Dict, Optional, Tuple

import sympy as sp

from .engine import CalculatorEngine
from .math_parser import MAX_INPUT_LEN

__all__ = ["LatexParseError", "is_latex", "parse_input", "to_latex",
           "render_latex", "mathtext_safe"]


class LatexParseError(Exception):
    """Ошибка разбора LaTeX — текст показывается пользователю как есть."""


# Признаки LaTeX: любая команда \foo, а также ^{...} / _{...}
_LATEX_HINT = re.compile(r"\\[a-zA-Z]+|[_^]\s*\{")

_DEFAULT_ENGINE: Optional[CalculatorEngine] = None


def is_latex(text: str) -> bool:
    """True, если строка похожа на LaTeX, а не на обычное выражение."""
    text = text or ""
    if "\\" in text:
        return True
    return bool(_LATEX_HINT.search(text))


def _default_engine() -> CalculatorEngine:
    """Ленивый движок в RAD-режиме — только для разбора выражений."""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = CalculatorEngine("RAD")
    return _DEFAULT_ENGINE


def parse_input(text: str, engine: Optional[CalculatorEngine] = None) -> sp.Expr:
    """Разобрать ввод пользователя в SymPy-выражение.

    Формат определяется автоматически: есть команды LaTeX — парсим как LaTeX,
    иначе — обычным парсером ядра (sin(x)/2, 2^x, константы и т.д.).

    Разбор ВСЕГДА математический (аргументы тригонометрии — в радианах):
    режим DEG/RAD применяется только в calculate() арифметического движка,
    а интегралы/производные от углов не зависят.
    """
    text = (text or "").strip()
    if not text:
        raise LatexParseError("Пустой ввод")
    if len(text) > MAX_INPUT_LEN:
        # ANTLR-парсер linearён по длине, но на сотнях килобайт уже висит
        raise LatexParseError(
            f"Выражение слишком длинное (максимум {MAX_INPUT_LEN} символов)")
    if is_latex(text):
        return _parse_latex_expr(text)
    eng = _default_engine()  # RAD: чистая математика без угловой конверсии
    try:
        expr = eng._parse(text, eng._namespace())
    except Exception as exc:
        raise LatexParseError(f"Не удалось разобрать выражение: {exc}") from exc
    if not isinstance(expr, sp.Basic):
        # всё выражение — имя-константа (e, C, Ans): движок даёт python-число
        from .engine import number_to_sympy
        expr = number_to_sympy(expr)
    return expr


# ---------------------------------------------------------------------------
# Разбор LaTeX
# ---------------------------------------------------------------------------

#: Замены перед parse_latex: конструкции, которые он не понимает.
_PRE_LATEX = (
    (r"\dfrac", r"\frac"),
    (r"\tfrac", r"\frac"),
    (r"\cdot", r"*"),
    (r"\left", ""),
    (r"\right", ""),
    (r"\,", " "),
    (r"\;", " "),
    (r"\!", ""),
    (r"^\prime", "'"),
)


def _clean_latex(text: str) -> str:
    text = text.strip().strip("$").strip()
    for bad, good in _PRE_LATEX:
        text = text.replace(bad, good)
    return text.strip()


def _parse_latex_expr(text: str) -> sp.Expr:
    cleaned = _clean_latex(text)
    if not cleaned:
        raise LatexParseError("Пустой ввод")
    try:
        result = _parse_latex_sympy(cleaned)
    except LatexParseError:
        raise
    except Exception as exc:  # любые внутренние ошибки парсера — в понятный текст
        raise LatexParseError(f"Не удалось разобрать LaTeX: «{cleaned}»") from exc
    if isinstance(result, (list, tuple)):  # некоторые версии возвращают список
        if not result:
            raise LatexParseError(f"Не удалось разобрать LaTeX: «{cleaned}»")
        result = result[0]
    if not isinstance(result, sp.Basic):
        raise LatexParseError(f"Не удалось разобрать LaTeX: «{cleaned}»")
    result = _latex_constants(result)
    try:  # parse_latex возвращает неупрощённые суммы: 1/3 + 1/2 -> 5/6
        result = sp.simplify(result)
    except Exception:
        pass
    return result


def _latex_constants(expr: sp.Basic) -> sp.Basic:
    r"""parse_latex даёт pi и e как СИМВОЛЫ — подменяем их на константы SymPy.

    Иначе \int_{0}^{\pi} считается с 'pi' как неизвестной переменной.
    (Побочный эффект: 'e' нельзя использовать как неизвестную в LaTeX-вводе —
    для калькулятора это осознанный компромисс.)
    """
    subs = {}
    for name, const in (("pi", sp.pi), ("e", sp.E)):
        sym = sp.Symbol(name)
        if expr.has(sym):
            subs[sym] = const
    return expr.subs(subs) if subs else expr


def _parse_latex_sympy(tex: str) -> sp.Basic:
    """parse_latex (строгий режим) с запасным парсером latex2sympy2_extended.

    strict=True обязателен: без него ANTLR-парсер «восстанавливается» молча —
    «x + » и «x^» разбираются как просто «x», лишняя «}» отбрасывается,
    то есть часть выражения немо теряется.
    """
    try:
        from sympy.parsing.latex import parse_latex
    except ImportError as exc:
        raise LatexParseError(
            "Для LaTeX-парсера нужен пакет antlr4-python3-runtime "
            "(pip install antlr4-python3-runtime)") from exc
    try:
        return parse_latex(tex, strict=True)
    except Exception as parse_exc:
        # Запасной вариант: latex2sympy2_extended понимает больше конструкций.
        try:
            from latex2sympy2_extended import latex2sympy
        except ImportError:
            raise parse_exc  # запасного нет — отдаём исходную ошибку наверх
        try:
            return latex2sympy(tex)
        except Exception:
            raise LatexParseError(f"Не удалось разобрать LaTeX: «{tex}»")


# ---------------------------------------------------------------------------
# LaTeX <-> текст
# ---------------------------------------------------------------------------

def to_latex(expr: Any) -> str:
    """SymPy-выражение -> строка LaTeX (для рендера и копирования)."""
    return sp.latex(expr)


#: Замены для matplotlib mathtext (он беднее полноценного LaTeX).
_MH_REPLACEMENTS = (
    (r"\big|", "|"), (r"\Big|", "|"), (r"\bigg|", "|"), (r"\Bigg|", "|"),
    (r"\limits", ""),
    (r"\displaystyle", ""),
    (r"\dfrac", r"\frac"),
    (r"\tfrac", r"\frac"),
    (r"\operatorname{", r"\mathrm{"),
)


def mathtext_safe(tex: str) -> str:
    """Привести LaTeX к подмножеству, которое понимает matplotlib mathtext."""
    for bad, good in _MH_REPLACEMENTS:
        tex = tex.replace(bad, good)
    return tex


# ---------------------------------------------------------------------------
# Рендеринг (matplotlib mathtext -> PNG -> QPixmap)
# ---------------------------------------------------------------------------

_RENDER_CACHE: Dict[Tuple[str, int, str], Any] = {}
_CACHE_LIMIT = 64
_DPI = 200          # рендерим с запасом, затем devicePixelRatio=2 -> чётко на HiDPI

_APP_HOLDERS: list = []  # держим ссылку на QGuiApplication, чтобы его не убил GC


def _ensure_qapp():
    """QPixmap требует QGuiApplication; создаём offscreen, если его нет."""
    from PySide6.QtGui import QGuiApplication

    app = QGuiApplication.instance()
    if app is None:
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QGuiApplication(["quantor-render"])
        _APP_HOLDERS.append(app)
    return app


def render_latex(tex: str, fontsize: int = 16, color: str = "#1a1a1a"):
    """LaTeX -> QPixmap (или None, если отрендерить не удалось).

    Кэшируется: повторный ввод не перерисовывается — предпросмотр быстрый.
    """
    tex = mathtext_safe(tex.strip())
    key = (tex, fontsize, color)
    cached = _RENDER_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        from PySide6.QtGui import QPixmap
    except ImportError:
        return None
    _ensure_qapp()

    import matplotlib
    matplotlib.use("Agg")  # без GUI-бэкендов и без TeX-дистрибутива
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure()
    _ = FigureCanvasAgg(fig)  #Agg-канвас нужен savefig'у
    try:
        fig.text(0, 0, f"${tex}$", fontsize=fontsize, color=color)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=_DPI, transparent=True,
                    bbox_inches="tight", pad_inches=0.04)
    except Exception:
        return None  # mathtext не осилил конструкцию — GUI покажет текстом

    pixmap = QPixmap()
    if not pixmap.loadFromData(buf.getvalue(), "PNG"):
        return None
    pixmap.setDevicePixelRatio(2.0)

    if len(_RENDER_CACHE) >= _CACHE_LIMIT:
        _RENDER_CACHE.clear()  # простая защита от разрастания
    _RENDER_CACHE[key] = pixmap
    return pixmap
