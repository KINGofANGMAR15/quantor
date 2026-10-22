"""Запуск «Квантора» командой ``python -m quantor``."""

import sys

from quantor.engine import CalculatorEngine
from quantor.gui import run_app


def main() -> int:
    engine = CalculatorEngine()
    run_app(engine)
    return 0


def selftest() -> int:
    """Быстрая проверка разбора (в т.ч. LaTeX) внутри собранного бинаря:

    ./Quantor --selftest   — печатает OK/FAIL по вариантам ввода, код 0/1.
    """
    from quantor.engine import CalculatorError
    from quantor.latex_input import parse_input

    engine = CalculatorEngine()

    cases = [
        ("plain log", "log_2 x", "log"),
        ("LaTeX log", r"\log_2 x", "log"),
        ("LaTeX frac", r"\frac{x}{2}", "x"),
        ("plain text", "sin(x)/2", "x"),
    ]
    failed: list[str] = []
    for name, text, needle in cases:
        try:
            s = str(parse_input(text, engine).free_symbols)
            ok = needle in str(parse_input(text, engine)) or needle in s
        except Exception as exc:                  # noqa: BLE001 — самотест
            ok, s = False, f"{type(exc).__name__}: {exc}"
        print(("OK   " if ok else "FAIL ") + f"{name}: {text!r} -> {s}")
        if not ok:
            failed.append(name)
    print("selftest:", "PASSED" if not failed else f"FAILED {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(main())
