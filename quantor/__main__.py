"""Запуск «Квантора» командой ``python -m quantor``."""

import sys

from quantor.engine import CalculatorEngine
from quantor.gui import run_app


def main() -> int:
    engine = CalculatorEngine()
    run_app(engine)
    return 0


if __name__ == "__main__":
    sys.exit(main())
