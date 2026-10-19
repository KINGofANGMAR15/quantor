"""Общие фикстуры тестов «Квантора».

Qt-виджеты требуют QApplication; в CI/контейнерах используем offscreen-платформу.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(["quantor-tests"])
    yield app
