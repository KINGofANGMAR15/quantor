# -*- mode: python ; coding: utf-8 -*-
# Спека PyInstaller для «Квантора»: сборка one-file GUI-приложения.
#   Windows:  pyinstaller --clean quantor.spec   -> dist/Quantor.exe
#   Linux:    pyinstaller --clean quantor.spec   -> dist/Quantor
#
# Tkinter (tcl/tk DLL и библиотеки) PyInstaller упаковывает автоматически.

a = Analysis(
    ['quantor/__main__.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    # Qt WebEngine/Quick не нужны калькулятору — экономим ~200 МБ в бинарнике
    excludes=[
        'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets',
        'PySide6.QtWebEngineQuick', 'PySide6.QtWebChannel',
        'PySide6.QtQml', 'PySide6.QtQuick', 'PySide6.QtQuick3D',
        'PySide6.QtQuickWidgets', 'PySide6.QtQuickControls2',
        'PySide6.Qt3DCore', 'PySide6.Qt3DRender', 'PySide6.Qt3DInput',
        'PySide6.Qt3DLogic', 'PySide6.Qt3DAnimation', 'PySide6.Qt3DExtras',
        'PySide6.QtPositioning', 'PySide6.QtLocation', 'PySide6.QtMultimedia',
        'PySide6.QtPdf', 'PySide6.QtPdfWidgets', 'PySide6.QtCharts',
        'PySide6.QtDataVisualization', 'PySide6.QtSensors', 'PySide6.QtSerialPort',
        'PySide6.QtTest', 'PySide6.QtDesigner', 'PySide6.QtHelp',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Quantor',
    debug=False,
    strip=False,
    upx=True,
    console=False,          # GUI-приложение: без консольного окна
    icon=None,
)
