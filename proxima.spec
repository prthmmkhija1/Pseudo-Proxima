# -*- mode: python ; coding: utf-8 -*-
"""
Proxima PyInstaller Specification File

Build standalone executables with:
    pyinstaller proxima.spec

Output will be in dist/proxima (or dist/proxima.exe on Windows)
"""

import sys
from pathlib import Path

# Try to import collect functions
try:
    from PyInstaller.utils.hooks import collect_data_files, collect_submodules
except ImportError:
    print("PyInstaller not installed. Run: pip install pyinstaller")
    sys.exit(1)

block_cipher = None

# Project root
ROOT = Path(SPECPATH)

# Collect data files from proxima package
datas = []
try:
    datas.extend(collect_data_files('proxima'))
except Exception:
    # Fallback: manually specify data files
    datas.extend([
        (str(ROOT / 'src' / 'proxima' / 'configs'), 'proxima/configs'),
    ])

# Add configuration files
config_files = [
    ('configs', 'configs'),
]
for src, dst in config_files:
    src_path = ROOT / src
    if src_path.exists():
        datas.append((str(src_path), dst))

# Hidden imports for dynamic module loading
hiddenimports = [
    # Core Python modules
    'asyncio',
    'json',
    'logging',
    'pathlib',
    'typing',
    
    # Proxima modules
    'proxima',
    'proxima.cli',
    'proxima.cli.main',
    'proxima.cli.commands',
    'proxima.backends',
    'proxima.backends.base',
    'proxima.backends.registry',
    'proxima.backends.lret',
    'proxima.backends.cirq_adapter',
    'proxima.backends.qiskit_adapter',
    'proxima.backends.quest_adapter',
    'proxima.backends.qsim_adapter',
    'proxima.backends.cuquantum_adapter',
    'proxima.core',
    'proxima.core.state',
    'proxima.core.executor',
    'proxima.core.planner',
    'proxima.config',
    'proxima.config.settings',
    'proxima.resources',
    'proxima.resources.timer',
    'proxima.resources.monitor',
    'proxima.data',
    'proxima.intelligence',
    
    # Third-party libraries
    'typer',
    'typer.main',
    'click',
    'rich',
    'rich.console',
    'rich.table',
    'rich.progress',
    'pydantic',
    'pydantic_settings',
    'structlog',
    'transitions',
    'transitions.core',
    'psutil',
    'httpx',
    'yaml',
    'pandas',
    'openpyxl',
    'keyring',
]

# Try to add quantum library submodules
try:
    hiddenimports.extend(collect_submodules('cirq'))
except Exception:
    hiddenimports.extend(['cirq', 'cirq.circuits', 'cirq.ops', 'cirq.sim'])

try:
    hiddenimports.extend(collect_submodules('qiskit'))
except Exception:
    hiddenimports.extend(['qiskit', 'qiskit.circuit', 'qiskit_aer'])

# Exclude unnecessary packages to reduce size
excludes = [
    'pytest',
    'pytest_asyncio',
    'pytest_mock',
    'pytest_cov',
    'coverage',
    'mkdocs',
    'mkdocstrings',
    'sphinx',
    'black',
    'ruff',
    'mypy',
    'pre_commit',
    'IPython',
    'jupyter',
    'notebook',
    'matplotlib',
    'tkinter',
    '_tkinter',
    'PyQt5',
    'PyQt6',
    'PySide2',
    'PySide6',
]

# Analysis
a = Analysis(
    ['src/proxima/__main__.py'],
    pathex=[str(ROOT / 'src')],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Remove duplicate files
seen = set()
a.datas = [x for x in a.datas if not (x[0] in seen or seen.add(x[0]))]

# Python archive
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

# Executable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='proxima',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if available: 'assets/icon.ico'
)

# For macOS .app bundle (optional)
# app = BUNDLE(
#     exe,
#     name='Proxima.app',
#     icon='assets/icon.icns',
#     bundle_identifier='io.proxima-project.proxima',
#     info_plist={
#         'NSHighResolutionCapable': 'True',
#         'CFBundleShortVersionString': '0.1.0',
#     },
# )
