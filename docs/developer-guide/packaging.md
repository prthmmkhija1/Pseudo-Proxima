# Packaging Guide

This guide covers building, packaging, and distributing Proxima for various platforms.

## Overview

Proxima supports multiple distribution methods:

| Method | Use Case | Format |
|--------|----------|--------|
| PyPI | Standard Python installation | wheel, sdist |
| Standalone | Single executable, no Python required | exe, AppImage |
| Docker | Container deployments | Docker image |
| Conda | Conda environments | conda package |

## Building from Source

### Prerequisites

```bash
# Install build dependencies
pip install build twine

# Install all development dependencies
pip install -e ".[dev,all]"
```

### Build Wheel and Source Distribution

```bash
# Build both wheel and sdist
python -m build

# Output files in dist/
# - proxima-0.1.0-py3-none-any.whl
# - proxima-0.1.0.tar.gz
```

### Verify Build

```bash
# Check package contents
python -m tarfile -l dist/proxima-0.1.0.tar.gz

# Verify wheel
pip install dist/proxima-0.1.0-py3-none-any.whl --dry-run
```

## PyPI Publishing

### Test PyPI (Recommended First)

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ proxima-agent
```

### Production PyPI

```bash
# Upload to PyPI
python -m twine upload dist/*

# Users can then install with
pip install proxima-agent
```

### Credentials

Store credentials securely:

```bash
# Option 1: Environment variables
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...

# Option 2: .pypirc file
cat > ~/.pypirc << EOF
[pypi]
username = __token__
password = pypi-...

[testpypi]
username = __token__
password = pypi-...
EOF
chmod 600 ~/.pypirc
```

## Standalone Executables

### Using PyInstaller

Build standalone executables that don't require Python installation.

#### Install PyInstaller

```bash
pip install pyinstaller
```

#### Build Configuration

Create `proxima.spec`:

```python
# proxima.spec
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all data files
datas = collect_data_files('proxima')

# Include all submodules for dynamic imports
hiddenimports = (
    collect_submodules('proxima.backends') +
    collect_submodules('proxima.cli') +
    collect_submodules('cirq') +
    collect_submodules('qiskit') +
    ['transitions', 'structlog', 'typer']
)

a = Analysis(
    ['src/proxima/__main__.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pytest', 'mkdocs'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

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
)
```

#### Build Commands

```bash
# Build standalone executable
pyinstaller proxima.spec

# Output: dist/proxima (or dist/proxima.exe on Windows)
```

### Platform-Specific Builds

#### Windows

```bash
# Build Windows executable
pyinstaller --onefile --name proxima src/proxima/__main__.py

# Create installer with NSIS (optional)
# Requires NSIS installed
makensis installer.nsi
```

#### macOS

```bash
# Build macOS executable
pyinstaller --onefile --name proxima src/proxima/__main__.py

# Create .app bundle (optional)
pyinstaller --onefile --windowed --name Proxima src/proxima/__main__.py
```

#### Linux

```bash
# Build Linux executable
pyinstaller --onefile --name proxima src/proxima/__main__.py

# Create AppImage (recommended for distribution)
./scripts/build-appimage.sh
```

## Docker Distribution

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install Proxima
RUN pip install --no-cache-dir .

# Create non-root user
RUN useradd --create-home appuser
USER appuser

# Set entrypoint
ENTRYPOINT ["proxima"]
CMD ["--help"]
```

### Build and Push

```bash
# Build image
docker build -t proxima:latest .

# Tag for registry
docker tag proxima:latest ghcr.io/proxima-project/proxima:latest

# Push to registry
docker push ghcr.io/proxima-project/proxima:latest
```

### Multi-Architecture Build

```bash
# Set up buildx
docker buildx create --use

# Build for multiple architectures
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    -t ghcr.io/proxima-project/proxima:latest \
    --push .
```

### Usage

```bash
# Run Proxima in Docker
docker run --rm -v $(pwd):/data proxima run /data/circuits/bell.json

# Interactive mode
docker run -it --rm proxima tui
```

## Conda Package

### Create conda recipe

`conda-recipe/meta.yaml`:

```yaml
package:
  name: proxima-agent
  version: "0.1.0"

source:
  path: ..

build:
  number: 0
  script: pip install . -vv
  entry_points:
    - proxima = proxima.cli.main:app

requirements:
  host:
    - python >=3.11
    - pip
    - setuptools
  run:
    - python >=3.11
    - typer >=0.9.0
    - pydantic >=2.5.0
    - structlog >=23.2.0
    - psutil >=5.9.0
    - httpx >=0.25.0
    - pandas >=2.1.0
    - pyyaml >=6.0

test:
  imports:
    - proxima
  commands:
    - proxima --help

about:
  home: https://github.com/proxima-project/proxima
  license: MIT
  summary: Intelligent Quantum Simulation Orchestration Framework
```

### Build Conda Package

```bash
# Build package
conda build conda-recipe

# Upload to Anaconda Cloud
anaconda upload /path/to/proxima-agent-0.1.0.tar.bz2
```

## Release Checklist

Before each release:

### 1. Update Version

```bash
# Update version in pyproject.toml
[project]
version = "0.1.0"

# Update version in __init__.py
__version__ = "0.1.0"
```

### 2. Update Changelog

```markdown
## [0.1.0] - 2025-01-16

### Added
- Initial release
- Support for 6 quantum backends
- LLM integration
- CLI and TUI interfaces
```

### 3. Run Tests

```bash
# Full test suite
pytest --cov=proxima

# All quality checks
pre-commit run --all-files
```

### 4. Build and Test

```bash
# Build
python -m build

# Test installation
pip install dist/proxima-*.whl
proxima --version
```

### 5. Tag Release

```bash
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

### 6. Publish

```bash
# PyPI
twine upload dist/*

# Docker
docker buildx build --push -t ghcr.io/proxima-project/proxima:0.1.0 .
```

## CI/CD Integration

### GitHub Actions

`.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install build twine
      
      - name: Build package
        run: python -m build
      
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
        run: twine upload dist/*

  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Login to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.ref_name }}
```

## Troubleshooting

### Common Build Issues

**Missing dependencies:**
```bash
# Ensure all dependencies are installed
pip install -e ".[dev,all]"
```

**Import errors in standalone build:**
```bash
# Add hidden imports to spec file
hiddenimports=['missing_module']
```

**Large executable size:**
```bash
# Exclude unnecessary packages
excludes=['pytest', 'mkdocs', 'sphinx']
```

### Verification

```bash
# Verify installed package
proxima --version
proxima backends list

# Check package metadata
pip show proxima-agent
```

## See Also

- [Contributing Guide](contributing.md)
- [Testing Guide](testing.md)
- [Deployment Guide](deployment.md)
