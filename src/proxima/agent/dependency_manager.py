"""Project-level dependency detection, installation, and build error resolution.

Phase 5: Dependency Management System
--------------------------------------
This module provides the ``ProjectDependencyManager`` class, which scans
project directories for dependency specification files, installs packages
via the appropriate package manager, checks whether individual packages
are installed, and diagnoses common build / import errors with suggested
fixes.

The class is consumed by:
* ``IntentToolBridge`` — for the ``INSTALL_DEPENDENCY``,
  ``CHECK_DEPENDENCY``, and ``CONFIGURE_ENVIRONMENT`` intents (Step 5.2).
* ``IntentToolBridge.dispatch()`` — for the automatic error-detection
  loop that runs after any failed ``RUN_COMMAND`` / ``RUN_SCRIPT``
  (Step 5.3).
* Backend build pre-check logic (Step 5.4).

Note
----
A separate class named ``DependencyManager`` already exists in
``src/proxima/agent/dynamic_tools/deployment_monitoring.py``.
That class handles *security auditing* (vulnerability scanning, licence
compliance, dependency pinning).  This ``ProjectDependencyManager``
focuses on *project-level* concerns: detection, installation, and
build-error auto-fix.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from configparser import ConfigParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── tomllib import (stdlib ≥3.11, tomli fallback for 3.10) ──────────
try:
    import tomllib  # type: ignore[import-untyped]
except ModuleNotFoundError:  # pragma: no cover — Python <3.11
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

# ── Optional yaml import ────────────────────────────────────────────
try:
    import yaml  # type: ignore[import-untyped]
except ModuleNotFoundError:
    yaml = None  # type: ignore[assignment]


# ═══════════════════════════════════════════════════════════════════════
# Build-error patterns
# ═══════════════════════════════════════════════════════════════════════

# Each tuple: (compiled regex, human description, optional fix command template)
# The fix template may contain ``{pkg}`` for a captured group.
_ERROR_PATTERNS: List[Tuple[re.Pattern, str, Optional[str]]] = [
    (
        re.compile(
            r"ModuleNotFoundError:\s*No module named ['\"]?(\w[\w.]*)['\"]*",
            re.IGNORECASE,
        ),
        "Missing Python module: {pkg}",
        "pip install {pkg}",
    ),
    (
        re.compile(
            r"ImportError:\s*cannot import name .+ from ['\"]?(\w[\w.]*)['\"]*",
            re.IGNORECASE,
        ),
        "Import error — possibly outdated or missing package: {pkg}",
        "pip install --upgrade {pkg}",
    ),
    (
        re.compile(
            r"error:\s*Microsoft Visual C\+\+ 14\.0 or greater is required",
            re.IGNORECASE,
        ),
        "Missing Visual Studio Build Tools",
        None,  # No single pip command can fix this
    ),
    (
        re.compile(r"CMake Error", re.IGNORECASE),
        "CMake is required but not available or misconfigured",
        "pip install cmake",
    ),
    (
        re.compile(
            r"fatal error:\s*['\"]?(\S+\.h)['\"]?\s*file not found",
            re.IGNORECASE,
        ),
        "Missing C/C++ header file: {pkg}",
        None,
    ),
    (
        re.compile(r"CUDA (?:error|runtime)|nvcc\s+not found", re.IGNORECASE),
        "CUDA toolkit is required but not installed or not on PATH",
        None,
    ),
    (
        re.compile(r"Permission denied", re.IGNORECASE),
        "Insufficient permissions — try running with elevated privileges",
        None,
    ),
    (
        re.compile(
            r"ERROR:\s*Could not find a version that satisfies the requirement (\S+)",
            re.IGNORECASE,
        ),
        "Package not found in PyPI: {pkg}",
        None,
    ),
    (
        re.compile(
            r"ERROR:\s*No matching distribution found for (\S+)",
            re.IGNORECASE,
        ),
        "No distribution found for: {pkg}",
        None,
    ),
    (
        re.compile(r"npm ERR!", re.IGNORECASE),
        "npm encountered an error — check package.json",
        "npm install",
    ),
    (
        re.compile(r"cargo error|error\[E\d+\]", re.IGNORECASE),
        "Rust/Cargo compilation error",
        None,
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# Manager class
# ═══════════════════════════════════════════════════════════════════════

class ProjectDependencyManager:
    """Detect, install, and troubleshoot project dependencies.

    This class is **stateless** — every public method takes explicit
    parameters and returns a result.  It is safe to instantiate once and
    reuse across an entire session.

    Parameters
    ----------
    python_executable : str, optional
        Path to the Python interpreter to use for ``pip`` calls.
        Defaults to ``sys.executable``.
    """

    def __init__(self, python_executable: Optional[str] = None) -> None:
        self._python = python_executable or sys.executable
        # Cache for batch-checked packages: {normalised_name: (installed, version)}
        self._pip_cache: Optional[Dict[str, Tuple[bool, Optional[str]]]] = None
        self._pip_cache_time: float = 0.0

    @property
    def python_executable(self) -> str:
        """Return the Python interpreter path used for pip calls."""
        return self._python

    # ── 5.1-a: Detect project dependencies ────────────────────

    def detect_project_dependencies(
        self, project_path: str
    ) -> Dict[str, Any]:
        """Scan *project_path* for dependency specification files.

        Returns a dictionary with keys:
        * ``python_packages``  — ``List[str]``
        * ``node_packages``    — ``Dict[str, str]`` (name → version spec)
        * ``system_packages``  — ``List[str]``
        * ``source_file``      — ``Optional[str]`` (which file was found)
        * ``detected_manager`` — ``Optional[str]`` (pip/conda/npm/yarn/pipenv)
        """
        root = Path(project_path)
        result: Dict[str, Any] = {
            "python_packages": [],
            "node_packages": {},
            "system_packages": [],
            "source_file": None,
            "detected_manager": None,
        }

        if not root.is_dir():
            logger.warning("Project path does not exist: %s", project_path)
            return result

        # Priority-ordered checks.  First match sets ``detected_manager``.
        # Multiple files can *contribute* packages though.

        # 1. requirements.txt
        req_txt = root / "requirements.txt"
        if req_txt.is_file():
            pkgs = self._parse_requirements_txt(req_txt)
            result["python_packages"].extend(pkgs)
            if result["source_file"] is None:
                result["source_file"] = "requirements.txt"
            if result["detected_manager"] is None:
                result["detected_manager"] = "pip"

        # 2. setup.py
        setup_py = root / "setup.py"
        if setup_py.is_file():
            pkgs = self._parse_setup_py(setup_py)
            result["python_packages"].extend(pkgs)
            if result["source_file"] is None:
                result["source_file"] = "setup.py"
            if result["detected_manager"] is None:
                result["detected_manager"] = "pip"

        # 3. pyproject.toml
        pyproject = root / "pyproject.toml"
        if pyproject.is_file() and tomllib is not None:
            pkgs, extras = self._parse_pyproject_toml(pyproject)
            result["python_packages"].extend(pkgs)
            for extra_name, extra_pkgs in extras.items():
                result["python_packages"].extend(extra_pkgs)
            if result["source_file"] is None:
                result["source_file"] = "pyproject.toml"
            if result["detected_manager"] is None:
                result["detected_manager"] = "pip"

        # 4. setup.cfg
        setup_cfg = root / "setup.cfg"
        if setup_cfg.is_file():
            pkgs = self._parse_setup_cfg(setup_cfg)
            result["python_packages"].extend(pkgs)
            if result["source_file"] is None:
                result["source_file"] = "setup.cfg"
            if result["detected_manager"] is None:
                result["detected_manager"] = "pip"

        # 5. package.json (Node)
        package_json = root / "package.json"
        if package_json.is_file():
            deps, dev_deps = self._parse_package_json(package_json)
            result["node_packages"].update(deps)
            result["node_packages"].update(dev_deps)
            if result["source_file"] is None:
                result["source_file"] = "package.json"
            if result["detected_manager"] is None:
                # Prefer yarn if lock-file exists
                if (root / "yarn.lock").is_file():
                    result["detected_manager"] = "yarn"
                else:
                    result["detected_manager"] = "npm"

        # 6. Pipfile
        pipfile = root / "Pipfile"
        if pipfile.is_file() and tomllib is not None:
            pkgs = self._parse_pipfile(pipfile)
            result["python_packages"].extend(pkgs)
            if result["source_file"] is None:
                result["source_file"] = "Pipfile"
            if result["detected_manager"] is None:
                result["detected_manager"] = "pipenv"

        # 7. environment.yml (Conda)
        env_yml = root / "environment.yml"
        if env_yml.is_file() and yaml is not None:
            pkgs = self._parse_environment_yml(env_yml)
            result["python_packages"].extend(pkgs)
            if result["source_file"] is None:
                result["source_file"] = "environment.yml"
            if result["detected_manager"] is None:
                result["detected_manager"] = "conda"

        # Deduplicate while preserving order
        result["python_packages"] = list(dict.fromkeys(result["python_packages"]))

        return result

    # ── 5.1-b: Install dependencies ───────────────────────────

    def install_dependencies(
        self,
        project_path: str,
        packages: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """Install dependencies for the project at *project_path*.

        If *packages* is provided explicitly, install those via ``pip``.
        Otherwise, detect the dependency file and run the appropriate
        manager command.

        Returns ``(success, output_text)``.
        """
        cwd = project_path
        if packages:
            cmd = [self._python, "-m", "pip", "install"] + packages
        else:
            info = self.detect_project_dependencies(project_path)
            mgr = info.get("detected_manager")
            source = info.get("source_file")

            if mgr == "pipenv":
                cmd = ["pipenv", "install"]
            elif mgr == "conda" and source == "environment.yml":
                cmd = ["conda", "env", "create", "-f", "environment.yml"]
            elif mgr in ("npm", "yarn"):
                cmd = [mgr, "install"]
            elif source == "requirements.txt":
                cmd = [self._python, "-m", "pip", "install", "-r", "requirements.txt"]
            elif source in ("setup.py", "pyproject.toml"):
                cmd = [self._python, "-m", "pip", "install", "-e", "."]
            elif info["python_packages"]:
                cmd = [self._python, "-m", "pip", "install"] + info["python_packages"]
            else:
                return False, "No dependency file detected and no packages specified."

        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            return proc.returncode == 0, output.strip()
        except FileNotFoundError as exc:
            return False, f"Command not found: {exc}"
        except subprocess.TimeoutExpired:
            return False, "Installation timed out after 300 seconds."
        except Exception as exc:
            return False, f"Installation failed: {exc}"

    # ── 5.1-c: Check a single package ─────────────────────────

    def check_installed(
        self, package_name: str
    ) -> Tuple[bool, Optional[str]]:
        """Check whether *package_name* is installed via ``pip show``.

        Returns ``(is_installed, version_or_none)``.
        """
        # Try batch cache first (populated by check_installed_batch)
        if self._pip_cache is not None:
            norm = self._normalise_pip_name(package_name)
            if norm in self._pip_cache:
                return self._pip_cache[norm]
        try:
            proc = subprocess.run(
                [self._python, "-m", "pip", "show", package_name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode != 0:
                return False, None
            # Parse version from output
            for line in proc.stdout.splitlines():
                if line.lower().startswith("version:"):
                    return True, line.split(":", 1)[1].strip()
            return True, None  # installed but version not parsed
        except Exception:
            return False, None

    def _refresh_pip_cache(self) -> None:
        """Populate ``_pip_cache`` from ``pip list --format=json``.

        Calling this once replaces N individual ``pip show`` invocations
        with a single subprocess call.  The cache is invalidated after
        60 seconds to stay reasonably fresh.
        """
        import time as _time

        now = _time.monotonic()
        if self._pip_cache is not None and (now - self._pip_cache_time) < 60:
            return  # still fresh

        cache: Dict[str, Tuple[bool, Optional[str]]] = {}
        try:
            proc = subprocess.run(
                [self._python, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode == 0 and proc.stdout:
                for entry in json.loads(proc.stdout):
                    name = self._normalise_pip_name(entry.get("name", ""))
                    version = entry.get("version")
                    cache[name] = (True, version)
        except Exception:
            pass  # cache remains empty — individual checks will work
        self._pip_cache = cache
        self._pip_cache_time = now

    def check_installed_batch(
        self,
        package_names: List[str],
    ) -> Dict[str, Tuple[bool, Optional[str]]]:
        """Check many packages at once using a single ``pip list`` call.

        Returns a dict mapping each *package_name* to
        ``(is_installed, version_or_none)``.
        """
        self._refresh_pip_cache()

        results: Dict[str, Tuple[bool, Optional[str]]] = {}
        for name in package_names:
            norm = self._normalise_pip_name(name)
            results[name] = self._pip_cache.get(norm, (False, None))
        return results

    @staticmethod
    def _normalise_pip_name(name: str) -> str:
        """Normalise a package name for cache lookup (PEP 503)."""
        return re.sub(r"[-_.]+", "-", name).lower().strip()

    # ── 5.1-d: Error detection and fix suggestion ─────────────

    def detect_and_fix_errors(
        self, error_output: str, project_path: str
    ) -> Optional[str]:
        """Analyse *error_output* and return a fix command if one can be inferred.

        Checks against a curated list of common build / import error
        patterns.  Returns ``None`` when no auto-fix is available.
        """
        if not error_output:
            return None

        for pattern, _description, fix_template in _ERROR_PATTERNS:
            m = pattern.search(error_output)
            if m:
                if fix_template is None:
                    continue  # pattern recognised, but no auto-fix
                # Substitute captured group if present
                pkg = m.group(1) if m.lastindex and m.lastindex >= 1 else ""
                # Normalise common module-to-package name mismatches
                pkg = self._normalise_package_name(pkg)
                return fix_template.format(pkg=pkg)

        return None

    def describe_errors(
        self, error_output: str
    ) -> List[Dict[str, Optional[str]]]:
        """Return *all* recognised error descriptions (not just the first).

        Each entry has keys ``description`` and ``fix`` (may be ``None``).
        """
        results: List[Dict[str, Optional[str]]] = []
        if not error_output:
            return results
        for pattern, description, fix_template in _ERROR_PATTERNS:
            m = pattern.search(error_output)
            if m:
                pkg = m.group(1) if m.lastindex and m.lastindex >= 1 else ""
                pkg = self._normalise_package_name(pkg)
                results.append({
                    "description": description.format(pkg=pkg),
                    "fix": fix_template.format(pkg=pkg) if fix_template else None,
                })
        return results

    # ── 5.4: Backend dependency pre-check ─────────────────────

    def check_backend_dependencies(
        self,
        packages: List[str],
    ) -> List[Dict[str, Any]]:
        """Check installation status for each package in *packages*.

        Uses batch checking (single ``pip list`` call) for efficiency.

        Returns a list of dicts with keys ``package``, ``installed``,
        ``version``, ``required`` (the raw specifier string).
        """
        # Extract bare names for batch lookup
        names: List[str] = []
        for spec in packages:
            name = re.split(r"[><=!~;]", spec, maxsplit=1)[0].strip()
            names.append(name)

        batch = self.check_installed_batch(names)

        results: List[Dict[str, Any]] = []
        for spec, name in zip(packages, names):
            installed, version = batch.get(name, (False, None))
            results.append({
                "package": name,
                "installed": installed,
                "version": version,
                "required": spec,
            })
        return results

    def get_missing_packages(
        self,
        packages: List[str],
    ) -> List[str]:
        """Return the subset of *packages* that are not currently installed.

        Uses batch checking for efficiency.
        """
        names = [
            re.split(r"[><=!~;]", spec, maxsplit=1)[0].strip()
            for spec in packages
        ]
        batch = self.check_installed_batch(names)
        return [
            spec
            for spec, name in zip(packages, names)
            if not batch.get(name, (False, None))[0]
        ]

    # ── Private helpers ───────────────────────────────────────

    @staticmethod
    def _parse_requirements_txt(path: Path) -> List[str]:
        """Parse package names from a pip ``requirements.txt``."""
        packages: List[str] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("-"):
                    continue
                # Remove inline comments
                line = line.split("#", 1)[0].strip()
                if line:
                    packages.append(line)
        except Exception as exc:
            logger.debug("Failed to parse %s: %s", path, exc)
        return packages

    @staticmethod
    def _parse_setup_py(path: Path) -> List[str]:
        """Extract ``install_requires`` from a ``setup.py`` via regex."""
        packages: List[str] = []
        try:
            content = path.read_text(encoding="utf-8")
            # Match install_requires=[...] across multiple lines
            m = re.search(
                r"install_requires\s*=\s*\[(.*?)\]",
                content,
                re.DOTALL,
            )
            if m:
                raw = m.group(1)
                for item in re.findall(r"""['"]([^'"]+)['"]""", raw):
                    packages.append(item.strip())
        except Exception as exc:
            logger.debug("Failed to parse %s: %s", path, exc)
        return packages

    @staticmethod
    def _parse_pyproject_toml(
        path: Path,
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        """Parse ``[project.dependencies]`` and ``[project.optional-dependencies]``."""
        deps: List[str] = []
        extras: Dict[str, List[str]] = {}
        if tomllib is None:
            return deps, extras
        try:
            with open(path, "rb") as fh:
                data = tomllib.load(fh)
            project = data.get("project", {})
            deps = [str(d) for d in project.get("dependencies", [])]
            for extra_name, extra_list in project.get(
                "optional-dependencies", {}
            ).items():
                extras[extra_name] = [str(d) for d in extra_list]
        except Exception as exc:
            logger.debug("Failed to parse %s: %s", path, exc)
        return deps, extras

    @staticmethod
    def _parse_setup_cfg(path: Path) -> List[str]:
        """Parse ``install_requires`` from ``setup.cfg``."""
        packages: List[str] = []
        try:
            cfg = ConfigParser()
            cfg.read(str(path), encoding="utf-8")
            raw = cfg.get("options", "install_requires", fallback="")
            for line in raw.strip().splitlines():
                line = line.strip()
                if line:
                    packages.append(line)
        except Exception as exc:
            logger.debug("Failed to parse %s: %s", path, exc)
        return packages

    @staticmethod
    def _parse_package_json(
        path: Path,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Parse ``dependencies`` and ``devDependencies`` from ``package.json``."""
        deps: Dict[str, str] = {}
        dev_deps: Dict[str, str] = {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            deps = data.get("dependencies", {})
            dev_deps = data.get("devDependencies", {})
        except Exception as exc:
            logger.debug("Failed to parse %s: %s", path, exc)
        return deps, dev_deps

    @staticmethod
    def _parse_pipfile(path: Path) -> List[str]:
        """Parse ``[packages]`` from a ``Pipfile``."""
        packages: List[str] = []
        if tomllib is None:
            return packages
        try:
            with open(path, "rb") as fh:
                data = tomllib.load(fh)
            for name in data.get("packages", {}):
                packages.append(name)
        except Exception as exc:
            logger.debug("Failed to parse %s: %s", path, exc)
        return packages

    @staticmethod
    def _parse_environment_yml(path: Path) -> List[str]:
        """Parse ``dependencies`` from an ``environment.yml``."""
        packages: List[str] = []
        if yaml is None:
            return packages
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            for item in data.get("dependencies", []):
                if isinstance(item, str):
                    packages.append(item)
                elif isinstance(item, dict):
                    # conda sometimes nests pip deps: {pip: [...]}
                    for sub_list in item.values():
                        if isinstance(sub_list, list):
                            packages.extend(str(s) for s in sub_list)
        except Exception as exc:
            logger.debug("Failed to parse %s: %s", path, exc)
        return packages

    # ── Package-name normalisation ────────────────────────────

    # Common import-name → PyPI-package-name mismatches
    _IMPORT_TO_PACKAGE: Dict[str, str] = {
        "cv2": "opencv-python",
        "PIL": "Pillow",
        "sklearn": "scikit-learn",
        "yaml": "pyyaml",
        "bs4": "beautifulsoup4",
        "attr": "attrs",
        "dateutil": "python-dateutil",
        "gi": "PyGObject",
        "lxml": "lxml",
        "serial": "pyserial",
        "usb": "pyusb",
        "wx": "wxPython",
        "Crypto": "pycryptodome",
        "dotenv": "python-dotenv",
        "jose": "python-jose",
        "jwt": "PyJWT",
        "magic": "python-magic",
        "docx": "python-docx",
        "pptx": "python-pptx",
        "toml": "tomli",
    }

    @classmethod
    def _normalise_package_name(cls, name: str) -> str:
        """Map an import-module name to the correct PyPI package name."""
        top_level = name.split(".")[0]
        return cls._IMPORT_TO_PACKAGE.get(top_level, top_level)
