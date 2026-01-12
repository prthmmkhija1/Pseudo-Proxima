"""Backend registry for managing available backends."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from proxima.backends.base import BaseBackendAdapter, Capabilities
from proxima.backends.cirq_adapter import CirqBackendAdapter
from proxima.backends.cuquantum_adapter import CuQuantumAdapter
from proxima.backends.lret import LRETBackendAdapter
from proxima.backends.qiskit_adapter import QiskitBackendAdapter
from proxima.backends.qsim_adapter import QsimAdapter  # Step 3.5: qsim added
from proxima.backends.quest_adapter import QuestBackendAdapter


@dataclass
class BackendStatus:
    name: str
    available: bool
    adapter: BaseBackendAdapter | None = None
    capabilities: Capabilities | None = None
    version: str | None = None
    reason: str | None = None


class BackendRegistry:
    """Maintains discovery and lookup of backend adapters."""

    def __init__(self) -> None:
        self._statuses: dict[str, BackendStatus] = {}

    def register(self, adapter: BaseBackendAdapter) -> None:
        name = adapter.get_name()
        capabilities = adapter.get_capabilities()
        version = self._safe_get_version(adapter)
        self._statuses[name] = BackendStatus(
            name=name,
            available=True,
            adapter=adapter,
            capabilities=capabilities,
            version=version,
            reason=None,
        )

    def discover(self) -> None:
        """Discover known backends, cache capabilities, and mark health status."""

        self._statuses = {}
        candidates: list[type[BaseBackendAdapter]] = [
            LRETBackendAdapter,
            CirqBackendAdapter,
            QiskitBackendAdapter,
            QuestBackendAdapter,
            CuQuantumAdapter,  # Step 2.3: cuQuantum added to registry
            QsimAdapter,  # Step 3.5: qsim added to registry
        ]

        for adapter_cls in candidates:
            status = self._init_adapter(adapter_cls)
            self._statuses[status.name] = status

    def _init_adapter(self, adapter_cls: type[BaseBackendAdapter]) -> BackendStatus:
        name = getattr(adapter_cls, "__name__", "unknown").lower()

        try:
            adapter = adapter_cls()
            name = adapter.get_name()
        except Exception as exc:  # pragma: no cover - defensive
            return BackendStatus(
                name=name,
                available=False,
                adapter=None,
                capabilities=None,
                version=None,
                reason=f"initialization failed: {exc}",
            )

        try:
            if not adapter.is_available():
                return BackendStatus(
                    name=name,
                    available=False,
                    adapter=None,
                    capabilities=None,
                    version=None,
                    reason=self._dependency_reason(adapter_cls),
                )
        except Exception as exc:  # pragma: no cover - defensive
            return BackendStatus(
                name=name,
                available=False,
                adapter=None,
                capabilities=None,
                version=None,
                reason=f"availability check failed: {exc}",
            )

        try:
            capabilities = adapter.get_capabilities()
        except Exception as exc:  # pragma: no cover - defensive
            return BackendStatus(
                name=name,
                available=False,
                adapter=None,
                capabilities=None,
                version=None,
                reason=f"capabilities check failed: {exc}",
            )

        version = self._safe_get_version(adapter)
        return BackendStatus(
            name=name,
            available=True,
            adapter=adapter,
            capabilities=capabilities,
            version=version,
            reason=None,
        )

    def _dependency_reason(self, adapter_cls: type[BaseBackendAdapter]) -> str:
        dependency_map = {
            "lretbackendadapter": ["lret"],
            "cirqbackendadapter": ["cirq"],
            "qiskitbackendadapter": ["qiskit", "qiskit_aer"],
            "questbackendadapter": ["pyQuEST"],
            "cuquantumadapter": [
                "qiskit",
                "qiskit_aer",
                "cuquantum",
            ],  # cuQuantum dependencies
            "qsimadapter": ["cirq", "qsimcirq"],  # Step 3.5: qsim dependencies
        }
        missing = []
        for dep in dependency_map.get(adapter_cls.__name__.lower(), []):
            if importlib.util.find_spec(dep) is None:
                missing.append(dep)
        if missing:
            return f"missing dependency: {', '.join(missing)}"
        return "adapter reported unavailable"

    def _safe_get_version(self, adapter: BaseBackendAdapter) -> str:
        try:
            return adapter.get_version()
        except Exception as exc:  # pragma: no cover - defensive
            return f"unknown (version check failed: {exc})"

    def get(self, name: str) -> BaseBackendAdapter:
        status = self._statuses.get(name)
        if not status:
            raise KeyError(f"Backend '{name}' not registered")
        if not status.available or not status.adapter:
            raise KeyError(
                f"Backend '{name}' is unavailable: {status.reason or 'unknown reason'}"
            )
        return status.adapter

    def is_available(self, name: str) -> bool:
        status = self._statuses.get(name)
        return bool(status and status.available)

    def list_available(self) -> list[str]:
        return [name for name, status in self._statuses.items() if status.available]

    def list_statuses(self) -> list[BackendStatus]:
        return list(self._statuses.values())

    def get_capabilities(self, name: str) -> Capabilities:
        return self.get(name).get_capabilities()

    def get_status(self, name: str) -> BackendStatus:
        status = self._statuses.get(name)
        if not status:
            raise KeyError(f"Backend '{name}' not registered")
        return status

    # ==========================================================================
    # Step 2.3: GPU-aware backend selection helpers
    # ==========================================================================

    def get_gpu_backends(self) -> list[str]:
        """Return list of GPU-enabled backends."""
        gpu_backends = []
        for name, status in self._statuses.items():
            if status.available and status.capabilities:
                if status.capabilities.supports_gpu:
                    gpu_backends.append(name)
        return gpu_backends

    def get_best_backend_for_circuit(
        self,
        qubit_count: int,
        simulation_type: str = "state_vector",
        prefer_gpu: bool = True,
    ) -> str | None:
        """Get best available backend for given circuit requirements.

        Args:
            qubit_count: Number of qubits in circuit
            simulation_type: "state_vector" or "density_matrix"
            prefer_gpu: Whether to prefer GPU backends

        Returns:
            Name of best backend, or None if none suitable
        """
        # Priority order based on simulation type and GPU preference
        # Step 3.5: qsim included in priority lists
        if simulation_type == "state_vector":
            if prefer_gpu:
                priority = ["cuquantum", "quest", "qsim", "qiskit", "cirq"]
            else:
                priority = ["qsim", "quest", "qiskit", "cirq"]
        elif simulation_type == "density_matrix":
            priority = ["quest", "cirq", "qiskit", "lret"]
        else:
            priority = ["qsim", "qiskit", "cirq", "quest"]

        for backend_name in priority:
            status = self._statuses.get(backend_name)
            if status and status.available and status.capabilities:
                if qubit_count <= status.capabilities.max_qubits:
                    return backend_name

        return None


backend_registry = BackendRegistry()
backend_registry.discover()
