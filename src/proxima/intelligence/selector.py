"""Backend auto-selection logic (Step 3.3).

Implements a simple scoring-based selector that:
- Filters to available backends that satisfy required capabilities.
- Scores candidates on feature match, performance, memory efficiency, and
  user-history prior (tracks successful executions).
- Returns the top backend with an explanation and per-factor scores.

This is intentionally lightweight and deterministic; it can be swapped for a
data-driven model later without changing the public interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from proxima.backends.base import BaseBackendAdapter, Capabilities, ResourceEstimate, SimulatorType
from proxima.backends.registry import backend_registry


# Simple history store for user preferences
_HISTORY_FILE = Path.home() / ".proxima" / "backend_history.json"


def _load_history() -> dict[str, dict[str, int]]:
    """Load execution history from disk."""
    if _HISTORY_FILE.exists():
        try:
            with open(_HISTORY_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"success_count": {}, "fail_count": {}}


def _save_history(history: dict[str, dict[str, int]]) -> None:
    """Persist execution history to disk."""
    try:
        _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_HISTORY_FILE, "w") as f:
            json.dump(history, f)
    except OSError:
        pass  # Best effort


def record_execution(backend: str, success: bool) -> None:
    """Record a backend execution result for history-based scoring.

    Args:
        backend: Name of the backend
        success: Whether the execution succeeded
    """
    history = _load_history()
    key = "success_count" if success else "fail_count"
    history[key][backend] = history[key].get(backend, 0) + 1
    _save_history(history)


@dataclass
class SelectionInput:
    """Input for backend selection with circuit characteristics.

    Implements the circuit characteristic extraction from Step 3.3:
    - qubit_count: Number of qubits in the circuit
    - gate_types: List of gate types used (e.g., ["H", "CNOT", "RZ"])
    - circuit_depth: Depth of the quantum circuit
    - has_measurements: Whether the circuit includes measurements
    - needs_noise: Whether noise simulation is required
    """

    qubit_count: int
    needs_noise: bool = False
    simulator_type: SimulatorType | None = None
    preferred_backend: str | None = None
    # Extended circuit characteristics per Step 3.3
    gate_types: list[str] = field(default_factory=list)
    circuit_depth: int = 0
    has_measurements: bool = False
    circuit_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendScore:
    name: str
    feature_score: float
    performance_score: float
    memory_score: float
    history_score: float
    total: float
    estimate: ResourceEstimate | None
    explanation: list[str]


@dataclass
class SelectionResult:
    backend: str
    scores: BackendScore
    tried: list[BackendScore]


class BackendSelector:
    FEATURE_WEIGHT = 0.4
    PERFORMANCE_WEIGHT = 0.3
    MEMORY_WEIGHT = 0.2
    HISTORY_WEIGHT = 0.1

    def __init__(self) -> None:
        self.registry = backend_registry

    def select(self, request: SelectionInput) -> SelectionResult:
        candidates = self._candidate_statuses()

        filtered: list[tuple[str, BaseBackendAdapter, Capabilities]] = []
        for name, adapter, caps in candidates:
            if not self._compatible(caps, request):
                continue
            filtered.append((name, adapter, caps))

        if not filtered:
            raise ValueError("No compatible backends available for the requested circuit.")

        scores = [self._score(name, adapter, caps, request) for name, adapter, caps in filtered]
        scores.sort(key=lambda s: s.total, reverse=True)

        winner = scores[0]
        return SelectionResult(backend=winner.name, scores=winner, tried=scores)

    def _candidate_statuses(self) -> list[tuple[str, BaseBackendAdapter, Capabilities]]:
        result: list[tuple[str, BaseBackendAdapter, Capabilities]] = []
        for status in self.registry.list_statuses():
            if not status.available or not status.adapter or not status.capabilities:
                continue
            result.append((status.name, status.adapter, status.capabilities))
        return result

    def _compatible(self, caps: Capabilities, req: SelectionInput) -> bool:
        if req.simulator_type and req.simulator_type not in caps.simulator_types:
            return False
        if caps.max_qubits and req.qubit_count > caps.max_qubits:
            return False
        if req.needs_noise and not caps.supports_noise:
            return False
        return True

    def _score(
        self,
        name: str,
        adapter: BaseBackendAdapter,
        caps: Capabilities,
        req: SelectionInput,
    ) -> BackendScore:
        explanation: list[str] = []

        # Feature score: 1.0 if all required are met, else 0.5 fallback
        feature_score = 1.0
        if req.simulator_type and req.simulator_type not in caps.simulator_types:
            feature_score = 0.0
        if req.needs_noise and not caps.supports_noise:
            feature_score = 0.0
        explanation.append(f"Feature match: {feature_score:.2f}")

        # Resource estimate for performance/memory scoring
        estimate: ResourceEstimate | None = None
        try:
            estimate = adapter.estimate_resources(
                req.circuit_metadata or {"qubits": req.qubit_count}
            )
        except Exception:
            estimate = None

        performance_score = self._performance_score(estimate)
        memory_score = self._memory_score(estimate)
        history_score = self._history_score(name)

        total = (
            feature_score * self.FEATURE_WEIGHT
            + performance_score * self.PERFORMANCE_WEIGHT
            + memory_score * self.MEMORY_WEIGHT
            + history_score * self.HISTORY_WEIGHT
        )

        explanation.append(f"Performance: {performance_score:.2f}")
        explanation.append(f"Memory: {memory_score:.2f}")
        explanation.append(f"History: {history_score:.2f}")

        if req.preferred_backend and name == req.preferred_backend:
            total += 0.05
            explanation.append("Preferred backend bonus: +0.05")

        return BackendScore(
            name=name,
            feature_score=feature_score,
            performance_score=performance_score,
            memory_score=memory_score,
            history_score=history_score,
            total=total,
            estimate=estimate,
            explanation=explanation,
        )

    def _performance_score(self, estimate: ResourceEstimate | None) -> float:
        if not estimate or estimate.time_ms is None:
            return 0.5
        # Lower time -> higher score; cap at a reasonable range
        if estimate.time_ms <= 100:
            return 1.0
        if estimate.time_ms >= 5000:
            return 0.1
        return max(0.1, 1.0 - (estimate.time_ms / 5000))

    def _memory_score(self, estimate: ResourceEstimate | None) -> float:
        if not estimate or estimate.memory_mb is None:
            return 0.5
        if estimate.memory_mb <= 512:
            return 1.0
        if estimate.memory_mb >= 8192:
            return 0.1
        return max(0.1, 1.0 - (estimate.memory_mb / 8192))

    def _history_score(self, backend: str) -> float:
        """Calculate history score based on past execution success rate.

        Returns a score between 0.0 and 1.0:
        - 1.0: High success rate (>90% success)
        - 0.5: No history or neutral
        - 0.0: High failure rate (>90% failure)
        """
        history = _load_history()
        successes = history["success_count"].get(backend, 0)
        failures = history["fail_count"].get(backend, 0)
        total = successes + failures

        if total == 0:
            return 0.5  # No history, neutral score

        success_rate = successes / total
        # Scale from 0.0-1.0 based on success rate
        return success_rate


selector = BackendSelector()
