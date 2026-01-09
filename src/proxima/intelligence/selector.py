"""Backend auto-selection logic (Step 3.3).

Implements a simple scoring-based selector that:
- Filters to available backends that satisfy required capabilities.
- Scores candidates on feature match, performance, memory efficiency, and a small
  user-history prior (currently neutral).
- Returns the top backend with an explanation and per-factor scores.

This is intentionally lightweight and deterministic; it can be swapped for a
data-driven model later without changing the public interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from proxima.backends.base import BaseBackendAdapter, Capabilities, ResourceEstimate, SimulatorType
from proxima.backends.registry import backend_registry


@dataclass
class SelectionInput:
	qubit_count: int
	needs_noise: bool = False
	simulator_type: Optional[SimulatorType] = None
	preferred_backend: Optional[str] = None
	circuit_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendScore:
	name: str
	feature_score: float
	performance_score: float
	memory_score: float
	history_score: float
	total: float
	estimate: Optional[ResourceEstimate]
	explanation: List[str]


@dataclass
class SelectionResult:
	backend: str
	scores: BackendScore
	tried: List[BackendScore]


class BackendSelector:
	FEATURE_WEIGHT = 0.4
	PERFORMANCE_WEIGHT = 0.3
	MEMORY_WEIGHT = 0.2
	HISTORY_WEIGHT = 0.1

	def __init__(self) -> None:
		self.registry = backend_registry

	def select(self, request: SelectionInput) -> SelectionResult:
		candidates = self._candidate_statuses()

		filtered: List[Tuple[str, BaseBackendAdapter, Capabilities]] = []
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

	def _candidate_statuses(self) -> List[Tuple[str, BaseBackendAdapter, Capabilities]]:
		result: List[Tuple[str, BaseBackendAdapter, Capabilities]] = []
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
		explanation: List[str] = []

		# Feature score: 1.0 if all required are met, else 0.5 fallback
		feature_score = 1.0
		if req.simulator_type and req.simulator_type not in caps.simulator_types:
			feature_score = 0.0
		if req.needs_noise and not caps.supports_noise:
			feature_score = 0.0
		explanation.append(f"Feature match: {feature_score:.2f}")

		# Resource estimate for performance/memory scoring
		estimate: Optional[ResourceEstimate] = None
		try:
			estimate = adapter.estimate_resources(req.circuit_metadata or {"qubits": req.qubit_count})
		except Exception:
			estimate = None

		performance_score = self._performance_score(estimate)
		memory_score = self._memory_score(estimate)
		history_score = 0.0  # Placeholder for future user-history weighting

		total = (
			feature_score * self.FEATURE_WEIGHT
			+ performance_score * self.PERFORMANCE_WEIGHT
			+ memory_score * self.MEMORY_WEIGHT
			+ history_score * self.HISTORY_WEIGHT
		)

		explanation.append(f"Performance: {performance_score:.2f}")
		explanation.append(f"Memory: {memory_score:.2f}")
		explanation.append("History: 0.00 (not yet tracked)")

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

	def _performance_score(self, estimate: Optional[ResourceEstimate]) -> float:
		if not estimate or estimate.time_ms is None:
			return 0.5
		# Lower time -> higher score; cap at a reasonable range
		if estimate.time_ms <= 100:
			return 1.0
		if estimate.time_ms >= 5000:
			return 0.1
		return max(0.1, 1.0 - (estimate.time_ms / 5000))

	def _memory_score(self, estimate: Optional[ResourceEstimate]) -> float:
		if not estimate or estimate.memory_mb is None:
			return 0.5
		if estimate.memory_mb <= 512:
			return 1.0
		if estimate.memory_mb >= 8192:
			return 0.1
		return max(0.1, 1.0 - (estimate.memory_mb / 8192))


selector = BackendSelector()
