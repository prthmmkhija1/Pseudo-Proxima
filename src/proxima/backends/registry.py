"""Backend registry for managing available backends."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

from proxima.backends.base import BaseBackendAdapter, Capabilities
from proxima.backends.cirq_adapter import CirqBackendAdapter
from proxima.backends.lret import LRETBackendAdapter
from proxima.backends.qiskit_adapter import QiskitBackendAdapter


@dataclass
class BackendStatus:
	name: str
	available: bool
	adapter: Optional[BaseBackendAdapter] = None
	capabilities: Optional[Capabilities] = None
	version: Optional[str] = None
	reason: Optional[str] = None


class BackendRegistry:
	"""Maintains discovery and lookup of backend adapters."""

	def __init__(self) -> None:
		self._statuses: Dict[str, BackendStatus] = {}

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
		candidates: List[Type[BaseBackendAdapter]] = [
			LRETBackendAdapter,
			CirqBackendAdapter,
			QiskitBackendAdapter,
		]

		for adapter_cls in candidates:
			status = self._init_adapter(adapter_cls)
			self._statuses[status.name] = status

	def _init_adapter(self, adapter_cls: Type[BaseBackendAdapter]) -> BackendStatus:
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

	def _dependency_reason(self, adapter_cls: Type[BaseBackendAdapter]) -> str:
		dependency_map = {
			"lretbackendadapter": ["lret"],
			"cirqbackendadapter": ["cirq"],
			"qiskitbackendadapter": ["qiskit", "qiskit_aer"],
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
			raise KeyError(f"Backend '{name}' is unavailable: {status.reason or 'unknown reason'}")
		return status.adapter

	def is_available(self, name: str) -> bool:
		status = self._statuses.get(name)
		return bool(status and status.available)

	def list_available(self) -> List[str]:
		return [name for name, status in self._statuses.items() if status.available]

	def list_statuses(self) -> List[BackendStatus]:
		return list(self._statuses.values())

	def get_capabilities(self, name: str) -> Capabilities:
		return self.get(name).get_capabilities()

	def get_status(self, name: str) -> BackendStatus:
		status = self._statuses.get(name)
		if not status:
			raise KeyError(f"Backend '{name}' not registered")
		return status


backend_registry = BackendRegistry()
backend_registry.discover()
