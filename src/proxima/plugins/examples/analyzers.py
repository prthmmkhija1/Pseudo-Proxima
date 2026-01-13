"""
Example Analyzer Plugins.

Demonstrates how to create analyzer plugins for result analysis.
"""

from __future__ import annotations

import logging
import math
from typing import Any, ClassVar

from proxima.plugins.base import AnalyzerPlugin, PluginMetadata, PluginType

logger = logging.getLogger(__name__)


class StatisticalAnalyzerPlugin(AnalyzerPlugin):
    """Statistical analysis of quantum circuit results.
    
    Computes statistical metrics including:
    - Mean and standard deviation
    - Entropy measures
    - Distribution analysis
    - Confidence intervals
    
    Example:
        analyzer = StatisticalAnalyzerPlugin()
        insights = analyzer.analyze({"counts": {"00": 500, "11": 500}})
        print(insights["entropy"])  # 1.0
    """
    
    METADATA: ClassVar[PluginMetadata] = PluginMetadata(
        name="statistical_analyzer",
        version="1.0.0",
        plugin_type=PluginType.ANALYZER,
        description="Statistical analysis of quantum results",
        author="Proxima Team",
        provides=["entropy", "distribution", "statistics"],
        config_schema={
            "type": "object",
            "properties": {
                "confidence_level": {"type": "number", "default": 0.95},
            },
        },
    )
    
    def get_analyzer_name(self) -> str:
        """Return the analyzer identifier."""
        return "statistical"
    
    def analyze(self, results: Any) -> dict[str, Any]:
        """Analyze results and return statistical insights.
        
        Args:
            results: Execution results containing counts.
        
        Returns:
            Dictionary of statistical insights.
        """
        if not isinstance(results, dict):
            return {"error": "Invalid results format"}
        
        counts = results.get("counts", {})
        if not counts:
            return {"error": "No counts data available"}
        
        total = sum(counts.values())
        probabilities = {k: v / total for k, v in counts.items()}
        
        # Calculate entropy
        entropy = self._calculate_entropy(probabilities)
        
        # Calculate uniformity (how close to uniform distribution)
        num_states = len(counts)
        max_entropy = math.log2(num_states) if num_states > 0 else 0
        uniformity = entropy / max_entropy if max_entropy > 0 else 1.0
        
        # Find dominant state
        dominant_state = max(counts, key=counts.get)
        dominant_probability = probabilities[dominant_state]
        
        # Distribution analysis
        sorted_probs = sorted(probabilities.values(), reverse=True)
        top_3_prob = sum(sorted_probs[:3]) if len(sorted_probs) >= 3 else sum(sorted_probs)
        
        # Confidence interval for binomial
        confidence_level = self.get_config("confidence_level", 0.95)
        intervals = {}
        for state, prob in probabilities.items():
            ci = self._binomial_confidence_interval(prob, total, confidence_level)
            intervals[state] = {"lower": ci[0], "upper": ci[1]}
        
        return {
            "total_shots": total,
            "num_states": num_states,
            "entropy": entropy,
            "max_entropy": max_entropy,
            "uniformity": uniformity,
            "dominant_state": dominant_state,
            "dominant_probability": dominant_probability,
            "top_3_probability": top_3_prob,
            "probabilities": probabilities,
            "confidence_intervals": intervals,
            "is_uniform": uniformity > 0.95,
            "is_peaked": dominant_probability > 0.8,
        }
    
    def _calculate_entropy(self, probabilities: dict[str, float]) -> float:
        """Calculate Shannon entropy."""
        entropy = 0.0
        for p in probabilities.values():
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy
    
    def _binomial_confidence_interval(
        self,
        p: float,
        n: int,
        confidence: float,
    ) -> tuple[float, float]:
        """Calculate binomial confidence interval using normal approximation."""
        if n == 0:
            return (0.0, 1.0)
        
        # Z-score for confidence level
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
        
        # Standard error
        se = math.sqrt(p * (1 - p) / n) if p > 0 and p < 1 else 0
        
        lower = max(0, p - z * se)
        upper = min(1, p + z * se)
        
        return (lower, upper)


class FidelityAnalyzerPlugin(AnalyzerPlugin):
    """Analyze fidelity between different backend results.
    
    Computes:
    - Classical fidelity
    - Statistical distance
    - KL divergence
    - Hellinger distance
    
    Example:
        analyzer = FidelityAnalyzerPlugin()
        insights = analyzer.analyze({
            "reference": {"00": 500, "11": 500},
            "comparison": {"00": 480, "11": 520}
        })
    """
    
    METADATA: ClassVar[PluginMetadata] = PluginMetadata(
        name="fidelity_analyzer",
        version="1.0.0",
        plugin_type=PluginType.ANALYZER,
        description="Fidelity analysis between backend results",
        author="Proxima Team",
        provides=["fidelity", "distance", "divergence"],
        config_schema={
            "type": "object",
            "properties": {
                "epsilon": {"type": "number", "default": 1e-10},
            },
        },
    )
    
    def get_analyzer_name(self) -> str:
        """Return the analyzer identifier."""
        return "fidelity"
    
    def analyze(self, results: Any) -> dict[str, Any]:
        """Analyze fidelity between result sets.
        
        Args:
            results: Dictionary with 'reference' and 'comparison' counts,
                    or list of counts dictionaries.
        
        Returns:
            Dictionary of fidelity metrics.
        """
        if isinstance(results, list) and len(results) >= 2:
            ref_counts = results[0]
            comp_counts = results[1]
        elif isinstance(results, dict):
            ref_counts = results.get("reference", results.get("counts", {}))
            comp_counts = results.get("comparison", {})
        else:
            return {"error": "Invalid results format"}
        
        if not ref_counts or not comp_counts:
            return {"error": "Missing reference or comparison data"}
        
        # Normalize to probabilities
        ref_total = sum(ref_counts.values())
        comp_total = sum(comp_counts.values())
        
        all_states = set(ref_counts.keys()) | set(comp_counts.keys())
        
        ref_probs = {s: ref_counts.get(s, 0) / ref_total for s in all_states}
        comp_probs = {s: comp_counts.get(s, 0) / comp_total for s in all_states}
        
        # Calculate metrics
        epsilon = self.get_config("epsilon", 1e-10)
        
        classical_fidelity = self._classical_fidelity(ref_probs, comp_probs)
        statistical_distance = self._statistical_distance(ref_probs, comp_probs)
        kl_divergence = self._kl_divergence(ref_probs, comp_probs, epsilon)
        hellinger_distance = self._hellinger_distance(ref_probs, comp_probs)
        
        # State-by-state comparison
        state_differences = {
            s: abs(ref_probs[s] - comp_probs[s])
            for s in all_states
        }
        max_diff_state = max(state_differences, key=state_differences.get)
        
        return {
            "classical_fidelity": classical_fidelity,
            "statistical_distance": statistical_distance,
            "kl_divergence": kl_divergence,
            "hellinger_distance": hellinger_distance,
            "num_states": len(all_states),
            "max_difference": state_differences[max_diff_state],
            "max_difference_state": max_diff_state,
            "state_differences": state_differences,
            "agreement": classical_fidelity > 0.99,
            "summary": self._generate_summary(
                classical_fidelity, statistical_distance
            ),
        }
    
    def _classical_fidelity(
        self,
        p: dict[str, float],
        q: dict[str, float],
    ) -> float:
        """Calculate classical fidelity (Bhattacharyya coefficient)."""
        fidelity = 0.0
        for s in p:
            fidelity += math.sqrt(p[s] * q.get(s, 0))
        return fidelity
    
    def _statistical_distance(
        self,
        p: dict[str, float],
        q: dict[str, float],
    ) -> float:
        """Calculate total variation distance."""
        distance = 0.0
        for s in set(p.keys()) | set(q.keys()):
            distance += abs(p.get(s, 0) - q.get(s, 0))
        return distance / 2
    
    def _kl_divergence(
        self,
        p: dict[str, float],
        q: dict[str, float],
        epsilon: float,
    ) -> float:
        """Calculate KL divergence."""
        divergence = 0.0
        for s in p:
            p_val = p[s]
            q_val = q.get(s, epsilon)
            if p_val > 0:
                divergence += p_val * math.log(p_val / max(q_val, epsilon))
        return divergence
    
    def _hellinger_distance(
        self,
        p: dict[str, float],
        q: dict[str, float],
    ) -> float:
        """Calculate Hellinger distance."""
        sum_sq = 0.0
        for s in set(p.keys()) | set(q.keys()):
            diff = math.sqrt(p.get(s, 0)) - math.sqrt(q.get(s, 0))
            sum_sq += diff ** 2
        return math.sqrt(sum_sq / 2)
    
    def _generate_summary(
        self,
        fidelity: float,
        distance: float,
    ) -> str:
        """Generate human-readable summary."""
        if fidelity > 0.999:
            return "Excellent agreement between distributions"
        elif fidelity > 0.99:
            return "Very good agreement with minor differences"
        elif fidelity > 0.95:
            return "Good agreement with some notable differences"
        elif fidelity > 0.90:
            return "Moderate agreement with significant differences"
        else:
            return "Poor agreement - distributions differ substantially"


class PerformanceAnalyzerPlugin(AnalyzerPlugin):
    """Analyze execution performance metrics.
    
    Computes:
    - Execution time statistics
    - Throughput metrics
    - Resource utilization estimates
    
    Example:
        analyzer = PerformanceAnalyzerPlugin()
        insights = analyzer.analyze({
            "execution_time_ms": 150,
            "shots": 1000,
            "num_qubits": 10
        })
    """
    
    METADATA: ClassVar[PluginMetadata] = PluginMetadata(
        name="performance_analyzer",
        version="1.0.0",
        plugin_type=PluginType.ANALYZER,
        description="Performance analysis of circuit execution",
        author="Proxima Team",
        provides=["performance", "throughput", "timing"],
    )
    
    def get_analyzer_name(self) -> str:
        """Return the analyzer identifier."""
        return "performance"
    
    def analyze(self, results: Any) -> dict[str, Any]:
        """Analyze performance metrics.
        
        Args:
            results: Execution results with timing information.
        
        Returns:
            Dictionary of performance insights.
        """
        if not isinstance(results, dict):
            return {"error": "Invalid results format"}
        
        execution_time_ms = results.get("execution_time_ms", 0)
        shots = results.get("shots", 1000)
        num_qubits = results.get("num_qubits", 2)
        num_gates = results.get("num_gates", 10)
        backend = results.get("backend", "unknown")
        
        # Calculate throughput
        shots_per_second = (shots / execution_time_ms * 1000) if execution_time_ms > 0 else 0
        gates_per_ms = (num_gates / execution_time_ms) if execution_time_ms > 0 else 0
        
        # Estimate memory usage
        state_vector_memory = (2 ** num_qubits) * 16  # 16 bytes per complex
        
        # Performance tier classification
        if execution_time_ms < 10:
            tier = "excellent"
        elif execution_time_ms < 100:
            tier = "good"
        elif execution_time_ms < 1000:
            tier = "moderate"
        else:
            tier = "slow"
        
        # Scaling estimate
        estimated_double_qubits = execution_time_ms * (2 ** num_qubits)
        
        return {
            "execution_time_ms": execution_time_ms,
            "shots": shots,
            "num_qubits": num_qubits,
            "num_gates": num_gates,
            "backend": backend,
            "shots_per_second": shots_per_second,
            "gates_per_ms": gates_per_ms,
            "state_vector_memory_bytes": state_vector_memory,
            "performance_tier": tier,
            "estimated_double_qubits_ms": estimated_double_qubits,
            "recommendations": self._generate_recommendations(
                execution_time_ms, num_qubits, backend
            ),
        }
    
    def _generate_recommendations(
        self,
        execution_time_ms: float,
        num_qubits: int,
        backend: str,
    ) -> list[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        if execution_time_ms > 1000:
            recommendations.append(
                "Consider using GPU acceleration for large circuits"
            )
        
        if num_qubits > 20:
            recommendations.append(
                "Large qubit count - ensure sufficient RAM available"
            )
        
        if backend == "unknown":
            recommendations.append(
                "Specify a backend for optimal performance"
            )
        
        return recommendations
