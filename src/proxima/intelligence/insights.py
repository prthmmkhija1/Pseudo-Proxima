"""Result interpretation and insights generation (Step 3.4 Insight Engine).

Analysis Pipeline:
  Raw Results  Statistical Analysis  Pattern Detection  LLM Synthesis (opt)
               Insight Formatting  InsightReport

Insight Categories:
  1. Summary – one-paragraph overview
  2. Key Findings – bullet-point significant observations
  3. Statistical Metrics – quantitative analysis
  4. Recommendations – suggested next steps
  5. Visualizations – chart suggestions or ASCII representations
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from proxima.backends.base import ExecutionResult, ResultType


@dataclass
class StatisticalMetrics:
    """Quantitative metrics extracted from execution results."""

    total_shots: int = 0
    unique_states: int = 0
    dominant_state: Optional[str] = None
    dominant_probability: float = 0.0
    entropy: float = 0.0
    mean_probability: float = 0.0
    variance: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InsightReport:
    """Structured insight object returned by the engine."""

    summary: str
    key_findings: List[str]
    metrics: StatisticalMetrics
    recommendations: List[str]
    visualizations: List[str]
    raw_data: Dict[str, Any] = field(default_factory=dict)


class InsightEngine:
    """Generates human-readable, analytical insights from execution results."""

    def __init__(
        self,
        llm_synthesizer: Optional[Callable[[str], str]] = None,
    ) -> None:
        """
        Args:
            llm_synthesizer: Optional callable that takes a prompt and returns
                             LLM-generated text. If None, LLM synthesis is skipped.
        """
        self.llm_synthesizer = llm_synthesizer

    def analyze(self, result: ExecutionResult) -> InsightReport:
        """Run the full analysis pipeline on an ExecutionResult."""

        metrics = self._statistical_analysis(result)
        patterns = self._pattern_detection(result, metrics)
        recommendations = self._generate_recommendations(result, metrics, patterns)
        visualizations = self._build_visualizations(result, metrics)
        summary = self._build_summary(result, metrics, patterns)

        if self.llm_synthesizer:
            summary = self._llm_enhance_summary(summary, metrics, patterns)

        return InsightReport(
            summary=summary,
            key_findings=patterns,
            metrics=metrics,
            recommendations=recommendations,
            visualizations=visualizations,
            raw_data=result.data,
        )

    # -------------------------------------------------------------------------
    # Statistical Analysis
    # -------------------------------------------------------------------------
    def _statistical_analysis(self, result: ExecutionResult) -> StatisticalMetrics:
        if result.result_type == ResultType.COUNTS:
            return self._analyze_counts(result.data.get("counts", {}))
        if result.result_type in (ResultType.STATEVECTOR, ResultType.DENSITY_MATRIX):
            return self._analyze_amplitudes(result.data)
        return StatisticalMetrics()

    def _analyze_counts(self, counts: Dict[str, int]) -> StatisticalMetrics:
        if not counts:
            return StatisticalMetrics()

        total = sum(counts.values())
        unique = len(counts)
        probs = {state: c / total for state, c in counts.items()}
        dominant = max(probs, key=lambda s: probs[s])
        dominant_prob = probs[dominant]

        mean_prob = 1.0 / unique if unique else 0.0
        variance = sum((p - mean_prob) ** 2 for p in probs.values()) / unique if unique else 0.0
        entropy = -sum(p * math.log2(p) for p in probs.values() if p > 0)

        return StatisticalMetrics(
            total_shots=total,
            unique_states=unique,
            dominant_state=dominant,
            dominant_probability=dominant_prob,
            entropy=entropy,
            mean_probability=mean_prob,
            variance=variance,
        )

    def _analyze_amplitudes(self, data: Dict[str, Any]) -> StatisticalMetrics:
        """Placeholder for statevector/density matrix analysis."""
        return StatisticalMetrics(metadata={"note": "amplitude analysis not yet implemented"})

    # -------------------------------------------------------------------------
    # Pattern Detection
    # -------------------------------------------------------------------------
    def _pattern_detection(
        self, result: ExecutionResult, metrics: StatisticalMetrics
    ) -> List[str]:
        findings: List[str] = []

        if metrics.dominant_probability >= 0.9:
            findings.append(
                f"Highly deterministic outcome: state '{metrics.dominant_state}' observed "
                f"{metrics.dominant_probability:.1%} of the time."
            )
        elif metrics.dominant_probability >= 0.5:
            findings.append(
                f"Dominant state '{metrics.dominant_state}' with probability "
                f"{metrics.dominant_probability:.1%}."
            )
        else:
            findings.append("Results are spread across multiple states (high entropy).")

        if metrics.entropy > 0:
            findings.append(f"Shannon entropy: {metrics.entropy:.3f} bits.")

        if metrics.unique_states > 1:
            findings.append(f"Observed {metrics.unique_states} unique measurement outcomes.")

        return findings

    # -------------------------------------------------------------------------
    # Recommendations
    # -------------------------------------------------------------------------
    def _generate_recommendations(
        self,
        result: ExecutionResult,
        metrics: StatisticalMetrics,
        patterns: List[str],
    ) -> List[str]:
        recs: List[str] = []

        if metrics.total_shots and metrics.total_shots < 1000:
            recs.append("Consider increasing shot count for more precise statistics.")

        if metrics.entropy and metrics.entropy > 3.0:
            recs.append(
                "High entropy suggests a noisy or highly superposed state; "
                "verify circuit correctness or add error mitigation."
            )

        if not recs:
            recs.append("No immediate action required; results appear consistent.")

        return recs

    # -------------------------------------------------------------------------
    # Visualizations (ASCII placeholders)
    # -------------------------------------------------------------------------
    def _build_visualizations(
        self, result: ExecutionResult, metrics: StatisticalMetrics
    ) -> List[str]:
        visuals: List[str] = []

        if result.result_type == ResultType.COUNTS and result.data.get("counts"):
            visuals.append(self._ascii_histogram(result.data["counts"]))

        return visuals

    def _ascii_histogram(self, counts: Dict[str, int], width: int = 40) -> str:
        if not counts:
            return """
        max_val = max(counts.values())
        lines = ["Histogram:"]
        for state, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            bar_len = int((cnt / max_val) * width) if max_val else 0
            lines.append(f"  {state}: {'#' * bar_len} ({cnt})")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    def _build_summary(
        self,
        result: ExecutionResult,
        metrics: StatisticalMetrics,
        patterns: List[str],
    ) -> str:
        parts = [
            f"Execution on backend '{result.backend}' completed in "
            f"{result.execution_time_ms:.1f} ms.",
        ]
        if metrics.total_shots:
            parts.append(f"Total shots: {metrics.total_shots}.")
        if patterns:
            parts.append(patterns[0])
        return " ".join(parts)

    # -------------------------------------------------------------------------
    # Optional LLM Synthesis
    # -------------------------------------------------------------------------
    def _llm_enhance_summary(
        self, base_summary: str, metrics: StatisticalMetrics, patterns: List[str]
    ) -> str:
        prompt = (
            "You are a quantum computing assistant. Given the following execution summary "
            "and key findings, rewrite them into a concise, insightful paragraph for a user.\n\n"
            f"Summary: {base_summary}\n"
            f"Key findings: {'; '.join(patterns)}\n\n"
            "Rewritten insight:"
        )
        try:
            return self.llm_synthesizer(prompt)  # type: ignore[misc]
        except Exception:
            return base_summary


insight_engine = InsightEngine()

