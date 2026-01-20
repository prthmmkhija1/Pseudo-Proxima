"""
Insight Engine with LLM-Enhanced Analysis.

Intelligent result interpretation, pattern detection, and
LLM-powered synthesis for quantum simulation results.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class InsightLevel(Enum):
    """Level of insight detail."""

    BASIC = auto()  # Quick summary
    STANDARD = auto()  # Normal analysis
    DETAILED = auto()  # Full analysis with recommendations
    EXPERT = auto()  # Include LLM synthesis


class PatternType(Enum):
    """Types of patterns detected in results."""

    UNIFORM = "uniform"  # All states equal probability
    PEAKED = "peaked"  # One or few dominant states
    BIMODAL = "bimodal"  # Two clusters of high probability
    EXPONENTIAL = "exponential"  # Exponential decay pattern
    OSCILLATING = "oscillating"  # Alternating high/low
    RANDOM = "random"  # No clear pattern
    ENTANGLED = "entangled"  # Bell-state-like correlations
    GHZ = "ghz"  # GHZ state pattern (all-zeros + all-ones)
    SPARSE = "sparse"  # Sparse distribution with few non-zero states


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class StatisticalMetrics:
    """Statistical metrics computed from simulation results."""

    # Probability metrics
    entropy: float  # Shannon entropy
    max_probability: float  # Highest probability state
    min_probability: float  # Lowest non-zero probability
    mean_probability: float  # Average probability
    std_probability: float  # Standard deviation

    # State metrics
    total_states: int  # Number of possible states
    non_zero_states: int  # States with non-zero probability
    dominant_state: str  # State with highest probability
    dominant_probability: float

    # Concentration metrics
    effective_dimension: float  # Participation ratio
    gini_coefficient: float  # Inequality measure (0=equal, 1=concentrated)
    top_k_coverage: float  # Coverage by top-k states (k=5)


@dataclass
class AmplitudeAnalysis:
    """Analysis of quantum state amplitudes."""

    real_mean: float
    real_std: float
    imag_mean: float
    imag_std: float
    phase_distribution: list[float]  # Histogram of phases
    magnitude_distribution: list[float]  # Histogram of magnitudes
    coherence_estimate: float  # 0-1, estimate of coherence


@dataclass
class PatternInfo:
    """Information about a detected pattern."""

    pattern_type: PatternType
    confidence: float  # 0-1
    description: str
    affected_states: list[str]
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class Recommendation:
    """A recommendation based on analysis."""

    title: str
    description: str
    priority: int  # 1=high, 2=medium, 3=low
    category: str  # 'optimization', 'accuracy', 'interpretation', etc.
    action: str | None = None  # Suggested action


@dataclass
class Visualization:
    """Suggested visualization for results."""

    viz_type: str  # 'bar', 'histogram', 'heatmap', 'bloch', etc.
    title: str
    description: str
    data_key: str  # Key in results to visualize
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class InsightReport:
    """Complete insight report for simulation results."""

    # Summary
    summary: str
    key_findings: list[str]

    # Detailed analysis
    statistics: StatisticalMetrics
    amplitude_analysis: AmplitudeAnalysis | None
    patterns: list[PatternInfo]

    # Actionable items
    recommendations: list[Recommendation]
    visualizations: list[Visualization]

    # LLM content
    llm_synthesis: str | None = None
    llm_questions: list[str] = field(default_factory=list)  # Suggested follow-up

    # Metadata
    analysis_level: InsightLevel = InsightLevel.STANDARD
    warnings: list[str] = field(default_factory=list)


# =============================================================================
# Statistical Analysis
# =============================================================================


class StatisticalAnalyzer:
    """Computes statistical metrics from probability distributions."""

    def analyze(self, probabilities: dict[str, float]) -> StatisticalMetrics:
        """
        Analyze a probability distribution.

        Args:
            probabilities: Mapping of state -> probability.

        Returns:
            StatisticalMetrics with computed values.
        """
        if not probabilities:
            return self._empty_metrics()

        probs = list(probabilities.values())
        states = list(probabilities.keys())
        non_zero = [p for p in probs if p > 1e-10]

        # Basic statistics
        max_prob = max(probs)
        min_prob = min(non_zero) if non_zero else 0.0
        mean_prob = sum(probs) / len(probs)

        # Standard deviation
        variance = sum((p - mean_prob) ** 2 for p in probs) / len(probs)
        std_prob = math.sqrt(variance)

        # Shannon entropy
        entropy = -sum(p * math.log2(p) for p in non_zero if p > 0)

        # Dominant state
        max_idx = probs.index(max_prob)
        dominant_state = states[max_idx]

        # Effective dimension (participation ratio)
        sum_p2 = sum(p**2 for p in probs)
        effective_dim = 1.0 / sum_p2 if sum_p2 > 0 else 0.0

        # Gini coefficient
        gini = self._compute_gini(sorted(probs))

        # Top-k coverage
        top_k = sorted(probs, reverse=True)[:5]
        top_k_coverage = sum(top_k)

        return StatisticalMetrics(
            entropy=entropy,
            max_probability=max_prob,
            min_probability=min_prob,
            mean_probability=mean_prob,
            std_probability=std_prob,
            total_states=len(probs),
            non_zero_states=len(non_zero),
            dominant_state=dominant_state,
            dominant_probability=max_prob,
            effective_dimension=effective_dim,
            gini_coefficient=gini,
            top_k_coverage=top_k_coverage,
        )

    def _compute_gini(self, sorted_probs: list[float]) -> float:
        """Compute Gini coefficient."""
        n = len(sorted_probs)
        if n == 0:
            return 0.0

        cumsum = 0.0
        weighted_sum = 0.0

        for i, p in enumerate(sorted_probs):
            cumsum += p
            weighted_sum += (i + 1) * p

        if cumsum == 0:
            return 0.0

        return (2 * weighted_sum / (n * cumsum)) - (n + 1) / n

    def _empty_metrics(self) -> StatisticalMetrics:
        """Return empty metrics."""
        return StatisticalMetrics(
            entropy=0.0,
            max_probability=0.0,
            min_probability=0.0,
            mean_probability=0.0,
            std_probability=0.0,
            total_states=0,
            non_zero_states=0,
            dominant_state="",
            dominant_probability=0.0,
            effective_dimension=0.0,
            gini_coefficient=0.0,
            top_k_coverage=0.0,
        )


class AmplitudeAnalyzer:
    """Analyzes quantum state amplitudes."""

    def analyze(self, amplitudes: dict[str, complex]) -> AmplitudeAnalysis:
        """
        Analyze amplitude distribution.

        Args:
            amplitudes: Mapping of state -> complex amplitude.

        Returns:
            AmplitudeAnalysis with computed values.
        """
        if not amplitudes:
            return self._empty_analysis()

        amps = list(amplitudes.values())

        # Real and imaginary parts
        reals = [a.real for a in amps]
        imags = [a.imag for a in amps]

        real_mean = sum(reals) / len(reals)
        imag_mean = sum(imags) / len(imags)

        real_var = sum((r - real_mean) ** 2 for r in reals) / len(reals)
        imag_var = sum((i - imag_mean) ** 2 for i in imags) / len(imags)

        # Phases (in units of pi)
        phases = []
        for a in amps:
            if abs(a) > 1e-10:
                phase = math.atan2(a.imag, a.real) / math.pi
                phases.append(phase)

        # Create phase histogram (8 bins from -1 to 1)
        phase_hist = [0.0] * 8
        for p in phases:
            bin_idx = int((p + 1) * 4) % 8
            phase_hist[bin_idx] += 1
        if phases:
            phase_hist = [c / len(phases) for c in phase_hist]

        # Magnitude distribution (8 bins from 0 to max)
        magnitudes = [abs(a) for a in amps]
        max_mag = max(magnitudes) if magnitudes else 1.0
        mag_hist = [0.0] * 8
        for m in magnitudes:
            bin_idx = min(int(m / max_mag * 8), 7) if max_mag > 0 else 0
            mag_hist[bin_idx] += 1
        if magnitudes:
            mag_hist = [c / len(magnitudes) for c in mag_hist]

        # Coherence estimate (based on phase spread)
        if len(phases) > 1:
            phase_std = math.sqrt(
                sum((p - sum(phases) / len(phases)) ** 2 for p in phases) / len(phases)
            )
            coherence = max(0.0, 1.0 - phase_std)
        else:
            coherence = 1.0

        return AmplitudeAnalysis(
            real_mean=real_mean,
            real_std=math.sqrt(real_var),
            imag_mean=imag_mean,
            imag_std=math.sqrt(imag_var),
            phase_distribution=phase_hist,
            magnitude_distribution=mag_hist,
            coherence_estimate=coherence,
        )

    def _empty_analysis(self) -> AmplitudeAnalysis:
        """Return empty analysis."""
        return AmplitudeAnalysis(
            real_mean=0.0,
            real_std=0.0,
            imag_mean=0.0,
            imag_std=0.0,
            phase_distribution=[0.0] * 8,
            magnitude_distribution=[0.0] * 8,
            coherence_estimate=0.0,
        )


# =============================================================================
# Pattern Detection
# =============================================================================


class PatternDetector:
    """Detects patterns in quantum simulation results."""

    def detect(self, probabilities: dict[str, float]) -> list[PatternInfo]:
        """
        Detect patterns in probability distribution.

        Args:
            probabilities: Mapping of state -> probability.

        Returns:
            List of detected patterns with confidence scores.
        """
        if not probabilities:
            return []

        patterns = []

        # Check for uniform distribution
        uniform = self._check_uniform(probabilities)
        if uniform:
            patterns.append(uniform)

        # Check for peaked distribution
        peaked = self._check_peaked(probabilities)
        if peaked:
            patterns.append(peaked)

        # Check for bimodal distribution
        bimodal = self._check_bimodal(probabilities)
        if bimodal:
            patterns.append(bimodal)

        # Check for entanglement signatures
        entangled = self._check_entanglement(probabilities)
        if entangled:
            patterns.append(entangled)

        # Check for GHZ state pattern
        ghz = self._check_ghz(probabilities)
        if ghz:
            patterns.append(ghz)

        # Check for sparse distribution
        sparse = self._check_sparse(probabilities)
        if sparse:
            patterns.append(sparse)

        # If no patterns found, classify as random
        if not patterns:
            patterns.append(
                PatternInfo(
                    pattern_type=PatternType.RANDOM,
                    confidence=0.5,
                    description="No clear pattern detected",
                    affected_states=list(probabilities.keys()),
                )
            )

        return patterns

    def _check_uniform(self, probs: dict[str, float]) -> PatternInfo | None:
        """Check for uniform distribution."""
        values = list(probs.values())
        n = len(values)

        if n == 0:
            return None

        expected = 1.0 / n
        max_deviation = max(abs(p - expected) for p in values)

        # If all probabilities are close to expected
        if max_deviation < 0.05:
            return PatternInfo(
                pattern_type=PatternType.UNIFORM,
                confidence=1.0 - max_deviation * 10,
                description=f"Uniform distribution across {n} states",
                affected_states=list(probs.keys()),
                metrics={"expected_prob": expected, "max_deviation": max_deviation},
            )

        return None

    def _check_peaked(self, probs: dict[str, float]) -> PatternInfo | None:
        """Check for peaked distribution (one dominant state)."""
        values = list(probs.values())
        states = list(probs.keys())

        max_prob = max(values)
        max_state = states[values.index(max_prob)]

        # If one state has more than 50% probability
        if max_prob > 0.5:
            return PatternInfo(
                pattern_type=PatternType.PEAKED,
                confidence=max_prob,
                description=f"Strong peak at state '{max_state}' ({max_prob:.1%})",
                affected_states=[max_state],
                metrics={"peak_probability": max_prob},
            )

        return None

    def _check_bimodal(self, probs: dict[str, float]) -> PatternInfo | None:
        """Check for bimodal distribution (two dominant states)."""
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_probs) < 2:
            return None

        top_two = sorted_probs[:2]
        combined_prob = top_two[0][1] + top_two[1][1]

        # If top two states account for most probability and are similar
        if combined_prob > 0.8:
            ratio = top_two[1][1] / top_two[0][1] if top_two[0][1] > 0 else 0
            if ratio > 0.5:  # Second is at least half of first
                return PatternInfo(
                    pattern_type=PatternType.BIMODAL,
                    confidence=combined_prob * ratio,
                    description=(
                        f"Bimodal distribution: '{top_two[0][0]}' ({top_two[0][1]:.1%}) "
                        f"and '{top_two[1][0]}' ({top_two[1][1]:.1%})"
                    ),
                    affected_states=[top_two[0][0], top_two[1][0]],
                    metrics={"combined_probability": combined_prob, "ratio": ratio},
                )

        return None

    def _check_entanglement(self, probs: dict[str, float]) -> PatternInfo | None:
        """Check for entanglement signatures (Bell-state patterns)."""
        # Look for Bell-state-like patterns: |00⟩ + |11⟩ or |01⟩ + |10⟩
        states = list(probs.keys())

        if not states:
            return None

        # Check if states are binary strings
        try:
            n_qubits = len(states[0])
            if not all(len(s) == n_qubits and set(s) <= {"0", "1"} for s in states):
                return None
        except (TypeError, ValueError):
            return None

        # Check for correlated states (00...0 with 11...1)
        all_zeros = "0" * n_qubits
        all_ones = "1" * n_qubits

        if all_zeros in probs and all_ones in probs:
            combined = probs[all_zeros] + probs[all_ones]
            if combined > 0.9:
                return PatternInfo(
                    pattern_type=PatternType.ENTANGLED,
                    confidence=combined,
                    description=(
                        f"Bell-state-like entanglement: |{all_zeros}⟩ + |{all_ones}⟩"
                    ),
                    affected_states=[all_zeros, all_ones],
                    metrics={"combined_probability": combined},
                )

        return None

    def _check_ghz(self, probs: dict[str, float]) -> PatternInfo | None:
        """Check for GHZ state pattern (multi-qubit entanglement)."""
        states = list(probs.keys())

        if not states:
            return None

        # Check if states are binary strings
        try:
            n_qubits = len(states[0])
            if n_qubits < 3:  # GHZ typically needs 3+ qubits
                return None
            if not all(len(s) == n_qubits and set(s) <= {"0", "1"} for s in states):
                return None
        except (TypeError, ValueError):
            return None

        # GHZ state: equal superposition of |00...0⟩ and |11...1⟩
        all_zeros = "0" * n_qubits
        all_ones = "1" * n_qubits

        if all_zeros in probs and all_ones in probs:
            p_zeros = probs[all_zeros]
            p_ones = probs[all_ones]
            combined = p_zeros + p_ones

            # Check for roughly equal probabilities (GHZ signature)
            if combined > 0.9 and abs(p_zeros - p_ones) < 0.1:
                return PatternInfo(
                    pattern_type=PatternType.GHZ,
                    confidence=combined * (1 - abs(p_zeros - p_ones)),
                    description=(
                        f"GHZ state pattern detected ({n_qubits} qubits): "
                        f"equal superposition of |{all_zeros}⟩ and |{all_ones}⟩"
                    ),
                    affected_states=[all_zeros, all_ones],
                    metrics={
                        "combined_probability": combined,
                        "p_zeros": p_zeros,
                        "p_ones": p_ones,
                        "n_qubits": n_qubits,
                    },
                )

        return None

    def _check_sparse(self, probs: dict[str, float]) -> PatternInfo | None:
        """Check for sparse distribution (few non-zero states)."""
        values = list(probs.values())
        n = len(values)

        if n == 0:
            return None

        # Count non-zero states (above threshold)
        threshold = 1e-6
        non_zero = [p for p in values if p > threshold]
        non_zero_count = len(non_zero)

        # Calculate sparsity ratio
        sparsity_ratio = 1 - (non_zero_count / n) if n > 0 else 0

        # Consider sparse if less than 10% of states are populated
        # OR if fewer than 5 states have significant probability
        if sparsity_ratio > 0.9 or (n > 10 and non_zero_count <= 5):
            # Get the populated states
            populated = [s for s, p in probs.items() if p > threshold]

            return PatternInfo(
                pattern_type=PatternType.SPARSE,
                confidence=sparsity_ratio,
                description=(
                    f"Sparse distribution: only {non_zero_count}/{n} states "
                    f"({100 * non_zero_count / n:.1f}%) have non-zero probability"
                ),
                affected_states=populated,
                metrics={
                    "sparsity_ratio": sparsity_ratio,
                    "non_zero_count": non_zero_count,
                    "total_states": n,
                },
            )

        return None


# =============================================================================
# Recommendation Engine
# =============================================================================


class RecommendationEngine:
    """Generates recommendations based on analysis."""

    def generate(
        self,
        statistics: StatisticalMetrics,
        patterns: list[PatternInfo],
    ) -> list[Recommendation]:
        """Generate recommendations based on analysis results."""
        recommendations = []

        # Based on entropy
        if statistics.entropy < 1.0:
            recommendations.append(
                Recommendation(
                    title="Low entropy detected",
                    description=(
                        "The result has low entropy, indicating a concentrated distribution. "
                        "This could indicate successful state preparation or collapse."
                    ),
                    priority=2,
                    category="interpretation",
                )
            )
        elif statistics.entropy > math.log2(statistics.total_states) - 1:
            recommendations.append(
                Recommendation(
                    title="Near-maximum entropy",
                    description=(
                        "The distribution is nearly uniform. This could indicate "
                        "a maximally mixed state or uniform superposition."
                    ),
                    priority=2,
                    category="interpretation",
                )
            )

        # Based on concentration
        if statistics.gini_coefficient > 0.8:
            recommendations.append(
                Recommendation(
                    title="Highly concentrated results",
                    description=(
                        f"Results are concentrated in a few states "
                        f"(Gini={statistics.gini_coefficient:.2f}). "
                        "Consider increasing shots for better statistics."
                    ),
                    priority=2,
                    category="accuracy",
                    action="Increase shot count to at least 4096",
                )
            )

        # Based on patterns
        for pattern in patterns:
            if pattern.pattern_type == PatternType.ENTANGLED:
                recommendations.append(
                    Recommendation(
                        title="Entanglement detected",
                        description=(
                            "Bell-state-like correlations suggest entanglement. "
                            "Consider measuring in different bases to verify."
                        ),
                        priority=1,
                        category="interpretation",
                        action="Run measurements in X and Y bases",
                    )
                )
            elif pattern.pattern_type == PatternType.BIMODAL:
                recommendations.append(
                    Recommendation(
                        title="Bimodal distribution",
                        description=(
                            "Two dominant states suggest a superposition. "
                            "Check for intended interference effects."
                        ),
                        priority=2,
                        category="interpretation",
                    )
                )

        # Based on effective dimension
        if statistics.effective_dimension < statistics.total_states * 0.1:
            recommendations.append(
                Recommendation(
                    title="Low effective dimension",
                    description=(
                        "Only a small fraction of states are populated. "
                        "The circuit may have limited quantum advantage."
                    ),
                    priority=3,
                    category="optimization",
                )
            )

        return recommendations


class VisualizationRecommender:
    """Recommends appropriate visualizations."""

    def recommend(
        self,
        statistics: StatisticalMetrics,
        patterns: list[PatternInfo],
        has_amplitudes: bool = False,
    ) -> list[Visualization]:
        """Recommend visualizations based on analysis."""
        visualizations = []

        # Always recommend probability histogram
        visualizations.append(
            Visualization(
                viz_type="bar",
                title="State Probabilities",
                description="Bar chart of measurement probabilities by state",
                data_key="probabilities",
                options={"sort": "probability", "top_k": 20},
            )
        )

        # For peaked distributions, show cumulative
        if statistics.gini_coefficient > 0.5:
            visualizations.append(
                Visualization(
                    viz_type="cumulative",
                    title="Cumulative Probability",
                    description="Shows how few states contain most probability",
                    data_key="probabilities",
                )
            )

        # For amplitude data, show phase
        if has_amplitudes:
            visualizations.append(
                Visualization(
                    viz_type="polar",
                    title="Amplitude Phases",
                    description="Polar plot of complex amplitudes",
                    data_key="amplitudes",
                )
            )

        # For entangled states, show correlation matrix
        for pattern in patterns:
            if pattern.pattern_type == PatternType.ENTANGLED:
                visualizations.append(
                    Visualization(
                        viz_type="heatmap",
                        title="State Correlations",
                        description="Heatmap showing correlations between qubits",
                        data_key="correlations",
                    )
                )
                break

        # For 1-2 qubit systems, Bloch sphere
        if statistics.total_states <= 4:
            visualizations.append(
                Visualization(
                    viz_type="bloch",
                    title="Bloch Sphere",
                    description="State visualization on Bloch sphere",
                    data_key="amplitudes",
                )
            )

        return visualizations


# =============================================================================
# ASCII Visualization Renderer
# =============================================================================


class ASCIIVisualizer:
    """Renders ASCII visualizations for terminal output."""

    # Unicode block characters for bar charts
    BLOCKS = " ▏▎▍▌▋▊▉█"

    def __init__(
        self,
        width: int = 50,
        height: int = 10,
        use_unicode: bool = True,
    ) -> None:
        """
        Initialize visualizer.

        Args:
            width: Maximum width for visualizations.
            height: Maximum height for vertical charts.
            use_unicode: Whether to use Unicode characters.
        """
        self.width = width
        self.height = height
        self.use_unicode = use_unicode

    def render_bar_chart(
        self,
        probabilities: dict[str, float],
        top_k: int = 10,
        show_values: bool = True,
    ) -> str:
        """
        Render horizontal bar chart of probabilities.

        Args:
            probabilities: State -> probability mapping.
            top_k: Number of top states to show.
            show_values: Whether to show probability values.

        Returns:
            ASCII bar chart string.
        """
        if not probabilities:
            return "No data to display"

        # Sort by probability and take top k
        sorted_probs = sorted(
            probabilities.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        if not sorted_probs:
            return "No data to display"

        max_prob = max(p for _, p in sorted_probs)
        max_state_len = max(len(s) for s, _ in sorted_probs)
        bar_width = self.width - max_state_len - 12  # Space for state + value

        lines = []
        for state, prob in sorted_probs:
            # Calculate bar length
            bar_len = int((prob / max_prob) * bar_width) if max_prob > 0 else 0

            if self.use_unicode:
                # Use fractional blocks for smooth bars
                full_blocks = bar_len
                bar = "█" * full_blocks
            else:
                bar = "#" * bar_len

            state_padded = state.rjust(max_state_len)

            if show_values:
                line = f"{state_padded} │{bar.ljust(bar_width)} {prob:6.2%}"
            else:
                line = f"{state_padded} │{bar}"

            lines.append(line)

        # Add summary line
        other_count = len(probabilities) - len(sorted_probs)
        if other_count > 0:
            other_prob = sum(
                p for s, p in probabilities.items()
                if s not in dict(sorted_probs)
            )
            lines.append(f"{'...' + str(other_count) + ' more':>{max_state_len}} │{'':.<{bar_width}} {other_prob:6.2%}")

        return "\n".join(lines)

    def render_histogram(
        self,
        probabilities: dict[str, float],
        bins: int = 10,
    ) -> str:
        """
        Render vertical histogram of probability distribution.

        Args:
            probabilities: State -> probability mapping.
            bins: Number of bins for histogram.

        Returns:
            ASCII histogram string.
        """
        if not probabilities:
            return "No data to display"

        values = list(probabilities.values())
        min_val, max_val = min(values), max(values)

        if max_val == min_val:
            # All same value
            return f"Uniform distribution: all states at {max_val:.2%}"

        # Create bins
        bin_width = (max_val - min_val) / bins
        bin_counts = [0] * bins

        for v in values:
            bin_idx = min(int((v - min_val) / bin_width), bins - 1)
            bin_counts[bin_idx] += 1

        max_count = max(bin_counts)
        if max_count == 0:
            return "No data to display"

        # Render vertical bars
        lines = []
        for row in range(self.height, 0, -1):
            threshold = (row / self.height) * max_count
            chars = []
            for count in bin_counts:
                if count >= threshold:
                    chars.append("█" if self.use_unicode else "#")
                else:
                    chars.append(" ")
            lines.append("│" + "".join(chars))

        # Add x-axis
        lines.append("└" + "─" * bins)
        lines.append(f" {min_val:.1%}{'':^{bins-8}}{max_val:.1%}")

        return "\n".join(lines)

    def render_sparkline(
        self,
        values: list[float],
        width: int | None = None,
    ) -> str:
        """
        Render a sparkline for a series of values.

        Args:
            values: List of values to plot.
            width: Optional width (will sample if values exceed).

        Returns:
            Single-line sparkline string.
        """
        if not values:
            return ""

        width = width or self.width

        # Sample if too many values
        if len(values) > width:
            step = len(values) / width
            sampled = [values[int(i * step)] for i in range(width)]
            values = sampled

        min_val, max_val = min(values), max(values)
        val_range = max_val - min_val

        if val_range == 0:
            return "▄" * len(values)  # Flat line

        # Map to 8 levels (block characters)
        blocks = "▁▂▃▄▅▆▇█"
        result = []
        for v in values:
            level = int(((v - min_val) / val_range) * 7)
            result.append(blocks[level])

        return "".join(result)

    def render_heatmap(
        self,
        matrix: list[list[float]],
        row_labels: list[str] | None = None,
        col_labels: list[str] | None = None,
    ) -> str:
        """
        Render an ASCII heatmap.

        Args:
            matrix: 2D list of values.
            row_labels: Optional row labels.
            col_labels: Optional column labels.

        Returns:
            ASCII heatmap string.
        """
        if not matrix or not matrix[0]:
            return "No data to display"

        # Find value range
        all_vals = [v for row in matrix for v in row]
        min_val, max_val = min(all_vals), max(all_vals)
        val_range = max_val - min_val if max_val != min_val else 1

        # Heat characters (light to dark)
        if self.use_unicode:
            heat_chars = " ░▒▓█"
        else:
            heat_chars = " .:-=#"

        lines = []

        # Column labels
        if col_labels:
            header = "  " + "".join(c[0] if c else " " for c in col_labels)
            lines.append(header)

        for i, row in enumerate(matrix):
            chars = []
            for v in row:
                level = int(((v - min_val) / val_range) * (len(heat_chars) - 1))
                chars.append(heat_chars[level])

            row_label = row_labels[i] if row_labels and i < len(row_labels) else str(i)
            lines.append(f"{row_label[0]} " + "".join(chars))

        return "\n".join(lines)

    def render_qubit_correlations(
        self,
        probabilities: dict[str, float],
    ) -> str:
        """
        Render qubit correlation matrix from probability distribution.

        Args:
            probabilities: State -> probability mapping (binary strings).

        Returns:
            ASCII correlation heatmap.
        """
        if not probabilities:
            return "No data to display"

        # Get number of qubits
        states = list(probabilities.keys())
        try:
            n_qubits = len(states[0])
            if not all(len(s) == n_qubits and set(s) <= {"0", "1"} for s in states):
                return "States must be binary strings of equal length"
        except (TypeError, ValueError):
            return "Invalid state format"

        # Compute correlation matrix
        # correlation[i][j] = P(qubit_i = qubit_j)
        corr_matrix: list[list[float]] = []
        labels = [f"q{i}" for i in range(n_qubits)]

        for i in range(n_qubits):
            row = []
            for j in range(n_qubits):
                # P(qi = qj)
                agreement_prob = sum(
                    p for s, p in probabilities.items()
                    if s[i] == s[j]
                )
                row.append(agreement_prob)
            corr_matrix.append(row)

        return self.render_heatmap(corr_matrix, labels, labels)

    def render_state_distribution(
        self,
        probabilities: dict[str, float],
    ) -> str:
        """
        Render a compact state distribution view.

        Args:
            probabilities: State -> probability mapping.

        Returns:
            Compact distribution visualization.
        """
        if not probabilities:
            return "No data to display"

        # Sort by state (binary order)
        sorted_probs = sorted(probabilities.items())

        # Render as sparkline
        values = [p for _, p in sorted_probs]
        sparkline = self.render_sparkline(values)

        # Summary stats
        max_state, max_prob = max(probabilities.items(), key=lambda x: x[1])
        min_prob = min(probabilities.values())

        lines = [
            sparkline,
            f"States: {len(probabilities)} | Max: {max_state}={max_prob:.2%} | Min: {min_prob:.2%}",
        ]

        return "\n".join(lines)


# =============================================================================
# LLM Integration
# =============================================================================


class LLMSynthesizer:
    """Synthesizes insights using LLM."""

    def __init__(
        self,
        llm_callback: Callable[[str], str] | None = None,
    ) -> None:
        """
        Initialize synthesizer.

        Args:
            llm_callback: Function that takes a prompt and returns LLM response.
        """
        self._llm_callback = llm_callback

    @property
    def available(self) -> bool:
        """Check if LLM is available."""
        return self._llm_callback is not None

    def synthesize(
        self,
        statistics: StatisticalMetrics,
        patterns: list[PatternInfo],
        circuit_info: dict[str, Any] | None = None,
    ) -> tuple[str, list[str]]:
        """
        Generate LLM synthesis of results.

        Returns:
            Tuple of (synthesis_text, suggested_questions).
        """
        if not self._llm_callback:
            return ("LLM not available for synthesis.", [])

        prompt = self._build_prompt(statistics, patterns, circuit_info)

        try:
            response = self._llm_callback(prompt)

            # Parse response for synthesis and questions
            synthesis = response
            questions = self._extract_questions(response)

            return synthesis, questions
        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}")
            return (f"LLM synthesis unavailable: {e}", [])

    def _build_prompt(
        self,
        statistics: StatisticalMetrics,
        patterns: list[PatternInfo],
        circuit_info: dict[str, Any] | None,
    ) -> str:
        """Build prompt for LLM."""
        lines = [
            "You are a quantum computing expert. Analyze these simulation results:",
            "",
            "**Statistics:**",
            f"- Total states: {statistics.total_states}",
            f"- Non-zero states: {statistics.non_zero_states}",
            f"- Entropy: {statistics.entropy:.3f} bits",
            f"- Dominant state: '{statistics.dominant_state}' ({statistics.dominant_probability:.1%})",
            f"- Effective dimension: {statistics.effective_dimension:.1f}",
            "",
            "**Detected Patterns:**",
        ]

        for pattern in patterns:
            lines.append(f"- {pattern.pattern_type.value}: {pattern.description}")

        if circuit_info:
            lines.extend(
                [
                    "",
                    "**Circuit Info:**",
                    f"- Qubits: {circuit_info.get('qubits', 'unknown')}",
                    f"- Gates: {circuit_info.get('gates', 'unknown')}",
                    f"- Depth: {circuit_info.get('depth', 'unknown')}",
                ]
            )

        lines.extend(
            [
                "",
                "Provide:",
                "1. A concise interpretation of what these results mean",
                "2. Any notable quantum phenomena observed",
                "3. Two follow-up questions the user might want to explore",
                "",
                "Keep response under 200 words.",
            ]
        )

        return "\n".join(lines)

    def _extract_questions(self, response: str) -> list[str]:
        """Extract suggested questions from LLM response."""
        questions = []

        # Look for numbered questions or question marks
        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if "?" in line and len(line) > 10:
                # Clean up the line
                clean = line.lstrip("0123456789.-) ")
                if clean and clean[0].isupper():
                    questions.append(clean)

        return questions[:3]  # Max 3 questions


# =============================================================================
# ENHANCED LLM INTEGRATION - Deeper Analysis Features
# =============================================================================


class EnhancedLLMAnalyzer:
    """Enhanced LLM analyzer with deeper integration for quantum insights.
    
    Provides:
    - Multi-turn contextual analysis
    - Quantum phenomena explanation
    - Circuit optimization suggestions
    - Comparative analysis across runs
    - Educational explanations
    - Error diagnosis assistance
    """
    
    # Specialized prompt templates for different analysis modes
    PROMPT_TEMPLATES = {
        "phenomena_explanation": """You are an expert quantum physicist. Explain the following quantum phenomenon observed in simulation results in a way that's educational yet accurate:

**Observed Phenomenon:** {phenomenon}

**Context:**
- Qubits: {qubits}
- Circuit type: {circuit_type}
- Key statistics: Entropy={entropy:.3f}, Dominant state='{dominant_state}' at {dominant_prob:.1%}

Provide:
1. What this phenomenon represents physically
2. Why it occurs in quantum systems
3. Real-world applications or implications
4. Connection to fundamental quantum principles

Keep explanation accessible but scientifically accurate. Use analogies where helpful.""",

        "optimization_suggestions": """As a quantum computing optimization expert, analyze this circuit simulation and suggest improvements:

**Circuit Metrics:**
- Qubits: {qubits}
- Gate count: {gate_count}
- Circuit depth: {depth}
- Two-qubit gates: {two_qubit_gates}

**Performance:**
- Execution time: {execution_time}ms
- Result entropy: {entropy:.3f} bits
- Dominant outcome probability: {dominant_prob:.1%}

**Detected Issues:**
{issues}

Provide specific, actionable optimization suggestions for:
1. Gate reduction strategies
2. Circuit depth optimization
3. Noise mitigation techniques
4. Hardware-specific improvements

Be specific and quantitative where possible.""",

        "comparative_analysis": """Compare these quantum simulation results across different backends/configurations:

**Run A ({backend_a}):**
{results_a}

**Run B ({backend_b}):**
{results_b}

Analyze:
1. Significant differences in output distributions
2. Potential causes for discrepancies
3. Which result is more trustworthy and why
4. Recommendations for resolving differences

Consider noise models, precision differences, and algorithmic variations.""",

        "error_diagnosis": """As a quantum error correction expert, diagnose potential issues in this simulation:

**Expected Behavior:** {expected}

**Actual Results:**
- Dominant state: '{dominant_state}' at {dominant_prob:.1%}
- Entropy: {entropy:.3f} bits
- Pattern detected: {pattern}

**Warning Signs:**
{warnings}

Diagnose:
1. Possible sources of error
2. Whether errors are systematic or random
3. Confidence level in the results
4. Suggested verification steps

Be specific about quantum error mechanisms.""",

        "educational_insight": """Explain these quantum simulation results for someone learning quantum computing:

**The Circuit:**
{circuit_description}

**What Happened:**
- Most common outcome: '{dominant_state}' ({dominant_prob:.1%})
- Number of possible states: {total_states}
- States with significant probability: {non_zero_states}
- Distribution type: {pattern}

Create an educational explanation that:
1. Explains what the results mean in plain language
2. Connects to quantum concepts (superposition, entanglement, measurement)
3. Uses helpful analogies
4. Suggests what to try next for learning

Target audience: intermediate programmer new to quantum computing.""",

        "multi_turn_context": """Continue the quantum analysis conversation. Previous context:

{conversation_history}

New question from user: {user_question}

Current simulation state:
- Results: {summary}
- Key findings: {key_findings}

Provide a helpful, contextual response that builds on the previous analysis.""",
    }
    
    def __init__(
        self,
        llm_callback: Callable[[str], str] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> None:
        """Initialize enhanced LLM analyzer.
        
        Args:
            llm_callback: Function that takes prompt and returns LLM response
            conversation_history: Optional list of previous exchanges
        """
        self._llm_callback = llm_callback
        self._conversation_history = conversation_history or []
        self._context_cache: dict[str, Any] = {}
    
    @property
    def available(self) -> bool:
        """Check if LLM is available."""
        return self._llm_callback is not None
    
    def explain_phenomenon(
        self,
        phenomenon: str,
        statistics: StatisticalMetrics,
        circuit_info: dict[str, Any] | None = None,
    ) -> str:
        """Generate educational explanation of a quantum phenomenon.
        
        Args:
            phenomenon: Name/description of the phenomenon
            statistics: Statistical metrics from simulation
            circuit_info: Optional circuit information
            
        Returns:
            Detailed explanation string
        """
        if not self._llm_callback:
            return self._fallback_phenomenon_explanation(phenomenon)
        
        prompt = self.PROMPT_TEMPLATES["phenomena_explanation"].format(
            phenomenon=phenomenon,
            qubits=circuit_info.get("qubits", "unknown") if circuit_info else "unknown",
            circuit_type=circuit_info.get("type", "quantum") if circuit_info else "quantum",
            entropy=statistics.entropy,
            dominant_state=statistics.dominant_state,
            dominant_prob=statistics.dominant_probability,
        )
        
        try:
            response = self._llm_callback(prompt)
            self._add_to_history("phenomenon_explanation", prompt, response)
            return response
        except Exception as e:
            logger.warning(f"LLM phenomenon explanation failed: {e}")
            return self._fallback_phenomenon_explanation(phenomenon)
    
    def suggest_optimizations(
        self,
        circuit_info: dict[str, Any],
        statistics: StatisticalMetrics,
        execution_time_ms: float,
        issues: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate circuit optimization suggestions.
        
        Args:
            circuit_info: Circuit metadata
            statistics: Simulation statistics
            execution_time_ms: Execution time
            issues: Known issues
            
        Returns:
            List of optimization suggestions with priorities
        """
        if not self._llm_callback:
            return self._fallback_optimizations(circuit_info)
        
        issues_text = "\n".join(f"- {issue}" for issue in (issues or [])) or "None detected"
        
        prompt = self.PROMPT_TEMPLATES["optimization_suggestions"].format(
            qubits=circuit_info.get("qubits", "unknown"),
            gate_count=circuit_info.get("gates", "unknown"),
            depth=circuit_info.get("depth", "unknown"),
            two_qubit_gates=circuit_info.get("two_qubit_gates", "unknown"),
            execution_time=execution_time_ms,
            entropy=statistics.entropy,
            dominant_prob=statistics.dominant_probability,
            issues=issues_text,
        )
        
        try:
            response = self._llm_callback(prompt)
            self._add_to_history("optimization", prompt, response)
            return self._parse_optimization_response(response)
        except Exception as e:
            logger.warning(f"LLM optimization suggestion failed: {e}")
            return self._fallback_optimizations(circuit_info)
    
    def compare_results(
        self,
        results_a: dict[str, Any],
        results_b: dict[str, Any],
        backend_a: str,
        backend_b: str,
    ) -> dict[str, Any]:
        """Compare results from different backends or configurations.
        
        Args:
            results_a: First set of results
            results_b: Second set of results
            backend_a: Name of first backend
            backend_b: Name of second backend
            
        Returns:
            Comparison analysis dictionary
        """
        if not self._llm_callback:
            return self._fallback_comparison(results_a, results_b, backend_a, backend_b)
        
        prompt = self.PROMPT_TEMPLATES["comparative_analysis"].format(
            backend_a=backend_a,
            backend_b=backend_b,
            results_a=self._format_results_for_prompt(results_a),
            results_b=self._format_results_for_prompt(results_b),
        )
        
        try:
            response = self._llm_callback(prompt)
            self._add_to_history("comparison", prompt, response)
            return {
                "analysis": response,
                "backend_a": backend_a,
                "backend_b": backend_b,
                "recommendation": self._extract_recommendation(response),
            }
        except Exception as e:
            logger.warning(f"LLM comparison failed: {e}")
            return self._fallback_comparison(results_a, results_b, backend_a, backend_b)
    
    def diagnose_errors(
        self,
        expected: str,
        statistics: StatisticalMetrics,
        patterns: list[PatternInfo],
        warnings: list[str],
    ) -> dict[str, Any]:
        """Diagnose potential errors or unexpected results.
        
        Args:
            expected: Description of expected behavior
            statistics: Actual statistics
            patterns: Detected patterns
            warnings: Warning messages
            
        Returns:
            Diagnosis dictionary with findings and recommendations
        """
        if not self._llm_callback:
            return self._fallback_diagnosis(expected, statistics, warnings)
        
        pattern_name = patterns[0].pattern_type.value if patterns else "unknown"
        warnings_text = "\n".join(f"- {w}" for w in warnings) or "None"
        
        prompt = self.PROMPT_TEMPLATES["error_diagnosis"].format(
            expected=expected,
            dominant_state=statistics.dominant_state,
            dominant_prob=statistics.dominant_probability,
            entropy=statistics.entropy,
            pattern=pattern_name,
            warnings=warnings_text,
        )
        
        try:
            response = self._llm_callback(prompt)
            self._add_to_history("diagnosis", prompt, response)
            return {
                "diagnosis": response,
                "severity": self._estimate_error_severity(statistics, expected),
                "confidence": self._estimate_diagnosis_confidence(response),
                "suggested_actions": self._extract_actions(response),
            }
        except Exception as e:
            logger.warning(f"LLM diagnosis failed: {e}")
            return self._fallback_diagnosis(expected, statistics, warnings)
    
    def generate_educational_insight(
        self,
        statistics: StatisticalMetrics,
        patterns: list[PatternInfo],
        circuit_description: str | None = None,
    ) -> str:
        """Generate educational explanation for learners.
        
        Args:
            statistics: Simulation statistics
            patterns: Detected patterns
            circuit_description: Optional circuit description
            
        Returns:
            Educational explanation string
        """
        if not self._llm_callback:
            return self._fallback_educational(statistics, patterns)
        
        pattern_name = patterns[0].pattern_type.value if patterns else "random"
        
        prompt = self.PROMPT_TEMPLATES["educational_insight"].format(
            circuit_description=circuit_description or "A quantum circuit simulation",
            dominant_state=statistics.dominant_state,
            dominant_prob=statistics.dominant_probability,
            total_states=statistics.total_states,
            non_zero_states=statistics.non_zero_states,
            pattern=pattern_name,
        )
        
        try:
            response = self._llm_callback(prompt)
            self._add_to_history("educational", prompt, response)
            return response
        except Exception as e:
            logger.warning(f"LLM educational insight failed: {e}")
            return self._fallback_educational(statistics, patterns)
    
    def continue_conversation(
        self,
        user_question: str,
        current_summary: str,
        key_findings: list[str],
    ) -> str:
        """Continue multi-turn analysis conversation.
        
        Args:
            user_question: User's follow-up question
            current_summary: Current results summary
            key_findings: Current key findings
            
        Returns:
            Contextual response
        """
        if not self._llm_callback:
            return "LLM not available for conversation continuation."
        
        history_text = self._format_conversation_history()
        findings_text = "\n".join(f"- {f}" for f in key_findings)
        
        prompt = self.PROMPT_TEMPLATES["multi_turn_context"].format(
            conversation_history=history_text,
            user_question=user_question,
            summary=current_summary,
            key_findings=findings_text,
        )
        
        try:
            response = self._llm_callback(prompt)
            self._add_to_history("conversation", user_question, response)
            return response
        except Exception as e:
            logger.warning(f"LLM conversation failed: {e}")
            return f"Unable to continue analysis: {e}"
    
    def _add_to_history(self, context_type: str, prompt: str, response: str) -> None:
        """Add exchange to conversation history."""
        self._conversation_history.append({
            "type": context_type,
            "prompt_summary": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "response_summary": response[:300] + "..." if len(response) > 300 else response,
        })
        # Keep only last 10 exchanges
        if len(self._conversation_history) > 10:
            self._conversation_history = self._conversation_history[-10:]
    
    def _format_conversation_history(self) -> str:
        """Format conversation history for context."""
        if not self._conversation_history:
            return "No previous conversation."
        
        lines = []
        for i, exchange in enumerate(self._conversation_history[-5:], 1):
            lines.append(f"[{i}] {exchange['type']}: {exchange['response_summary']}")
        return "\n".join(lines)
    
    def _format_results_for_prompt(self, results: dict[str, Any]) -> str:
        """Format results dictionary for LLM prompt."""
        lines = []
        for key, value in results.items():
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.4f}")
            elif isinstance(value, dict):
                lines.append(f"- {key}: {len(value)} items")
            else:
                lines.append(f"- {key}: {value}")
        return "\n".join(lines) or "No results"
    
    def _parse_optimization_response(self, response: str) -> list[dict[str, Any]]:
        """Parse optimization suggestions from LLM response."""
        suggestions = []
        lines = response.split("\n")
        current_suggestion = None
        
        for line in lines:
            line = line.strip()
            # Look for numbered items or bullet points
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
                if current_suggestion:
                    suggestions.append(current_suggestion)
                # Clean the line
                clean_line = line.lstrip("0123456789.-•) ")
                current_suggestion = {
                    "title": clean_line[:100],
                    "description": clean_line,
                    "priority": len(suggestions) + 1,  # Earlier = higher priority
                    "category": self._categorize_suggestion(clean_line),
                }
            elif current_suggestion and line:
                current_suggestion["description"] += " " + line
        
        if current_suggestion:
            suggestions.append(current_suggestion)
        
        return suggestions[:10]  # Max 10 suggestions
    
    def _categorize_suggestion(self, text: str) -> str:
        """Categorize an optimization suggestion."""
        text_lower = text.lower()
        if any(word in text_lower for word in ["gate", "cnot", "swap", "decompos"]):
            return "gate_optimization"
        if any(word in text_lower for word in ["depth", "layer", "parallel"]):
            return "depth_optimization"
        if any(word in text_lower for word in ["noise", "error", "decoher"]):
            return "noise_mitigation"
        if any(word in text_lower for word in ["memory", "gpu", "hardware"]):
            return "hardware_optimization"
        return "general"
    
    def _extract_recommendation(self, response: str) -> str:
        """Extract main recommendation from LLM response."""
        lines = response.split("\n")
        for line in lines:
            if "recommend" in line.lower():
                return line.strip()
        # Return last substantive line
        for line in reversed(lines):
            if len(line.strip()) > 20:
                return line.strip()
        return "See full analysis for recommendations."
    
    def _extract_actions(self, response: str) -> list[str]:
        """Extract suggested actions from LLM response."""
        actions = []
        lines = response.split("\n")
        
        for line in lines:
            line_lower = line.lower()
            if any(word in line_lower for word in ["try", "check", "verify", "run", "test", "add", "remove"]):
                clean = line.strip().lstrip("0123456789.-•) ")
                if len(clean) > 10:
                    actions.append(clean)
        
        return actions[:5]  # Max 5 actions
    
    def _estimate_error_severity(self, statistics: StatisticalMetrics, expected: str) -> str:
        """Estimate severity of error based on statistics."""
        # High entropy when expecting concentrated = likely error
        if statistics.entropy > 3.0 and "concentrated" in expected.lower():
            return "high"
        # Low dominant probability when expecting clear outcome
        if statistics.dominant_probability < 0.3 and "clear" in expected.lower():
            return "high"
        if statistics.dominant_probability < 0.5:
            return "medium"
        return "low"
    
    def _estimate_diagnosis_confidence(self, response: str) -> float:
        """Estimate confidence in diagnosis based on response."""
        response_lower = response.lower()
        
        # High confidence indicators
        if any(word in response_lower for word in ["clearly", "definitely", "certainly"]):
            return 0.9
        # Medium confidence
        if any(word in response_lower for word in ["likely", "probably", "appears"]):
            return 0.7
        # Low confidence
        if any(word in response_lower for word in ["possibly", "might", "unclear"]):
            return 0.5
        
        return 0.6  # Default moderate confidence
    
    # Fallback methods when LLM is unavailable
    
    def _fallback_phenomenon_explanation(self, phenomenon: str) -> str:
        """Provide fallback explanation without LLM."""
        explanations = {
            "entangled": "Entanglement is a quantum correlation between qubits where the state of one qubit is dependent on the state of another, even when separated.",
            "superposition": "Superposition means the qubit exists in multiple states simultaneously until measured, collapsing to a definite state.",
            "interference": "Quantum interference occurs when probability amplitudes add constructively or destructively, affecting measurement outcomes.",
            "ghz": "A GHZ state is a maximally entangled state of 3+ qubits, showing non-classical correlations.",
            "bell": "A Bell state is a maximally entangled two-qubit state demonstrating quantum non-locality.",
        }
        
        for key, explanation in explanations.items():
            if key in phenomenon.lower():
                return explanation
        
        return f"'{phenomenon}' is a quantum phenomenon observed in these results. Enable LLM for detailed explanation."
    
    def _fallback_optimizations(self, circuit_info: dict[str, Any]) -> list[dict[str, Any]]:
        """Provide fallback optimizations without LLM."""
        return [
            {
                "title": "Reduce two-qubit gate count",
                "description": "Consider decomposing multi-qubit gates into simpler operations",
                "priority": 1,
                "category": "gate_optimization",
            },
            {
                "title": "Parallelize independent operations",
                "description": "Gates on different qubits can often be parallelized to reduce depth",
                "priority": 2,
                "category": "depth_optimization",
            },
            {
                "title": "Consider noise mitigation",
                "description": "Apply error mitigation techniques for more accurate results",
                "priority": 3,
                "category": "noise_mitigation",
            },
        ]
    
    def _fallback_comparison(
        self,
        results_a: dict[str, Any],
        results_b: dict[str, Any],
        backend_a: str,
        backend_b: str,
    ) -> dict[str, Any]:
        """Provide fallback comparison without LLM."""
        return {
            "analysis": f"Comparison of {backend_a} vs {backend_b}. Enable LLM for detailed analysis.",
            "backend_a": backend_a,
            "backend_b": backend_b,
            "recommendation": "Run with LLM enabled for detailed comparison.",
        }
    
    def _fallback_diagnosis(
        self,
        expected: str,
        statistics: StatisticalMetrics,
        warnings: list[str],
    ) -> dict[str, Any]:
        """Provide fallback diagnosis without LLM."""
        severity = self._estimate_error_severity(statistics, expected)
        
        return {
            "diagnosis": f"Results may deviate from expected ({expected}). Dominant state: '{statistics.dominant_state}'. Enable LLM for detailed diagnosis.",
            "severity": severity,
            "confidence": 0.5,
            "suggested_actions": [
                "Verify circuit construction",
                "Check for measurement errors",
                "Compare with another backend",
            ],
        }
    
    def _fallback_educational(
        self,
        statistics: StatisticalMetrics,
        patterns: list[PatternInfo],
    ) -> str:
        """Provide fallback educational content without LLM."""
        pattern_name = patterns[0].pattern_type.value if patterns else "random"
        
        return f"""**Simulation Results Explained**

Your quantum circuit produced a {pattern_name} distribution with {statistics.non_zero_states} different outcomes.

The most common result was '{statistics.dominant_state}' appearing {statistics.dominant_probability:.1%} of the time.

Key concept: In quantum computing, we measure probabilities rather than certainties. The entropy of {statistics.entropy:.2f} bits tells us how "spread out" the results are.

To learn more, try:
- Modifying gate parameters to see how outcomes change
- Adding more qubits to explore scaling
- Comparing different quantum backends

Enable LLM for more detailed educational explanations."""


# =============================================================================
# Main Insight Engine
# =============================================================================


class InsightEngine:
    """
    Main engine for generating insights from quantum simulation results.

    Example usage:
        engine = InsightEngine()
        report = engine.analyze({"00": 0.5, "11": 0.5})
        print(report.summary)

        # For deeper LLM-powered analysis:
        engine = InsightEngine(llm_callback=my_llm_function)
        explanation = engine.explain_phenomenon("Bell state entanglement", probabilities)
        suggestions = engine.get_optimization_suggestions(probabilities, circuit_info)
    """

    def __init__(
        self,
        llm_callback: Callable[[str], str] | None = None,
    ) -> None:
        """
        Initialize the insight engine.

        Args:
            llm_callback: Optional callback for LLM-enhanced insights.
                Should take a prompt string and return the LLM response.
        """
        self._stat_analyzer = StatisticalAnalyzer()
        self._amp_analyzer = AmplitudeAnalyzer()
        self._pattern_detector = PatternDetector()
        self._recommendation_engine = RecommendationEngine()
        self._viz_recommender = VisualizationRecommender()
        self._llm_synthesizer = LLMSynthesizer(llm_callback)
        self._enhanced_llm = EnhancedLLMAnalyzer(llm_callback)
        self._ascii_viz = ASCIIVisualizer()

    @property
    def enhanced_llm(self) -> EnhancedLLMAnalyzer:
        """Access the enhanced LLM analyzer for deeper analysis."""
        return self._enhanced_llm

    @property
    def llm_available(self) -> bool:
        """Check if LLM integration is available."""
        return self._enhanced_llm.available

    def analyze(
        self,
        probabilities: dict[str, float],
        amplitudes: dict[str, complex] | None = None,
        circuit_info: dict[str, Any] | None = None,
        level: InsightLevel = InsightLevel.STANDARD,
    ) -> InsightReport:
        """
        Analyze simulation results and generate insights.

        Args:
            probabilities: Mapping of state -> probability.
            amplitudes: Optional mapping of state -> complex amplitude.
            circuit_info: Optional circuit metadata for context.
            level: Level of detail for analysis.

        Returns:
            InsightReport with complete analysis.
        """
        # Statistical analysis
        statistics = self._stat_analyzer.analyze(probabilities)

        # Amplitude analysis (if available)
        amplitude_analysis = None
        if amplitudes:
            amplitude_analysis = self._amp_analyzer.analyze(amplitudes)

        # Pattern detection
        patterns = self._pattern_detector.detect(probabilities)

        # Generate recommendations
        recommendations = []
        if level in (InsightLevel.DETAILED, InsightLevel.EXPERT):
            recommendations = self._recommendation_engine.generate(statistics, patterns)

        # Visualization recommendations
        visualizations = self._viz_recommender.recommend(
            statistics, patterns, amplitudes is not None
        )

        # Generate summary and key findings
        summary = self._generate_summary(statistics, patterns)
        key_findings = self._generate_key_findings(statistics, patterns)

        # LLM synthesis (for expert level)
        llm_synthesis = None
        llm_questions: list[str] = []
        if level == InsightLevel.EXPERT and self._llm_synthesizer.available:
            llm_synthesis, llm_questions = self._llm_synthesizer.synthesize(
                statistics, patterns, circuit_info
            )

        # Generate warnings
        warnings = self._generate_warnings(statistics)

        return InsightReport(
            summary=summary,
            key_findings=key_findings,
            statistics=statistics,
            amplitude_analysis=amplitude_analysis,
            patterns=patterns,
            recommendations=recommendations,
            visualizations=visualizations,
            llm_synthesis=llm_synthesis,
            llm_questions=llm_questions,
            analysis_level=level,
            warnings=warnings,
        )

    def quick_analyze(
        self,
        probabilities: dict[str, float],
    ) -> str:
        """
        Quick analysis returning just a summary string.

        Args:
            probabilities: Mapping of state -> probability.

        Returns:
            Brief summary string.
        """
        report = self.analyze(probabilities, level=InsightLevel.BASIC)
        return report.summary

    def _generate_summary(
        self,
        statistics: StatisticalMetrics,
        patterns: list[PatternInfo],
    ) -> str:
        """Generate a summary of the results."""
        parts = []

        # Dominant result
        if statistics.dominant_probability > 0.8:
            parts.append(
                f"Strong measurement of state '{statistics.dominant_state}' "
                f"with {statistics.dominant_probability:.1%} probability."
            )
        elif statistics.dominant_probability > 0.4:
            parts.append(
                f"State '{statistics.dominant_state}' is most likely "
                f"at {statistics.dominant_probability:.1%}."
            )
        else:
            parts.append(f"Results spread across {statistics.non_zero_states} states.")

        # Pattern information
        main_pattern = patterns[0] if patterns else None
        if main_pattern and main_pattern.pattern_type != PatternType.RANDOM:
            parts.append(main_pattern.description)

        # Entropy characterization
        max_entropy = (
            math.log2(statistics.total_states) if statistics.total_states > 0 else 0
        )
        if max_entropy > 0:
            entropy_ratio = statistics.entropy / max_entropy
            if entropy_ratio < 0.2:
                parts.append("Distribution is highly concentrated.")
            elif entropy_ratio > 0.8:
                parts.append("Distribution is nearly uniform.")

        return " ".join(parts)

    def _generate_key_findings(
        self,
        statistics: StatisticalMetrics,
        patterns: list[PatternInfo],
    ) -> list[str]:
        """Generate list of key findings."""
        findings = []

        # Top state
        findings.append(
            f"Most probable state: '{statistics.dominant_state}' "
            f"({statistics.dominant_probability:.1%})"
        )

        # Entropy
        findings.append(f"Shannon entropy: {statistics.entropy:.2f} bits")

        # Effective dimension
        findings.append(
            f"Effective dimension: {statistics.effective_dimension:.1f} "
            f"out of {statistics.total_states} states"
        )

        # Pattern findings
        for pattern in patterns:
            if pattern.confidence > 0.7:
                findings.append(
                    f"Detected {pattern.pattern_type.value} pattern "
                    f"({pattern.confidence:.0%} confidence)"
                )

        return findings

    def _generate_warnings(
        self,
        statistics: StatisticalMetrics,
    ) -> list[str]:
        """Generate warnings about potential issues."""
        warnings = []

        # Check for suspicious results
        if statistics.max_probability > 0.999:
            warnings.append(
                "Result shows 100% probability for single state - "
                "verify circuit has measurements"
            )

        if statistics.non_zero_states == 1:
            warnings.append(
                "Only one state has non-zero probability - "
                "may indicate deterministic circuit"
            )

        return warnings

    # =========================================================================
    # Enhanced LLM Integration Methods
    # =========================================================================

    def explain_phenomenon(
        self,
        phenomenon: str,
        probabilities: dict[str, float],
        circuit_info: dict[str, Any] | None = None,
    ) -> str:
        """
        Get an educational explanation of a quantum phenomenon.

        Args:
            phenomenon: The phenomenon to explain (e.g., "Bell state entanglement")
            probabilities: Current simulation results
            circuit_info: Optional circuit metadata

        Returns:
            LLM-generated explanation or fallback description
        """
        statistics = self._stat_analyzer.analyze(probabilities)
        return self._enhanced_llm.explain_phenomenon(
            phenomenon, statistics, circuit_info
        )

    def get_optimization_suggestions(
        self,
        probabilities: dict[str, float],
        circuit_info: dict[str, Any] | None = None,
    ) -> str:
        """
        Get circuit optimization suggestions based on results.

        Args:
            probabilities: Simulation results
            circuit_info: Circuit metadata (gate count, depth, etc.)

        Returns:
            Optimization suggestions
        """
        statistics = self._stat_analyzer.analyze(probabilities)
        patterns = self._pattern_detector.detect(probabilities)
        return self._enhanced_llm.suggest_optimizations(
            statistics, patterns, circuit_info or {}
        )

    def compare_results(
        self,
        results_a: dict[str, float],
        results_b: dict[str, float],
        backend_a: str = "Backend A",
        backend_b: str = "Backend B",
    ) -> str:
        """
        Compare results from two different backends or configurations.

        Args:
            results_a: First set of results
            results_b: Second set of results
            backend_a: Name of first backend
            backend_b: Name of second backend

        Returns:
            Comparative analysis
        """
        stats_a = self._stat_analyzer.analyze(results_a)
        stats_b = self._stat_analyzer.analyze(results_b)
        return self._enhanced_llm.compare_results(
            stats_a, stats_b, backend_a, backend_b
        )

    def diagnose_errors(
        self,
        probabilities: dict[str, float],
        expected_behavior: str,
        circuit_info: dict[str, Any] | None = None,
    ) -> str:
        """
        Diagnose potential errors in simulation results.

        Args:
            probabilities: Actual simulation results
            expected_behavior: Description of expected output
            circuit_info: Optional circuit metadata

        Returns:
            Error diagnosis and suggestions
        """
        statistics = self._stat_analyzer.analyze(probabilities)
        patterns = self._pattern_detector.detect(probabilities)
        return self._enhanced_llm.diagnose_errors(
            statistics, patterns, expected_behavior, circuit_info
        )

    def get_educational_insight(
        self,
        probabilities: dict[str, float],
        circuit_info: dict[str, Any] | None = None,
    ) -> str:
        """
        Get an educational explanation suitable for learners.

        Args:
            probabilities: Simulation results
            circuit_info: Optional circuit metadata

        Returns:
            Educational explanation
        """
        statistics = self._stat_analyzer.analyze(probabilities)
        patterns = self._pattern_detector.detect(probabilities)
        return self._enhanced_llm.generate_educational_insight(
            statistics, patterns, circuit_info
        )

    def ask_followup(
        self,
        question: str,
        probabilities: dict[str, float],
    ) -> str:
        """
        Ask a follow-up question about current results.

        Args:
            question: User's question
            probabilities: Current simulation results

        Returns:
            Contextual response
        """
        statistics = self._stat_analyzer.analyze(probabilities)
        patterns = self._pattern_detector.detect(probabilities)
        key_findings = self._generate_key_findings(statistics, patterns)
        
        summary = self._generate_summary(statistics, patterns)
        findings_str = "; ".join(key_findings[:3])
        
        return self._enhanced_llm.continue_conversation(
            question, summary, findings_str
        )

    def render_visualization(
        self,
        probabilities: dict[str, float],
        viz_type: str = "bar",
        **kwargs: Any,
    ) -> str:
        """
        Render an ASCII visualization of the results.

        Args:
            probabilities: State -> probability mapping.
            viz_type: Type of visualization ('bar', 'histogram', 'sparkline',
                      'heatmap', 'correlations', 'distribution').
            **kwargs: Additional options for the visualization.

        Returns:
            ASCII visualization string.
        """
        if viz_type == "bar":
            return self._ascii_viz.render_bar_chart(
                probabilities,
                top_k=kwargs.get("top_k", 10),
                show_values=kwargs.get("show_values", True),
            )
        elif viz_type == "histogram":
            return self._ascii_viz.render_histogram(
                probabilities,
                bins=kwargs.get("bins", 10),
            )
        elif viz_type == "sparkline":
            values = list(sorted(probabilities.values(), reverse=True))
            return self._ascii_viz.render_sparkline(
                values,
                width=kwargs.get("width"),
            )
        elif viz_type == "correlations":
            return self._ascii_viz.render_qubit_correlations(probabilities)
        elif viz_type == "distribution":
            return self._ascii_viz.render_state_distribution(probabilities)
        else:
            return f"Unknown visualization type: {viz_type}"

    def full_report(
        self,
        probabilities: dict[str, float],
        amplitudes: dict[str, complex] | None = None,
        circuit_info: dict[str, Any] | None = None,
        include_viz: bool = True,
    ) -> str:
        """
        Generate a complete text report with visualizations.

        Args:
            probabilities: State -> probability mapping.
            amplitudes: Optional amplitudes.
            circuit_info: Optional circuit metadata.
            include_viz: Whether to include ASCII visualizations.

        Returns:
            Complete report as string.
        """
        report = self.analyze(
            probabilities, amplitudes, circuit_info, InsightLevel.DETAILED
        )

        lines = [
            "=" * 60,
            "QUANTUM SIMULATION ANALYSIS REPORT",
            "=" * 60,
            "",
            "## Summary",
            report.summary,
            "",
            "## Key Findings",
        ]

        for finding in report.key_findings:
            lines.append(f"  • {finding}")

        lines.extend(["", "## Patterns Detected"])
        for pattern in report.patterns:
            lines.append(
                f"  • {pattern.pattern_type.value} ({pattern.confidence:.0%}): "
                f"{pattern.description}"
            )

        if include_viz:
            lines.extend([
                "",
                "## State Probability Distribution",
                self.render_visualization(probabilities, "bar", top_k=8),
            ])

            # Add correlation matrix for multi-qubit systems
            if report.statistics.total_states > 4:
                try:
                    corr_viz = self.render_visualization(
                        probabilities, "correlations"
                    )
                    if "must be binary" not in corr_viz.lower():
                        lines.extend([
                            "",
                            "## Qubit Correlations",
                            corr_viz,
                        ])
                except Exception:
                    pass  # Skip if correlation rendering fails

        if report.recommendations:
            lines.extend(["", "## Recommendations"])
            for rec in report.recommendations:
                priority_marker = "!" * (4 - rec.priority)
                lines.append(f"  [{priority_marker}] {rec.title}: {rec.description}")

        if report.warnings:
            lines.extend(["", "## Warnings"])
            for warning in report.warnings:
                lines.append(f"  ⚠ {warning}")

        if report.llm_synthesis:
            lines.extend([
                "",
                "## AI Analysis",
                report.llm_synthesis,
            ])

        lines.extend(["", "=" * 60])

        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================


def analyze_results(
    probabilities: dict[str, float],
    amplitudes: dict[str, complex] | None = None,
) -> InsightReport:
    """
    Quick analysis of simulation results.

    Args:
        probabilities: Mapping of state -> probability.
        amplitudes: Optional mapping of state -> amplitude.

    Returns:
        InsightReport with analysis.
    """
    engine = InsightEngine()
    return engine.analyze(probabilities, amplitudes)


def summarize_results(probabilities: dict[str, float]) -> str:
    """
    Get a quick one-line summary of results.

    Args:
        probabilities: Mapping of state -> probability.

    Returns:
        Summary string.
    """
    engine = InsightEngine()
    return engine.quick_analyze(probabilities)


def visualize_results(
    probabilities: dict[str, float],
    viz_type: str = "bar",
    **kwargs: Any,
) -> str:
    """
    Generate ASCII visualization of results.

    Args:
        probabilities: Mapping of state -> probability.
        viz_type: Type of visualization ('bar', 'histogram', 'sparkline',
                  'correlations', 'distribution').
        **kwargs: Additional visualization options.

    Returns:
        ASCII visualization string.
    """
    engine = InsightEngine()
    return engine.render_visualization(probabilities, viz_type, **kwargs)


def full_analysis_report(
    probabilities: dict[str, float],
    amplitudes: dict[str, complex] | None = None,
    circuit_info: dict[str, Any] | None = None,
) -> str:
    """
    Generate a complete analysis report with visualizations.

    Args:
        probabilities: Mapping of state -> probability.
        amplitudes: Optional amplitudes.
        circuit_info: Optional circuit metadata.

    Returns:
        Complete report as string.
    """
    engine = InsightEngine()
    return engine.full_report(probabilities, amplitudes, circuit_info)
