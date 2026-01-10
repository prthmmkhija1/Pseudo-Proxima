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
                    description=(f"Bell-state-like entanglement: |{all_zeros}⟩ + |{all_ones}⟩"),
                    affected_states=[all_zeros, all_ones],
                    metrics={"combined_probability": combined},
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
# Main Insight Engine
# =============================================================================


class InsightEngine:
    """
    Main engine for generating insights from quantum simulation results.

    Example usage:
        engine = InsightEngine()
        report = engine.analyze({"00": 0.5, "11": 0.5})
        print(report.summary)
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
        max_entropy = math.log2(statistics.total_states) if statistics.total_states > 0 else 0
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
                "Only one state has non-zero probability - " "may indicate deterministic circuit"
            )

        return warnings


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
