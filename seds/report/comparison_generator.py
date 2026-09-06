"""Report generator for SEDS Phase G evaluation comparison."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from domains.base import Score


class ComparisonReportGenerator:
    """
    Report generator for comparing baseline vs evolved RolloutResult sets.

    Produces multi-axis comparison reports:
    - Accuracy + Confidence Intervals
    - Reliability (failure types distribution)
    - Cost (time/memory usage)
    - Speed (task duration distribution)
    """

    def __init__(
        self,
        baseline_name: str,
        evolved_name: str,
        baseline_results: List[Score],
        evolved_results: List[Score],
    ):
        """
        Initialize the report generator.

        Args:
            baseline_name: Name of the baseline system
            evolved_name: Name of the evolved system
            baseline_results: List of Score objects from baseline
            evolved_results: List of Score objects from evolved system
        """
        self.baseline_name = baseline_name
        self.evolved_name = evolved_name
        self.baseline_results = baseline_results
        self.evolved_results = evolved_results

    def _calculate_pass_rate(self, scores: List[Score]) -> Tuple[float, float, float, float]:
        """Calculate pass rate and 95% CI."""
        n = len(scores)
        if n == 0:
            return 0.0, 0.0, 0.0, 0.0

        passes = [1.0 if s.correct else 0.0 for s in scores]
        p = sum(passes) / n

        if n >= 2:
            se = (p * (1 - p) / n) ** 0.5
            ci_low = p - 1.96 * se
            ci_high = p + 1.96 * se
        else:
            ci_low = p
            ci_high = p

        return p, ci_low, ci_high, n

    def _extract_partial_score(self, score: Score) -> float:
        """Extract partial score (0.0–1.0) from Score."""
        return score.partial

    def _extract_cost_from_score(self, score: Score) -> float:
        """Extract cost metric from Score detail."""
        cost = (
            score.detail.get("time_cost_ms", score.detail.get("cost_ms", 0.0))
            or 0.0
        )
        return float(cost)

    def _extract_speed_from_score(self, score: Score) -> float:
        """Extract speed metric (time in seconds) from Score detail."""
        speed = (
            score.detail.get("duration_sec", score.detail.get("duration_ms", 0.0) / 1000.0)
            or 0.0
        )
        return float(speed)

    def _group_failures_by_category(self, scores: List[Score]) -> Dict[str, int]:
        """Group failures by category."""
        categories = {}
        for score in scores:
            if not score.correct:  # Failed
                detail = score.detail.get("failure_category", "unknown") or "unknown"
                categories[detail] = categories.get(detail, 0) + 1
        return categories

    def _calculate_reliability_metrics(
        self, baseline_scores: List[Score], evolved_scores: List[Score]
    ) -> Dict[str, Any]:
        """Calculate reliability metrics (failure type distribution)."""
        baseline_failures = self._group_failures_by_category(baseline_scores)
        evolved_failures = self._group_failures_by_category(evolved_scores)
        all_categories = set(baseline_failures.keys()) | set(evolved_failures.keys())

        reliability = {"categories": {}}
        for cat in sorted(all_categories):
            reliability["categories"][cat] = {
                "baseline": baseline_failures.get(cat, 0),
                "evolved": evolved_failures.get(cat, 0),
            }
        return reliability

    def _calculate_cost_metrics(
        self, baseline_scores: List[Score], evolved_scores: List[Score]
    ) -> Dict[str, Any]:
        """Calculate cost metrics (time/memory usage)."""
        baseline_costs = [self._extract_cost_from_score(s) for s in baseline_scores]
        evolved_costs = [self._extract_cost_from_score(s) for s in evolved_scores]

        def summarize(costs: List[float]) -> Dict[str, Any]:
            if not costs:
                return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
            costs_sorted = sorted(costs)
            return {
                "mean": sum(costs) / len(costs),
                "median": costs_sorted[len(costs) // 2],
                "min": costs_sorted[0],
                "max": costs_sorted[-1],
            }

        return {
            "baseline": summarize(baseline_costs),
            "evolved": summarize(evolved_costs),
        }

    def _calculate_speed_metrics(
        self, baseline_scores: List[Score], evolved_scores: List[Score]
    ) -> Dict[str, Any]:
        """Calculate speed metrics (task duration distribution)."""
        baseline_speeds = [self._extract_speed_from_score(s) for s in baseline_scores]
        evolved_speeds = [self._extract_speed_from_score(s) for s in evolved_scores]

        def summarize(speeds: List[float]) -> Dict[str, Any]:
            if not speeds:
                return {"mean": 0.0, "median": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0}
            speeds_sorted = sorted(speeds)
            n = len(speeds_sorted)
            return {
                "mean": sum(speeds_sorted) / n,
                "median": speeds_sorted[n // 2],
                "p50": speeds_sorted[n // 2],
                "p90": speeds_sorted[int(n * 0.9)],
                "p95": speeds_sorted[int(n * 0.95)],
            }

        return {
            "baseline": summarize(baseline_speeds),
            "evolved": summarize(evolved_speeds),
        }

    def _analyze_baseline_vs_evolved(self) -> Dict[str, Any]:
        """Analyze baseline vs evolved performance across all axes."""
        baseline_pass, baseline_ci_low, baseline_ci_high, n_baseline = self._calculate_pass_rate(
            self.baseline_results
        )
        evolved_pass, evolved_ci_low, evolved_ci_high, n_evolved = self._calculate_pass_rate(
            self.evolved_results
        )

        accuracy_metrics = {
            "baseline": {"pass_rate": baseline_pass, "ci_low": baseline_ci_low, "ci_high": baseline_ci_high, "n_samples": n_baseline},
            "evolved": {"pass_rate": evolved_pass, "ci_low": evolved_ci_low, "ci_high": evolved_ci_high, "n_samples": n_evolved},
        }

        reliability_metrics = self._calculate_reliability_metrics(self.baseline_results, self.evolved_results)

        cost_metrics = self._calculate_cost_metrics(self.baseline_results, self.evolved_results)

        speed_metrics = self._calculate_speed_metrics(self.baseline_results, self.evolved_results)

        return {
            "accuracy": accuracy_metrics,
            "reliability": reliability_metrics,
            "cost": cost_metrics,
            "speed": speed_metrics,
        }

    def _generate_failure_histogram_shift(self, reliability: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate failure histogram shift for Pareto plot."""
        shifts = []
        for category, counts in reliability["categories"].items():
            baseline_count = counts["baseline"]
            evolved_count = counts["evolved"]
            shift = evolved_count - baseline_count

            shifts.append({
                "category": category,
                "baseline_count": baseline_count,
                "evolved_count": evolved_count,
                "shift": shift,
            })

        return sorted(shifts, key=lambda x: -abs(x["shift"]))

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate the full comparison report.

        Returns:
            Dictionary containing all analysis results
        """
        analysis = self._analyze_baseline_vs_evolved()
        failure_histogram_shift = self._generate_failure_histogram_shift(analysis["reliability"])

        report = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "baseline_system": self.baseline_name,
                "evolved_system": self.evolved_name,
                "total_baseline_samples": len(self.baseline_results),
                "total_evolved_samples": len(self.evolved_results),
            },
            "accuracy": analysis["accuracy"],
            "reliability": analysis["reliability"],
            "cost": analysis["cost"],
            "speed": analysis["speed"],
            "failure_histogram_shift": failure_histogram_shift,
        }

        return report

    def save_report(
        self,
        report: Dict[str, Any],
        output_path: Path,
    ) -> None:
        """Save report to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

    def save_json(
        self,
        data: Dict[str, Any],
        output_path: Path,
    ) -> None:
        """Save JSON data to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
