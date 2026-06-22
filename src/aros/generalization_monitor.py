"""Generalization Monitor — detects overfitting and benchmark gaming.

Hidden benchmark scores are never exposed to the Research Agent.
The monitor alerts when visible scores improve while hidden scores
worsen, indicating benchmark overfitting.
"""

from typing import Optional

from .config import Config
from .database import ExperimentDB
from .models import ExperimentFeedback


class GeneralizationMonitor:
    """Monitors generalization by comparing visible vs hidden benchmark trends."""

    def __init__(self, config: Config, db: ExperimentDB):
        self.config = config
        self.db = db
        self.alert_threshold = 0.05  # 5% divergence triggers alert
        self.pause_threshold = -0.15  # -15% generalization score triggers pause

    def compute_generalization_score(self, experiment_id: str) -> float:
        """Compute generalization score = hidden_avg_delta - visible_avg_delta.

        Positive score means generalization is improving.
        Negative score means hidden benchmarks are regressing relative to visible.
        """
        results = self.db.get_benchmark_results(experiment_id)
        if not results:
            return 0.0

        visible_deltas = []
        hidden_deltas = []

        for r in results:
            if r.delta is not None:
                # We need to know if it's hidden — check via benchmark name
                if r.benchmark_name in self.config.evaluator.hidden_benchmarks:
                    hidden_deltas.append(r.delta)
                else:
                    visible_deltas.append(r.delta)

        avg_visible = sum(visible_deltas) / len(visible_deltas) if visible_deltas else 0.0
        avg_hidden = sum(hidden_deltas) / len(hidden_deltas) if hidden_deltas else 0.0

        return round(avg_hidden - avg_visible, 4)

    def check_alert_conditions(self, experiment_id: str) -> list[str]:
        """Check for conditions that should trigger alerts.

        Returns a list of alert messages. Empty list = all clear.
        """
        alerts = []

        # 1. Check generalization score for this experiment
        gen_score = self.compute_generalization_score(experiment_id)
        if gen_score < -self.alert_threshold:
            alerts.append(f"Generalization score {gen_score:.3f} below alert threshold ({-self.alert_threshold})")

        # 2. Check trend over last N experiments
        recent_completed = self.db.list_experiments(limit=10)
        scores = []
        for exp in recent_completed:
            if exp.id == experiment_id:
                continue
            try:
                s = self.compute_generalization_score(exp.id)
                scores.append(s)
            except Exception:
                continue

        if len(scores) >= 3:
            # Check if the last 3 scores are declining
            last_3 = scores[:3]
            if all(last_3[i] < last_3[i-1] for i in range(1, 3)):
                alerts.append(f"Generalization declining over last 3 experiments: {[round(s, 3) for s in last_3]}")

            # Check for divergence: visible improves, hidden worsens
            recent_with_results = []
            for exp in recent_completed:
                results = self.db.get_benchmark_results(exp.id)
                if results:
                    vis = [r.delta for r in results if r.delta is not None and r.benchmark_name not in self.config.evaluator.hidden_benchmarks]
                    hid = [r.delta for r in results if r.delta is not None and r.benchmark_name in self.config.evaluator.hidden_benchmarks]
                    if vis and hid:
                        recent_with_results.append((sum(vis)/len(vis), sum(hid)/len(hid)))

            if len(recent_with_results) >= 3:
                last_vis = sum(v for v, h in recent_with_results[:3]) / 3
                last_hid = sum(h for v, h in recent_with_results[:3]) / 3
                if last_vis > 0 and last_hid < 0:
                    alerts.append(f"Divergence detected: visible avg +{last_vis:.2f} vs hidden avg {last_hid:.2f}")

        return alerts

    def should_pause_autonomy(self) -> bool:
        """Check if autonomous execution should be paused.

        Returns True if alert conditions are severe enough to require
        human intervention.
        """
        recent = self.db.list_experiments(limit=10)
        if not recent:
            return False

        bad_count = 0
        for exp in recent:
            try:
                gen_score = self.compute_generalization_score(exp.id)
                if gen_score < self.pause_threshold:
                    bad_count += 1
            except Exception:
                continue

        # Pause if more than 40% of recent experiments have bad gen scores
        pause = bad_count >= 3 and (bad_count / len(recent)) > 0.4
        return pause
