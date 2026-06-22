"""Generalization Monitor — detects overfitting and benchmark gaming.

Hidden benchmark scores are recorded internally and never exposed to the
Research Agent. The monitor alerts when visible scores improve while
hidden scores worsen, indicating benchmark overfitting.
"""

from typing import Optional

from .config import Config
from .database import ExperimentDB


class GeneralizationMonitor:
    """Monitors generalization by comparing visible vs hidden benchmark trends."""

    def __init__(self, config: Config, db: ExperimentDB):
        self.config = config
        self.db = db
        self.alert_threshold = 0.05  # 5% divergence triggers alert

    def compute_generalization_score(self, experiment_id: str) -> float:
        """Compute generalization score = hidden_delta - visible_delta.

        Positive score means generalization is improving.
        Negative score means hidden benchmarks are regressing.
        """
        # --- Implementation placeholder ---
        # TODO:
        # 1. Get visible benchmark results for this experiment
        # 2. Get hidden benchmark results for this experiment
        # 3. Compute average deltas
        # 4. Return generalization score
        return 0.0

    def check_alert_conditions(self, experiment_id: str) -> list[str]:
        """Check for conditions that should trigger alerts.

        Returns a list of alert messages. Empty list = all clear.
        """
        alerts = []

        # --- Implementation placeholder ---
        # TODO: Implement actual alert logic:
        # 1. Hidden scores declined repeatedly (last N experiments)
        # 2. Visible improves while hidden worsens (divergence)
        # 3. Generalization score crosses threshold

        return alerts

    def should_pause_autonomy(self) -> bool:
        """Check if autonomous execution should be paused.

        Returns True if alert conditions are severe enough to require
        human intervention.
        """
        return False
