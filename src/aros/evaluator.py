"""Evaluator — the sole authority on experiment success.

The Research Agent cannot self-evaluate. The Evaluator runs benchmarks
(visible and hidden) and produces objective metrics.
"""

from typing import Optional

from .config import Config
from .database import ExperimentDB
from .models import BenchmarkResult, Experiment, ExperimentFeedback


class Evaluator:
    """Independent evaluation authority. Runs benchmarks and produces metrics."""

    def __init__(self, config: Config, db: ExperimentDB):
        self.config = config
        self.db = db
        self.visible_benchmarks = config.evaluator.visible_benchmarks
        self.hidden_benchmarks = config.evaluator.hidden_benchmarks

    def evaluate(self, experiment_id: str) -> ExperimentFeedback:
        """Run all registered benchmarks and produce feedback.

        Returns an ExperimentFeedback object with benchmark results,
        loss metrics, and forgetting index.
        """
        # --- Implementation placeholder ---
        # TODO:
        # 1. Load model checkpoint for experiment
        # 2. Run visible benchmarks
        # 3. Run hidden benchmarks (store but don't return to agent)
        # 4. Compute forgetting index
        # 5. Return ExperimentFeedback with visible results
        return ExperimentFeedback(
            experiment_id=experiment_id,
            training_loss=0.0,
            validation_loss=0.0,
            benchmark_results=[],
            forgetting_index=0.0,
            notes="Evaluation not yet implemented.",
        )

    def evaluate_visible(self, experiment_id: str) -> list[BenchmarkResult]:
        """Run only the visible benchmarks."""
        return []

    def evaluate_hidden(self, experiment_id: str) -> list[BenchmarkResult]:
        """Run hidden benchmarks. Results are stored internally,
        never exposed to the Research Agent."""
        return []

    def store_hidden_results(self, experiment_id: str, results: list[BenchmarkResult]):
        """Store hidden benchmark results without exposing them to the agent."""
        for result in results:
            self.db.save_benchmark_result(experiment_id, result, is_hidden=True)
