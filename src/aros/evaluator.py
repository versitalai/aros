"""Evaluator — the sole authority on experiment success.

Runs visible and hidden benchmarks, produces objective metrics.
For the MVP, this simulates benchmark results based on experiment
config and training metrics, producing plausible scores.
"""

import random
from typing import Optional

from .config import Config
from .database import ExperimentDB
from .models import BenchmarkResult, ExperimentFeedback, Experiment
from .benchmark_registry import create_default_registry


class Evaluator:
    """Independent evaluation authority. Runs benchmarks and produces metrics."""

    def __init__(self, config: Config, db: ExperimentDB):
        self.config = config
        self.db = db
        self.registry = create_default_registry()
        self.visible_benchmarks = config.evaluator.visible_benchmarks
        self.hidden_benchmarks = config.evaluator.hidden_benchmarks

    def evaluate(self, experiment_id: str) -> ExperimentFeedback:
        """Run all benchmarks and produce feedback.

        Benchmark scores are derived from actual training metrics:
        - training_loss directly maps to score (lower loss = higher score)
        - forgetting_index penalizes scores
        - Scores vary deterministically by config fingerprint
        """
        exp = self.db.get_experiment(experiment_id)
        if exp is None:
            return ExperimentFeedback(
                experiment_id=experiment_id,
                training_loss=0.0,
                validation_loss=0.0,
                notes="Experiment not found",
            )

        fp = exp.config.fingerprint() if exp.config else experiment_id
        rng = random.Random(hash(fp) % (2**31))

        training_loss = exp.metrics.get("training_loss", 1.0)
        val_loss = exp.metrics.get("validation_loss", 1.2)
        forgetting = exp.metrics.get("forgetting_index", 0.3)

        # Lower training_loss → higher scores
        quality_boost = max(-10, min(25, (2.0 - training_loss) * 12))

        # Forgetting index directly penalizes (especially planning)
        forgetting_penalty = forgetting * 8

        baseline = 68.0

        # Run visible benchmarks — each responds differently to forgetting
        visible_results = []
        for bm_name in self.visible_benchmarks:
            if bm_name == "planning":
                # Planning is hit hardest by forgetting
                score = baseline + quality_boost - forgetting_penalty + rng.uniform(-2.0, 4.0)
            elif bm_name == "coding":
                # Coding moderately affected
                score = baseline + quality_boost - forgetting_penalty * 0.5 + rng.uniform(-3.0, 5.0)
            else:
                # Agent tasks least affected
                score = baseline + quality_boost - forgetting_penalty * 0.3 + rng.uniform(-2.5, 4.5)
            score = max(30.0, min(100.0, score))
            delta = rng.uniform(-2.0, 3.0)

            visible_results.append(BenchmarkResult(
                benchmark_name=bm_name,
                score=round(score, 1),
                delta=round(delta, 1),
            ))
            self.db.save_benchmark_result(experiment_id, visible_results[-1], is_hidden=False)

        # Run hidden benchmarks — more sensitive to forgetting
        hidden_results = []
        for bm_name in self.hidden_benchmarks:
            score = baseline + quality_boost * 0.7 - forgetting_penalty * 1.2 + rng.uniform(-3.0, 3.0)
            score = max(30.0, min(100.0, score))

            hidden_results.append(BenchmarkResult(
                benchmark_name=bm_name,
                score=round(score, 1),
                delta=round(rng.uniform(-2.0, 2.0), 1),
            ))
            self.db.save_benchmark_result(experiment_id, hidden_results[-1], is_hidden=True)

        forgetting_index = exp.metrics.get("forgetting_index", 0.3)
        gpu_hours = exp.metrics.get("gpu_hours", 1.0)

        notes_parts = []
        if visible_results:
            best = max(visible_results, key=lambda r: r.score)
            worst = min(visible_results, key=lambda r: r.score)
            notes_parts.append(f"{best.benchmark_name} improved most ({best.score:.1f})")
            notes_parts.append(f"{worst.benchmark_name} weakest ({worst.score:.1f})")
        if forgetting_index > 0.5:
            notes_parts.append("moderate forgetting detected")
        notes = "; ".join(notes_parts) if notes_parts else "No notable patterns"

        feedback = ExperimentFeedback(
            experiment_id=experiment_id,
            training_loss=round(training_loss, 4),
            validation_loss=round(val_loss, 4),
            benchmark_results=visible_results,
            forgetting_index=round(forgetting_index, 3),
            gpu_hours=gpu_hours,
            notes=notes,
        )

        # Update the experiment with feedback
        exp.feedback = feedback
        self.db.update_experiment(exp)

        return feedback

    def evaluate_visible(self, experiment_id: str) -> list[BenchmarkResult]:
        """Run only the visible benchmarks."""
        exp = self.db.get_experiment(experiment_id)
        if exp is None:
            return []
        return self.db.get_benchmark_results(experiment_id)

    def evaluate_hidden(self, experiment_id: str) -> list[BenchmarkResult]:
        """Run hidden benchmarks. Results are stored internally."""
        exp = self.db.get_experiment(experiment_id)
        if exp is None:
            return []
        return self.db.get_benchmark_results(experiment_id)

    def store_hidden_results(self, experiment_id: str, results: list[BenchmarkResult]):
        """Store hidden benchmark results without exposing them to the agent."""
        for result in results:
            self.db.save_benchmark_result(experiment_id, result, is_hidden=True)
