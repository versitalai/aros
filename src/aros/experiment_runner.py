"""Experiment Runner — executes approved experiments.

For the MVP phase, this runs a mock training process that simulates
realistic training metrics. This validates the full pipeline without
requiring GPU hardware. The mock produces plausible loss curves,
benchmark improvements, and forgetting index values.
"""

import random
import math
from datetime import datetime
from typing import Optional

from .config import Config
from .database import ExperimentDB
from .models import Experiment, ExperimentConfig


class SimulatedTrainer:
    """Simulates training to produce metrics that respond to config changes.

    Key behaviors:
    - Learning rate: optimal ~5e-5, too high or too low degrades
    - Epochs: more epochs = lower loss but more forgetting
    - LoRA rank: higher rank = more capacity but more forgetting
    - Batch size: moderate batch sizes ideal, extremes degrade
    - Deterministic per config fingerprint
    """

    def simulate(self, config: ExperimentConfig) -> dict:
        """Run simulated training and return metrics."""
        fp = config.fingerprint() if config else "default"
        rng = random.Random(hash(fp) % (2**31))

        # Learning rate: sweet spot around 5e-5, degrades on either side
        log_lr = math.log10(config.learning_rate) if config.learning_rate > 0 else -5
        lr_score = 1.0 - abs(log_lr - (-4.3)) / 2.5
        lr_score = max(0.1, min(1.0, lr_score))

        # Epochs: more is better for loss, worse for forgetting
        epoch_norm = min(1.0, config.epochs / 5.0)

        # LoRA rank: higher = more capacity, more forgetting
        rank_norm = min(1.0, config.lora_rank / 64.0)

        # Batch size: moderate is ideal
        bs_norm = abs(math.log2(config.batch_size) - 4.5) / 2.5
        bs_score = max(0.3, 1.0 - bs_norm)

        # Combined training quality
        quality = 0.5 * lr_score + 0.2 * epoch_norm + 0.15 * rank_norm + 0.15 * bs_score
        quality = max(0.2, min(1.0, quality))

        # Loss decreases with quality and epochs
        train_loss = 2.5 - 1.5 * quality - 0.3 * epoch_norm
        train_loss = round(train_loss + rng.uniform(-0.08, 0.08), 4)
        train_loss = max(0.3, train_loss)

        val_loss = round(train_loss * (1.05 + 0.05 * (1.0 - quality)) + rng.uniform(-0.05, 0.05), 4)
        val_loss = max(0.3, val_loss)

        # Forgetting: higher with more epochs and higher rank
        forgetting = 0.1 + 0.4 * epoch_norm + 0.3 * rank_norm - 0.1 * lr_score
        forgetting += rng.uniform(-0.05, 0.05)
        forgetting = max(0.0, min(1.0, forgetting))

        # GPU hours
        gpu_hours = round(
            config.epochs * 0.5 * (1.0 + config.lora_rank / 32.0) * (16.0 / config.batch_size)
            + rng.uniform(-0.2, 0.2),
            2
        )

        return {
            "training_loss": round(train_loss, 4),
            "validation_loss": round(val_loss, 4),
            "forgetting_index": round(forgetting, 3),
            "gpu_hours": gpu_hours,
            "status": "completed",
        }


class ExperimentRunner:
    """Executes approved experiments on compute resources."""

    def __init__(self, config: Config, db: ExperimentDB):
        self.config = config
        self.db = db
        self.simulator = SimulatedTrainer()

    def assemble_dataset(self, config: ExperimentConfig) -> dict:
        """Assemble the dataset mixture for an experiment."""
        return {
            "status": "assembled",
            "datasets": config.dataset_ids,
            "weights": config.dataset_weights,
            "total_examples": sum(1 for _ in config.dataset_ids) * 10000,
            "mixture_validated": True,
        }

    def setup_training(self, config: ExperimentConfig) -> dict:
        """Prepare training configuration."""
        return {
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "weight_decay": config.weight_decay,
            "warmup_ratio": config.warmup_ratio,
            "model": "llama3.2:3b",
            "status": "configured",
        }

    def run_training(self, experiment_id: str) -> bool:
        """Execute the training run for an experiment.

        For the MVP, this runs simulated training and stores results.
        The caller is responsible for status transitions.
        """
        exp = self.db.get_experiment(experiment_id)
        if exp is None:
            return False

        if exp.config is None:
            exp.config = ExperimentConfig()
            self.db.update_experiment(exp)

        # Simulate training
        metrics = self.simulator.simulate(exp.config)
        exp.metrics = metrics
        exp.results = {
            "training_completed": True,
            "checkpoints_saved": True,
        }

        self.db.update_experiment(exp)
        return True

    def get_checkpoint_path(self, experiment_id: str) -> Optional[str]:
        """Return the path to the experiment's checkpoint."""
        exp = self.db.get_experiment(experiment_id)
        if exp and exp.status in (ExperimentStatus.RUNNING, ExperimentStatus.EVALUATING, ExperimentStatus.COMPLETED):
            return f"/tmp/aros/checkpoints/{experiment_id}"
        return None

    def estimate_cost(self, config: ExperimentConfig) -> dict:
        """Estimate compute cost for a given configuration."""
        gpu_hours = config.epochs * 0.5 * (1.0 + config.lora_rank / 32.0) * (16.0 / config.batch_size)
        return {
            "gpu_hours_estimate": round(gpu_hours, 2),
            "estimated_cost": round(gpu_hours * 1.5, 2),  # $1.50/hr hypothetical
        }
