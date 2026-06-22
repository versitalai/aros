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
    """Simulates training to produce realistic-looking metrics.

    Given a config, produces:
    - Training loss: starts high, decreases with more epochs
    - Validation loss: similar but slightly higher
    - Forgetting index: increases with more parameter updates
    - Benchmark improvements: small positive effects from most configs
    """

    def simulate(self, config: ExperimentConfig) -> dict:
        """Run simulated training and return metrics."""
        rng = random.Random(hash(str(config.to_dict())) % (2**31))

        base_train_loss = 2.0
        base_val_loss = 2.2

        # Better params = lower loss
        lr_quality = 1.0 - abs(math.log10(config.learning_rate) - 4.5) / 2.0
        lr_quality = max(0.3, min(1.0, lr_quality))

        # More epochs = more training but more forgetting
        epoch_factor = min(1.0, config.epochs / 5.0)

        train_loss = base_train_loss * (0.8 - 0.3 * lr_quality) * (0.9 ** epoch_factor) + rng.uniform(-0.05, 0.05)
        val_loss = train_loss * (1.0 + 0.1 * lr_quality) + rng.uniform(-0.03, 0.03)

        # Forgetting index: higher with more aggressive training
        forgetting = 0.1 + 0.3 * epoch_factor + 0.2 * (1.0 - lr_quality)
        forgetting += rng.uniform(-0.05, 0.05)
        forgetting = max(0.0, min(1.0, forgetting))

        # GPU hours: scales with epochs, batch size, rank
        gpu_hours = config.epochs * 0.5 * (1.0 + config.lora_rank / 32.0) * (16.0 / config.batch_size)
        gpu_hours = round(gpu_hours + rng.uniform(-0.2, 0.2), 2)

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
