"""Experiment Runner — executes approved experiments.

Handles dataset assembly, training configuration, LoRA setup,
training execution, checkpoint storage, logging, and artifact management.
The Research Agent never interacts directly with compute resources.
"""

from typing import Optional

from .config import Config
from .database import ExperimentDB
from .models import Experiment, ExperimentConfig


class ExperimentRunner:
    """Executes approved experiments on compute resources."""

    def __init__(self, config: Config, db: ExperimentDB):
        self.config = config
        self.db = db

    def assemble_dataset(self, config: ExperimentConfig) -> dict:
        """Assemble the dataset mixture for an experiment.

        Returns metadata about the assembled dataset (paths, sizes, ratios).
        """
        # --- Implementation placeholder ---
        # TODO: 
        # 1. Resolve dataset IDs from registry
        # 2. Apply weightings
        # 3. Build curriculum schedule if specified
        # 4. Return assembled dataset metadata
        return {"status": "not_implemented"}

    def setup_training(self, config: ExperimentConfig) -> dict:
        """Prepare training configuration.

        Sets up LoRA config, optimizer, scheduler, and checkpointing.
        """
        return {
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "status": "not_implemented",
        }

    def run_training(self, experiment_id: str) -> bool:
        """Execute the training run for an experiment.

        This is the heavy lifter — handles actual model training.
        """
        # --- Implementation placeholder ---
        # TODO: 
        # 1. Update experiment status to RUNNING
        # 2. Launch training process
        # 3. Monitor progress, handle failures
        # 4. Save checkpoints
        # 5. Update experiment with metrics
        return False

    def get_checkpoint_path(self, experiment_id: str) -> Optional[str]:
        """Return the path to the experiment's checkpoint."""
        return None

    def estimate_cost(self, config: ExperimentConfig) -> dict:
        """Estimate compute cost for a given configuration."""
        return {
            "gpu_hours_estimate": 0.0,
            "estimated_cost": 0.0,
        }
