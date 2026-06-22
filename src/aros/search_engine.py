"""Search Engine — converts hypotheses into executable experiments.

Takes strategic directions from the Research Agent and performs
numerical exploration to produce concrete experiment configurations.
Supports simple grid and random search for the local MVP.
"""

import random
from typing import Optional

from .config import Config
from .database import ExperimentDB
from .models import Hypothesis, ExperimentConfig, SearchSpace


# Default parameter ranges for the search space
DEFAULT_PARAM_GRID = {
    "learning_rate": {
        "type": "float",
        "min": 1e-5,
        "max": 5e-4,
        "num_values": 5,
        "log_scale": True,
    },
    "lora_rank": {
        "type": "int",
        "min": 4,
        "max": 64,
        "num_values": 4,
    },
    "lora_alpha": {
        "type": "int",
        "min": 8,
        "max": 64,
        "num_values": 4,
    },
    "batch_size": {
        "type": "int",
        "values": [8, 16, 32],
    },
    "epochs": {
        "type": "int",
        "values": [2, 3, 5],
    },
    "weight_decay": {
        "type": "float",
        "min": 0.0,
        "max": 0.1,
        "num_values": 3,
    },
    "warmup_ratio": {
        "type": "float",
        "min": 0.0,
        "max": 0.2,
        "num_values": 3,
    },
}


def _generate_grid_values(param_def: dict) -> list:
    """Generate discrete values from a parameter definition."""
    ptype = param_def["type"]

    if "values" in param_def:
        return param_def["values"]

    if ptype == "float":
        if param_def.get("log_scale", False):
            import math
            lo = math.log10(param_def["min"])
            hi = math.log10(param_def["max"])
            step = (hi - lo) / (param_def["num_values"] - 1) if param_def["num_values"] > 1 else 0
            return [round(10 ** (lo + i * step), 8) for i in range(param_def["num_values"])]
        else:
            step = (param_def["max"] - param_def["min"]) / (param_def["num_values"] - 1) if param_def["num_values"] > 1 else 0
            return [round(param_def["min"] + i * step, 6) for i in range(param_def["num_values"])]

    if ptype == "int":
        step = max(1, (param_def["max"] - param_def["min"]) // (param_def["num_values"] - 1)) if param_def["num_values"] > 1 else 1
        return [param_def["min"] + i * step for i in range(param_def["num_values"])]

    return []


class SearchEngine:
    """Numerical optimizer that converts hypotheses into experiments."""

    def __init__(self, config: Config, db: ExperimentDB):
        self.config = config
        self.db = db
        self.algorithm = config.search_engine.algorithm
        self.experiments_per_hypothesis = config.search_engine.experiments_per_hypothesis
        self.search_space = SearchSpace()

    def expand_hypothesis(self, hypothesis: Hypothesis) -> list[ExperimentConfig]:
        """Convert a hypothesis into a set of concrete experiment configurations.

        Uses the hypothesis's target_region to guide parameter selection,
        then generates N variations around the baseline.
        """
        configs = []

        # Parse target region for guidance
        target = hypothesis.target_region or {}

        # Base config
        base = self.get_baseline_config()

        # Apply any target region hints (e.g., "learning_rate": "1e-4 to 3e-4")
        for param_name, param_range in target.items():
            if isinstance(param_range, str) and "to" in param_range:
                parts = param_range.replace("%", "").split("to")
                if len(parts) == 2:
                    try:
                        lo, hi = float(parts[0].strip()), float(parts[1].strip())
                        target[param_name] = {"min": lo, "max": hi}
                    except ValueError:
                        pass

        # Generate N config variations
        for i in range(self.experiments_per_hypothesis):
            cfg = self._vary_config(base, target, i)
            configs.append(cfg)

        return configs

    def _vary_config(self, base: ExperimentConfig, target: dict, seed_offset: int) -> ExperimentConfig:
        """Create a varied configuration, using target to guide ranges."""
        rng = random.Random(seed_offset)
        cfg = ExperimentConfig(
            dataset_ids=list(base.dataset_ids),
            dataset_weights=dict(base.dataset_weights),
            learning_rate=base.learning_rate,
            batch_size=base.batch_size,
            epochs=base.epochs,
            lora_rank=base.lora_rank,
            lora_alpha=base.lora_alpha,
            weight_decay=base.weight_decay,
            warmup_ratio=base.warmup_ratio,
            extras=dict(base.extras),
        )

        # Apply targeted variations based on hypothesis
        for param_name, param_hint in target.items():
            if isinstance(param_hint, dict) and "min" in param_hint and "max" in param_hint:
                lo, hi = param_hint["min"], param_hint["max"]
                if param_name in ("learning_rate", "weight_decay", "warmup_ratio"):
                    value = _log_uniform(rng, lo, hi)
                else:
                    value = rng.uniform(lo, hi)

                if hasattr(cfg, param_name):
                    setattr(cfg, param_name, value)
                else:
                    cfg.extras[param_name] = value

        # Apply small random perturbations to remaining params for diversity
        if not any(k in target for k in ("learning_rate",)):
            cfg.learning_rate = _log_uniform(rng, 1e-5, 5e-4)

        if not any(k in target for k in ("lora_rank",)):
            cfg.lora_rank = rng.choice([4, 8, 16, 32, 64])
            cfg.lora_alpha = cfg.lora_rank * 2

        if not any(k in target for k in ("batch_size",)):
            cfg.batch_size = rng.choice([8, 16, 32])

        return cfg

    def get_baseline_config(self) -> ExperimentConfig:
        """Return a default/baseline experiment configuration."""
        return ExperimentConfig(
            dataset_ids=[],
            dataset_weights={},
            learning_rate=5e-5,
            batch_size=16,
            epochs=3,
            lora_rank=16,
            lora_alpha=32,
            weight_decay=0.01,
            warmup_ratio=0.1,
        )

    def register_search_space(self, space: SearchSpace):
        """Register a formal search space definition."""
        self.search_space = space

    def suggest_next_parameters(self, experiment_ids: list[str]) -> dict:
        """Given completed experiment IDs, suggest next parameter set.

        Simple random suggestion for the MVP. Future: Bayesian Optimization.
        """
        return {
            "learning_rate": _log_uniform(random, 1e-5, 5e-4),
            "lora_rank": random.choice([8, 16, 32]),
        }


def _log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    """Sample from a log-uniform distribution."""
    import math
    log_lo = math.log10(lo)
    log_hi = math.log10(hi)
    return round(10 ** rng.uniform(log_lo, log_hi), 8)
