"""Search Engine — converts hypotheses into executable experiments.

Takes strategic directions from the Research Agent and performs
numerical exploration to produce concrete experiment configurations.

Supports:
- Config-driven search space (loaded from YAML)
- Budget tracking per hypothesis
- Iterative refinement based on past results
- Deduplication via config fingerprinting
"""

import copy
import math
import random
from typing import Optional

from .config import Config
from .database import ExperimentDB
from .models import Hypothesis, ExperimentConfig, ExperimentStatus


class SearchEngine:
    """Numerical optimizer that converts hypotheses into experiments."""

    def __init__(self, config: Config, db: ExperimentDB):
        self.config = config
        self.db = db
        self.algorithm = config.search_engine.algorithm
        self.experiments_per_hypothesis = config.search_engine.experiments_per_hypothesis
        self.default_budget = config.search_engine.default_budget
        self.top_k = config.search_engine.top_k_to_exploit
        self.param_defs = config.search_space.parameters or {}

    def _get_param_values(self, param_name: str, param_def: dict) -> list:
        """Generate discrete values from a parameter definition."""
        ptype = param_def.get("type", "float")

        if "values" in param_def:
            return param_def["values"]

        if ptype == "float":
            lo, hi = param_def.get("min", 0), param_def.get("max", 1)
            n = param_def.get("num_values", 5)
            if n < 2:
                return [round((lo + hi) / 2, 6)]
            if param_def.get("log_scale", False):
                log_lo = math.log10(lo)
                log_hi = math.log10(hi)
                step = (log_hi - log_lo) / (n - 1)
                return [round(10 ** (log_lo + i * step), 8) for i in range(n)]
            step = (hi - lo) / (n - 1)
            return [round(lo + i * step, 6) for i in range(n)]

        if ptype == "int":
            lo, hi = param_def.get("min", 0), param_def.get("max", 10)
            n = param_def.get("num_values", 4)
            if n < 2:
                return [int(round((lo + hi) / 2))]
            step = max(1, (hi - lo) // (n - 1))
            return [lo + i * step for i in range(n)]

        return []

    def get_baseline_config(self) -> ExperimentConfig:
        """Return a default/baseline experiment configuration."""
        cfg = ExperimentConfig()
        # Override from param_defs if available
        pd = self.param_defs
        if "learning_rate" in pd:
            vals = self._get_param_values("learning_rate", pd["learning_rate"])
            if vals:
                cfg.learning_rate = vals[len(vals) // 2]
        if "lora_rank" in pd:
            vals = self._get_param_values("lora_rank", pd["lora_rank"])
            if vals:
                cfg.lora_rank = vals[len(vals) // 2]
        if "lora_alpha" in pd:
            vals = self._get_param_values("lora_alpha", pd["lora_alpha"])
            if vals:
                cfg.lora_alpha = vals[len(vals) // 2]
        if "batch_size" in pd:
            vals = self._get_param_values("batch_size", pd["batch_size"])
            if vals:
                cfg.batch_size = vals[len(vals) // 2]
        if "epochs" in pd:
            vals = self._get_param_values("epochs", pd["epochs"])
            if vals:
                cfg.epochs = vals[len(vals) // 2]
        return cfg

    def _has_budget(self, hypothesis: Hypothesis) -> bool:
        """Check if a hypothesis still has budget remaining."""
        return hypothesis.budget_spent < (hypothesis.budget_total or self.default_budget)

    def _is_duplicate(self, config: ExperimentConfig) -> bool:
        """Check if a config has already been tried."""
        fp = config.fingerprint()
        existing = self.db.get_experiments_by_fingerprint(fp)
        return len(existing) > 0

    def expand_hypothesis(self, hypothesis: Hypothesis) -> list[ExperimentConfig]:
        """Convert a hypothesis into a set of concrete experiment configurations.

        Uses the config-driven search space, hypothesis target_region for
        guidance, and checks budget + dedup constraints.
        """
        baseline = self.get_baseline_config()
        target = hypothesis.target_region or {}

        # Parse string ranges like "1e-4 to 3e-4" into min/max dicts
        for key, val in list(target.items()):
            if isinstance(val, str) and "to" in val:
                parts = val.replace("%", "").split("to")
                if len(parts) == 2:
                    try:
                        lo, hi = float(parts[0].strip()), float(parts[1].strip())
                        target[key] = {"min": lo, "max": hi}
                    except ValueError:
                        pass

        configs = []
        max_to_generate = self.experiments_per_hypothesis * 3
        attempts = 0

        while len(configs) < self.experiments_per_hypothesis and attempts < max_to_generate:
            if not self._has_budget(hypothesis):
                break

            cfg = self._vary_config(baseline, target, attempts)

            # Dedup check
            if self._is_duplicate(cfg):
                attempts += 1
                continue

            configs.append(cfg)
            attempts += 1

        # Apply the best-performing configs from prior cycles (if any)
        hypothesis.budget_spent += len(configs)
        self.db.save_hypothesis(hypothesis)

        return configs

    def _vary_config(self, base: ExperimentConfig, target: dict, seed_offset: int) -> ExperimentConfig:
        """Create a varied configuration, using target/param_defs to guide ranges."""
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

        pd = self.param_defs

        # Apply targeted variations from hypothesis
        for param_name, hint in target.items():
            if isinstance(hint, dict) and "min" in hint and "max" in hint:
                lo, hi = hint["min"], hint["max"]
                if param_name in ("learning_rate", "weight_decay", "warmup_ratio"):
                    value = self._log_uniform(rng, lo, hi)
                else:
                    value = rng.uniform(lo, hi)

                if hasattr(cfg, param_name):
                    setattr(cfg, param_name, value)
                else:
                    cfg.extras[param_name] = value

        # Fill remaining params from search space
        for param_name, pdef in pd.items():
            if param_name in target:
                continue  # already set above

            values = self._get_param_values(param_name, pdef)
            if not values:
                continue

            value = rng.choice(values)

            # Map param name to config attribute
            if hasattr(cfg, param_name):
                if param_name == "learning_rate":
                    cfg.learning_rate = value
                elif param_name == "lora_rank":
                    cfg.lora_rank = value
                    cfg.lora_alpha = value * 2
                elif param_name == "lora_alpha":
                    cfg.lora_alpha = value
                elif param_name == "batch_size":
                    cfg.batch_size = value
                elif param_name == "epochs":
                    cfg.epochs = value
                elif param_name == "weight_decay":
                    cfg.weight_decay = value
                elif param_name == "warmup_ratio":
                    cfg.warmup_ratio = value
                else:
                    cfg.extras[param_name] = value
            else:
                cfg.extras[param_name] = value

        return cfg

    def suggest_next_parameters(self, hypothesis_id: str) -> list[ExperimentConfig]:
        """Given a hypothesis with past results, suggest next parameter set.

        Picks the top-k best-performing experiment configs for this hypothesis
        and generates refinements around them.
        """
        hypothesis = self.db.get_hypothesis(hypothesis_id)
        if not hypothesis:
            return []

        past_experiments = self.db.list_experiments(limit=50)

        # Find experiments for this hypothesis with scores
        scored = []
        for exp in past_experiments:
            if exp.hypothesis_id != hypothesis_id or not exp.feedback:
                continue
            if not exp.feedback.benchmark_results:
                continue
            avg_score = sum(r.score for r in exp.feedback.benchmark_results) / len(exp.feedback.benchmark_results)
            scored.append((avg_score, exp))

        if not scored:
            return []

        scored.sort(key=lambda x: -x[0])
        top_configs = [exp.config for _, exp in scored[:self.top_k]]

        # Generate refinements around top configs
        refinements = []
        for i, cfg in enumerate(top_configs):
            if not self._has_budget(hypothesis):
                break
            refined = self._refine_config(cfg, i)
            if not self._is_duplicate(refined):
                refinements.append(refined)
                hypothesis.budget_spent += 1

        if refinements:
            self.db.save_hypothesis(hypothesis)

        return refinements[:self.experiments_per_hypothesis]

    def _refine_config(self, config: ExperimentConfig, seed_offset: int) -> ExperimentConfig:
        """Create a slightly refined version of a config, narrowing around best values."""
        rng = random.Random(seed_offset + 999)
        cfg = copy.deepcopy(config)

        pd = self.param_defs

        # Slightly perturb learning rate (narrow range)
        if "learning_rate" in pd:
            factor = 1 + rng.uniform(-0.3, 0.3)
            cfg.learning_rate = round(cfg.learning_rate * factor, 8)

        # Slightly adjust epochs
        if "epochs" in pd:
            cfg.epochs = max(1, cfg.epochs + rng.choice([-1, 0, 1]))

        return cfg

    def register_search_space(self, params: dict):
        """Register parameter definitions from config."""
        self.param_defs = params

    @staticmethod
    def _log_uniform(rng: random.Random, lo: float, hi: float) -> float:
        """Sample from a log-uniform distribution."""
        log_lo = math.log10(lo)
        log_hi = math.log10(hi)
        return round(10 ** rng.uniform(log_lo, log_hi), 8)
