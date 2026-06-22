"""Search Engine — converts hypotheses into executable experiments.

Takes strategic directions from the Research Agent and performs
numerical exploration to produce concrete experiment configurations.
Supports Bayesian Optimization, Evolutionary Search, Optuna, and Tree Search.
"""

from typing import Optional

from .config import Config
from .database import ExperimentDB
from .models import Hypothesis, Experiment, ExperimentConfig, SearchSpace


class SearchEngine:
    """Numerical optimizer that converts hypotheses into experiments."""

    def __init__(self, config: Config, db: ExperimentDB):
        self.config = config
        self.db = db
        self.algorithm = config.search_engine.algorithm
        self.search_space = SearchSpace()

    def expand_hypothesis(self, hypothesis: Hypothesis) -> list[ExperimentConfig]:
        """Convert a hypothesis into a set of concrete experiment configurations.

        For a hypothesis like 'increase planning data', this might produce
        configurations at 20%, 25%, 30%, 35%, 40% planning ratios.
        """
        # --- Implementation placeholder ---
        # TODO: Implement actual search algorithm
        # 1. Parse target_region from hypothesis
        # 2. Apply algorithm (bayesian, evolutionary, grid)
        # 3. Generate n experiment configs
        # 4. Return list of ExperimentConfig
        return []

    def get_baseline_config(self) -> ExperimentConfig:
        """Return a default/baseline experiment configuration."""
        return ExperimentConfig()

    def register_search_space(self, space: SearchSpace):
        """Register a formal search space definition."""
        self.search_space = space

    def suggest_next_parameters(self, experiment_ids: list[str]) -> dict:
        """Given completed experiment IDs, suggest next parameter set.

        Used for iterative optimization (Bayesian Optimization loop).
        """
        return {}
