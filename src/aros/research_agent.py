"""Research Agent — the system's scientist.

Generates hypotheses, analyzes experiment history, detects patterns,
and proposes strategic directions. Does NOT train models or choose
exact hyperparameters — that's the Search Engine's job.
"""

from typing import Optional

from .config import Config
from .database import ExperimentDB
from .models import Hypothesis, Experiment


class ResearchAgent:
    """The Research Agent analyzes experiment history and forms hypotheses."""

    def __init__(self, config: Config, db: ExperimentDB):
        self.config = config
        self.db = db
        self.model = config.research_agent.model
        self.provider = config.research_agent.provider

    def analyze_history(self, limit: int = 20) -> list[Experiment]:
        """Retrieve recent experiment history for analysis."""
        experiments = self.db.list_experiments(limit=limit)
        hypotheses = self.db.list_hypotheses(limit=limit)
        return experiments

    def propose_hypotheses(self, recent_experiments: Optional[list[Experiment]] = None) -> list[Hypothesis]:
        """Generate new research hypotheses based on experiment history.

        In the full implementation, this calls an LLM. For now, returns
        a placeholder hypothesis demonstrating the interface.
        """
        # --- Implementation placeholder ---
        # TODO: Call LLM with experiment history to generate hypotheses
        # The LLM should receive:
        #   - Recent experiment feedback (losses, benchmark deltas, forgetting index)
        #   - Recent hypotheses and their outcomes
        #   - Current exploration/exploitation balance
        # The LLM should return structured hypotheses with reasoning
        return []

    def recommend_dataset_changes(self) -> list[dict]:
        """Recommend changes to the dataset composition.

        Returns a list of recommendation dicts with 'action', 'dataset_id',
        and 'reasoning'.
        """
        return []

    def explain_reasoning(self, hypothesis_id: str) -> str:
        """Return the reasoning trace for a given hypothesis."""
        hyp = self.db.get_hypothesis(hypothesis_id)
        if hyp:
            return hyp.reasoning
        return "Hypothesis not found."
