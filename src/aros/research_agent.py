"""Research Agent — the system's scientist.

Generates hypotheses by calling a local LLM (Ollama) that analyzes
experiment history, detects patterns, and proposes strategic directions.
"""

import json
import uuid
from typing import Optional

import httpx

from .config import Config
from .database import ExperimentDB
from .models import Hypothesis, Experiment, ExperimentStatus


# System prompt for the Research Agent
SYSTEM_PROMPT = """You are a senior AI research scientist. Your job is to analyze experiment results and propose the next research hypothesis.

You have access to:
1. Recent experiment history (configs, metrics, and feedback)
2. A set of available datasets with metadata (quality, novelty, topic coverage)
3. A search space of tunable parameters

Your job is NOT to pick exact hyperparameters. Your job is to propose STRATEGIC DIRECTIONS.

A good hypothesis:
- Identifies a pattern in the data ("coding improved but planning regressed")
- Proposes a specific change ("increase planning data proportion")
- Provides a rough target region ("20-40% of training data")
- Includes a confidence level (0.0 to 1.0)
- Explains the reasoning clearly
- Flags whether this is exploitation (refining a known area) or exploration (testing something new)

You must respond in JSON format ONLY, with no additional text:

{
    "hypotheses": [
        {
            "description": "Clear, specific description of the hypothesis",
            "target_region": {"parameter_name": "value_or_range"},
            "confidence": 0.78,
            "reasoning": "Step-by-step reasoning based on experiment data",
            "is_exploration": false
        }
    ]
}

If you cannot generate a hypothesis (e.g., not enough data), return {"hypotheses": []}.
"""


def _build_experiment_context(experiments: list[Experiment]) -> str:
    """Build a readable summary of recent experiments for the LLM."""
    if not experiments:
        return "No experiments have been run yet. This is the first cycle."

    lines = ["Recent Experiments:", ""]
    for exp in experiments:
        lines.append(f"--- Experiment {exp.id} ---")
        lines.append(f"  Status: {exp.status.value}")
        lines.append(f"  Hypothesis: {exp.hypothesis_id or 'N/A'}")
        lines.append(f"  Config:")
        if exp.config:
            lines.append(f"    Dataset IDs: {exp.config.dataset_ids}")
            lines.append(f"    Learning Rate: {exp.config.learning_rate}")
            lines.append(f"    LoRA Rank: {exp.config.lora_rank}")
            lines.append(f"    Epochs: {exp.config.epochs}")
        lines.append(f"  Results: {json.dumps(exp.results, indent=4)}")
        lines.append(f"  Metrics: {json.dumps(exp.metrics, indent=4)}")
        if exp.feedback:
            fb = exp.feedback
            lines.append(f"  Feedback:")
            lines.append(f"    Training Loss: {fb.training_loss}")
            lines.append(f"    Validation Loss: {fb.validation_loss}")
            lines.append(f"    Forgetting Index: {fb.forgetting_index}")
            lines.append(f"    GPU Hours: {fb.gpu_hours}")
            lines.append(f"    Notes: {fb.notes}")
            for br in fb.benchmark_results:
                delta_str = f" ({br.delta:+.1f})" if br.delta is not None else ""
                lines.append(f"    {br.benchmark_name}: {br.score:.1f}{delta_str}")
        lines.append("")

    return "\n".join(lines)


def _build_dataset_context(datasets: list) -> str:
    """Build context about available datasets."""
    if not datasets:
        return "No datasets registered."

    lines = ["Available Datasets:", ""]
    for ds in datasets:
        lines.append(f"- {ds.id}: {ds.name}")
        lines.append(f"  Type: {ds.type}, Examples: {ds.num_examples}")
        lines.append(f"  Quality: {ds.quality_score}, Novelty: {ds.novelty_score}, Overlap: {ds.overlap_pct}%")
        lines.append(f"  Topics: {', '.join(ds.topics) if ds.topics else 'N/A'}")
        lines.append(f"  Synthetic: {ds.is_synthetic}")
        lines.append("")
    return "\n".join(lines)


class ResearchAgent:
    """The Research Agent uses a local LLM to generate hypotheses."""

    def __init__(self, config: Config, db: ExperimentDB):
        self.config = config
        self.db = db
        self.model = config.research_agent.model
        self.provider = config.research_agent.provider
        self.temperature = config.research_agent.temperature
        self.ollama_url = "http://localhost:11434/api/chat"

    def analyze_history(self, limit: int = 10) -> list[Experiment]:
        """Retrieve recent completed experiments for analysis."""
        return self.db.list_experiments(status=ExperimentStatus.COMPLETED, limit=limit)

    def _call_llm(self, prompt: str) -> str:
        """Call Ollama and return raw response text."""
        try:
            resp = httpx.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                    },
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]
        except httpx.RequestError as e:
            return f'{{"hypotheses": [], "error": "LLM call failed: {e}"}}'
        except (KeyError, json.JSONDecodeError) as e:
            return f'{{"hypotheses": [], "error": "Parse error: {e}"}}'

    def _parse_hypotheses(self, raw: str) -> list[Hypothesis]:
        """Parse LLM output into Hypothesis objects. Tolerant of JSON in markdown."""
        # Strip markdown code fences if present
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []

        raw_hyps = data.get("hypotheses", [])
        if not raw_hyps:
            return []

        hyps = []
        for rh in raw_hyps:
            hyps.append(Hypothesis(
                id=f"hyp_{uuid.uuid4().hex[:12]}",
                description=rh.get("description", "No description"),
                target_region=rh.get("target_region", {}),
                confidence=float(rh.get("confidence", 0.5)),
                reasoning=rh.get("reasoning", ""),
                is_exploration=bool(rh.get("is_exploration", False)),
            ))
        return hyps

    def propose_hypotheses(self, recent_experiments: Optional[list[Experiment]] = None) -> list[Hypothesis]:
        """Generate research hypotheses based on experiment history.

        Calls the local Ollama LLM with experiment context and parses
        the structured response into Hypothesis objects.
        """
        if recent_experiments is None:
            recent_experiments = self.analyze_history()

        exp_context = _build_experiment_context(recent_experiments)
        dataset_context = _build_dataset_context(self.db.list_datasets())

        prompt = f"""Analyze the following experiment history and propose hypotheses for the next round of experimentation.

{exp_context}

{dataset_context}

Remember to return ONLY valid JSON in this format:
{{"hypotheses": [{{"description": "...", "target_region": {{...}}, "confidence": 0.0, "reasoning": "...", "is_exploration": false}}]}}
"""

        raw = self._call_llm(prompt)
        hyps = self._parse_hypotheses(raw)

        return hyps

    def recommend_dataset_changes(self) -> list[dict]:
        """Recommend changes to the dataset composition.

        Returns a list of recommendation dicts with 'action', 'dataset_id',
        and 'reasoning'.
        """
        datasets = self.db.list_datasets()
        stats = {
            "total": len(datasets),
            "synthetic": sum(1 for d in datasets if d.is_synthetic),
        }
        prompt = f"""Based on the current dataset registry state, recommend any changes.
Current state: {json.dumps(stats)}

Available datasets:
{_build_dataset_context(datasets)}

Return JSON: {{"recommendations": [{{"action": "add/remove/adjust", "dataset_id": "...", "reasoning": "..."}}]}}
"""
        raw = self._call_llm(prompt)
        try:
            data = json.loads(raw)
            return data.get("recommendations", [])
        except (json.JSONDecodeError, KeyError):
            return []

    def explain_reasoning(self, hypothesis_id: str) -> str:
        """Return the reasoning trace for a given hypothesis."""
        hyp = self.db.get_hypothesis(hypothesis_id)
        if hyp:
            return hyp.reasoning
        return "Hypothesis not found."
