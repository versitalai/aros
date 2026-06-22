"""Benchmark Registry — manages visible and hidden benchmarks.

Benchmarks are registered independently with a standard interface:
    evaluate(model) -> score

The registry maintains the visible/hidden split to prevent benchmark
overfitting (Goodhart's Law defense).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

from .models import BenchmarkResult


@dataclass
class BenchmarkDefinition:
    """Definition of a single benchmark."""
    name: str
    description: str
    is_hidden: bool = False
    max_score: float = 100.0
    min_score: float = 0.0
    metadata: dict = field(default_factory=dict)


class BaseBenchmark(ABC):
    """Abstract base for all benchmarks."""

    @abstractmethod
    def evaluate(self, model_path: str) -> float:
        """Run evaluation on a model checkpoint.

        Args:
            model_path: Path to the model checkpoint or identifier.

        Returns:
            Numerical score.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Return the benchmark name."""
        ...


class BenchmarkRegistry:
    """Registry of all benchmarks with visible/hidden tracking."""

    def __init__(self):
        self._benchmarks: dict[str, BenchmarkDefinition] = {}
        self._evaluators: dict[str, BaseBenchmark] = {}

    def register(self, definition: BenchmarkDefinition, evaluator: Optional[BaseBenchmark] = None):
        """Register a benchmark with an optional evaluator."""
        self._benchmarks[definition.name] = definition
        if evaluator:
            self._evaluators[definition.name] = evaluator

    def register_evaluator(self, benchmark_name: str, evaluator: BaseBenchmark):
        """Register or replace an evaluator for an existing benchmark."""
        if benchmark_name not in self._benchmarks:
            raise KeyError(f"Benchmark '{benchmark_name}' not registered. Call register() first.")
        self._evaluators[benchmark_name] = evaluator

    def get_visible(self) -> list[BenchmarkDefinition]:
        """Get all visible benchmarks."""
        return [b for b in self._benchmarks.values() if not b.is_hidden]

    def get_hidden(self) -> list[BenchmarkDefinition]:
        """Get all hidden benchmarks."""
        return [b for b in self._benchmarks.values() if b.is_hidden]

    def get_all(self) -> list[BenchmarkDefinition]:
        """Get all benchmarks."""
        return list(self._benchmarks.values())

    def evaluate_all_visible(self, model_path: str) -> list[BenchmarkResult]:
        """Evaluate all visible benchmarks and return results."""
        results = []
        for bm in self.get_visible():
            evaluator = self._evaluators.get(bm.name)
            if evaluator is None:
                continue
            score = evaluator.evaluate(model_path)
            results.append(BenchmarkResult(benchmark_name=bm.name, score=score))
        return results

    def evaluate_all_hidden(self, model_path: str) -> list[BenchmarkResult]:
        """Evaluate all hidden benchmarks and return results."""
        results = []
        for bm in self.get_hidden():
            evaluator = self._evaluators.get(bm.name)
            if evaluator is None:
                continue
            score = evaluator.evaluate(model_path)
            results.append(BenchmarkResult(benchmark_name=bm.name, score=score, delta=None))
        return results

    def evaluate_all(self, model_path: str) -> tuple[list[BenchmarkResult], list[BenchmarkResult]]:
        """Evaluate all benchmarks. Returns (visible_results, hidden_results).

        Hidden results should be stored but never shown to the Research Agent.
        """
        visible = self.evaluate_all_visible(model_path)
        hidden = self.evaluate_all_hidden(model_path)
        return visible, hidden

    def get_definition(self, name: str) -> Optional[BenchmarkDefinition]:
        """Get a benchmark definition by name."""
        return self._benchmarks.get(name)


# --- Built-in benchmark stubs ---

class DummyBenchmark(BaseBenchmark):
    """A dummy benchmark for testing. Always returns a fixed score."""

    def __init__(self, name: str, fixed_score: float = 75.0):
        self._name = name
        self._fixed_score = fixed_score

    def evaluate(self, model_path: str) -> float:
        return self._fixed_score

    def name(self) -> str:
        return self._name


def create_default_registry() -> BenchmarkRegistry:
    """Create a BenchmarkRegistry with default benchmarks matching the AROS plan."""
    registry = BenchmarkRegistry()

    # Visible benchmarks
    registry.register(BenchmarkDefinition(
        name="coding",
        description="Code generation and completion (HumanEval, MBPP, Repo tasks)",
        is_hidden=False,
    ))
    registry.register(BenchmarkDefinition(
        name="planning",
        description="Long-horizon reasoning and task decomposition",
        is_hidden=False,
    ))
    registry.register(BenchmarkDefinition(
        name="agent_tasks",
        description="Agentic coding: multi-file modifications, tool use, repo navigation",
        is_hidden=False,
    ))

    # Hidden benchmarks (never shown to Research Agent)
    registry.register(BenchmarkDefinition(
        name="generalization",
        description="Generalization holdout — measures true out-of-distribution performance",
        is_hidden=True,
    ))
    registry.register(BenchmarkDefinition(
        name="robustness",
        description="Robustness to distribution shift and adversarial inputs",
        is_hidden=True,
    ))

    # Register dummy evaluators for testing
    for bm in registry.get_all():
        registry.register_evaluator(bm.name, DummyBenchmark(bm.name))

    return registry
