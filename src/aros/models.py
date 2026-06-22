"""Core data models for AROS."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class ExperimentStatus(str, Enum):
    """Lifecycle states for experiments."""
    PROPOSED = "proposed"
    APPROVED = "approved"
    QUEUED = "queued"
    RUNNING = "running"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"

    @classmethod
    def valid_transitions(cls) -> dict:
        """Returns valid state transitions."""
        return {
            cls.PROPOSED: {cls.APPROVED, cls.ARCHIVED},
            cls.APPROVED: {cls.QUEUED, cls.ARCHIVED},
            cls.QUEUED: {cls.RUNNING, cls.FAILED},
            cls.RUNNING: {cls.EVALUATING, cls.FAILED},
            cls.EVALUATING: {cls.COMPLETED, cls.FAILED},
            cls.COMPLETED: {cls.ARCHIVED},
            cls.FAILED: {cls.ARCHIVED},
            cls.ARCHIVED: set(),
        }


@dataclass
class Hypothesis:
    """A research hypothesis proposing a strategic direction."""
    id: str
    description: str
    target_region: dict  # e.g. {"planning_percentage": "20-40%"}
    confidence: float
    reasoning: str
    is_exploration: bool = False
    budget_spent: int = 0
    budget_total: int = 10
    best_score: Optional[float] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "target_region": self.target_region,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "is_exploration": self.is_exploration,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class SearchSpaceParameter:
    """A single parameter in the search space."""
    name: str
    type: str  # "float", "int", "categorical"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    categories: Optional[list] = None
    default: Optional[float] = None


@dataclass
class SearchSpace:
    """Formal search space definition."""
    parameters: list[SearchSpaceParameter] = field(default_factory=list)

    def add_parameter(self, param: SearchSpaceParameter):
        self.parameters.append(param)
        return self


@dataclass
class ExperimentConfig:
    """Full configuration for an experiment."""
    dataset_ids: list[str] = field(default_factory=list)
    dataset_weights: dict[str, float] = field(default_factory=dict)
    learning_rate: float = 5e-5
    batch_size: int = 16
    epochs: int = 3
    lora_rank: int = 16
    lora_alpha: int = 32
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    curriculum_schedule: Optional[dict] = None
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "dataset_ids": self.dataset_ids,
            "dataset_weights": self.dataset_weights,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "curriculum_schedule": self.curriculum_schedule,
            "extras": self.extras,
        }

    def fingerprint(self) -> str:
        """Deterministic hash of config values for deduplication."""
        import hashlib
        raw = str(sorted(self.to_dict().items()))
        return hashlib.md5(raw.encode()).hexdigest()


@dataclass
class BenchmarkResult:
    """Result from a single benchmark evaluation."""
    benchmark_name: str
    score: float
    delta: Optional[float] = None  # change from baseline

    def to_dict(self) -> dict:
        return {
            "benchmark_name": self.benchmark_name,
            "score": self.score,
            "delta": self.delta,
        }


@dataclass
class ExperimentFeedback:
    """Rich feedback provided to the Research Agent after evaluation."""
    experiment_id: str
    training_loss: float
    validation_loss: float
    benchmark_results: list[BenchmarkResult] = field(default_factory=list)
    forgetting_index: float = 0.0
    gpu_hours: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "training_loss": self.training_loss,
            "validation_loss": self.validation_loss,
            "benchmark_results": [b.to_dict() for b in self.benchmark_results],
            "forgetting_index": self.forgetting_index,
            "gpu_hours": self.gpu_hours,
            "notes": self.notes,
        }


@dataclass
class Experiment:
    """A complete experiment record."""
    id: str
    status: ExperimentStatus = ExperimentStatus.PROPOSED
    hypothesis_id: Optional[str] = None
    config: Optional[ExperimentConfig] = None
    results: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    resource_usage: dict = field(default_factory=dict)
    feedback: Optional[ExperimentFeedback] = None
    config_fingerprint: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    human_comment: Optional[str] = None

    def transition(self, new_status: ExperimentStatus) -> bool:
        """Attempt to transition to a new status. Returns True if valid."""
        valid = ExperimentStatus.valid_transitions()
        if new_status in valid.get(self.status, set()):
            self.status = new_status
            self.updated_at = datetime.now(timezone.utc)
            if new_status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED):
                self.completed_at = datetime.now(timezone.utc)
            return True
        return False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status.value,
            "hypothesis_id": self.hypothesis_id,
            "config": self.config.to_dict() if self.config else None,
            "results": self.results,
            "metrics": self.metrics,
            "resource_usage": self.resource_usage,
            "feedback": self.feedback.to_dict() if self.feedback else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "human_comment": self.human_comment,
        }


@dataclass
class DatasetInfo:
    """Metadata-only representation of a dataset."""
    id: str
    name: str
    type: str  # "coding", "planning", "general_qa", "agent_traces", "synthetic"
    num_examples: int
    quality_score: float  # 0-10
    novelty_score: float  # 0-10
    overlap_pct: float   # 0-100
    topics: list[str] = field(default_factory=list)
    avg_length: int = 0  # tokens
    source: str = "unknown"
    is_synthetic: bool = False
    path: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "num_examples": self.num_examples,
            "quality_score": self.quality_score,
            "novelty_score": self.novelty_score,
            "overlap_pct": self.overlap_pct,
            "topics": self.topics,
            "avg_length": self.avg_length,
            "source": self.source,
            "is_synthetic": self.is_synthetic,
        }
