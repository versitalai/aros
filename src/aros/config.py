"""Configuration loader for AROS."""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DatabaseConfig:
    path: str = "data/experiments.db"
    echo: bool = False


@dataclass
class ResearchAgentConfig:
    model: str = "gemma-4-12b"
    provider: str = "local"
    temperature: float = 0.7
    max_history_experiments: int = 50
    exploration_threshold: float = 0.2


@dataclass
class SearchEngineConfig:
    algorithm: str = "bayesian_optimization"
    experiments_per_hypothesis: int = 5
    max_concurrent: int = 3
    default_budget: int = 10


@dataclass
class ExperimentRunnerConfig:
    max_retries: int = 2
    timeout_hours: int = 24
    checkpoint_frequency: int = 100
    default_epochs: int = 3
    default_lora_rank: int = 16
    default_lora_alpha: int = 32


@dataclass
class EvaluatorConfig:
    visible_benchmarks: list = field(default_factory=lambda: ["coding", "planning", "agent_tasks"])
    hidden_benchmarks: list = field(default_factory=lambda: ["generalization", "robustness"])
    evaluation_batch_size: int = 32


@dataclass
class DatasetRegistryConfig:
    max_datasets: int = 100
    min_quality_score: float = 5.0
    synthetic_cap: float = 0.5


@dataclass
class LoopConfig:
    sleep_seconds: int = 60
    max_iterations: int = 0  # 0 = unlimited
    exploitation_ratio: float = 0.8
    exploration_ratio: float = 0.2


@dataclass
class Config:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    research_agent: ResearchAgentConfig = field(default_factory=ResearchAgentConfig)
    search_engine: SearchEngineConfig = field(default_factory=SearchEngineConfig)
    experiment_runner: ExperimentRunnerConfig = field(default_factory=ExperimentRunnerConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    dataset_registry: DatasetRegistryConfig = field(default_factory=DatasetRegistryConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)


def _env_override(key: str, default):
    """Check for AROS_<section>_<key> env var override."""
    env_key = f"AROS_{key.upper().replace('.', '_')}"
    val = os.environ.get(env_key)
    if val is None:
        return default

    # Try to coerce to the same type as default
    if isinstance(default, bool):
        return val.lower() in ("true", "1", "yes")
    if isinstance(default, int):
        return int(val)
    if isinstance(default, float):
        return float(val)
    return val


def _dict_to_dataclass(cls, data: dict, prefix: str = ""):
    """Recursively convert a nested dict to a dataclass instance."""
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for key, value in data.items():
        env_key = f"{prefix}.{key}" if prefix else key
        if key in field_types:
            ftype = field_types[key]
            # Check if the field type is itself a dataclass
            if hasattr(ftype, "__dataclass_fields__") and isinstance(value, dict):
                kwargs[key] = _dict_to_dataclass(ftype, value, prefix=env_key)
            else:
                kwargs[key] = _env_override(env_key, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: Optional[str] = None) -> Config:
    """Load configuration from YAML file with environment variable overrides.

    Environment variables follow the pattern: AROS_<SECTION>_<KEY>
    Example: AROS_DATABASE_PATH=/custom/path.db
    """
    if path is None:
        # Look in default locations
        search_paths = [
            Path("config/default.yaml"),
            Path.home() / ".aros" / "config.yaml",
            Path.cwd() / "config" / "default.yaml",
            Path("/etc/aros/config.yaml"),
        ]
        for sp in search_paths:
            if sp.exists():
                path = str(sp)
                break

    if path and Path(path).exists():
        with open(path) as f:
            raw = yaml.safe_load(f)
    else:
        raw = {}

    return _dict_to_dataclass(Config, raw)
