"""Smoke tests for AROS.

Verifies that all core components import, instantiate, and perform
basic operations correctly. Run with:

    cd /home/q/aros && python -m pytest tests/ -v
"""

import json
import tempfile
from datetime import datetime

import pytest


# =========================================================================
# Phase 1: Config
# =========================================================================

class TestConfig:
    def test_load_default_config(self):
        """Config loads with defaults when no file provided."""
        from aros.config import load_config, Config
        cfg = load_config()
        assert isinstance(cfg, Config)
        assert cfg.database.path == "data/experiments.db"
        assert cfg.research_agent.model == "llama3.2:3b"
        assert cfg.search_engine.algorithm == "bayesian_optimization"
        assert cfg.loop.exploitation_ratio == 0.8
        assert cfg.loop.exploration_ratio == 0.2

    def test_env_override(self, monkeypatch):
        """Environment variables override config values."""
        monkeypatch.setenv("AROS_DATABASE_PATH", "/tmp/test_aros.db")
        monkeypatch.setenv("AROS_LOOP_MAX_ITERATIONS", "42")
        from aros.config import load_config
        cfg = load_config()
        assert cfg.database.path == "/tmp/test_aros.db"
        assert cfg.loop.max_iterations == 42

    def test_yaml_loading(self, tmp_path):
        """Config loads from YAML file."""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text("""
database:
  path: "/custom/path.db"
  echo: true
research_agent:
  model: "qwen3-8b"
  temperature: 0.5
""")
        from aros.config import load_config
        cfg = load_config(str(yaml_file))
        assert cfg.database.path == "/custom/path.db"
        assert cfg.database.echo == True
        assert cfg.research_agent.model == "qwen3-8b"
        assert cfg.research_agent.temperature == 0.5


# =========================================================================
# Phase 1: Models
# =========================================================================

class TestModels:
    def test_experiment_status_enum(self):
        """ExperimentStatus has correct values and transitions."""
        from aros.models import ExperimentStatus
        assert ExperimentStatus.PROPOSED.value == "proposed"
        assert ExperimentStatus.COMPLETED.value == "completed"

    def test_valid_status_transition(self):
        """Valid state transition works."""
        from aros.models import Experiment, ExperimentStatus
        exp = Experiment(id="test1")
        assert exp.transition(ExperimentStatus.APPROVED) == True
        assert exp.status == ExperimentStatus.APPROVED

    def test_invalid_status_transition(self):
        """Invalid state transition is rejected."""
        from aros.models import Experiment, ExperimentStatus
        exp = Experiment(id="test2")
        exp.status = ExperimentStatus.COMPLETED
        assert exp.transition(ExperimentStatus.RUNNING) == False
        assert exp.status == ExperimentStatus.COMPLETED

    def test_hypothesis_creation(self):
        """Hypothesis can be created with all fields."""
        from aros.models import Hypothesis
        hyp = Hypothesis(
            id="hyp_001",
            description="Increasing planning data may improve long-horizon reasoning.",
            target_region={"planning_percentage": "20-40%"},
            confidence=0.78,
            reasoning="Recent experiments show planning gains with minimal coding regression.",
        )
        assert hyp.id == "hyp_001"
        assert hyp.confidence == 0.78
        assert hyp.target_region["planning_percentage"] == "20-40%"

    def test_experiment_config(self):
        """ExperimentConfig stores all training parameters."""
        from aros.models import ExperimentConfig
        cfg = ExperimentConfig(
            dataset_ids=["ds_001", "ds_002"],
            learning_rate=2e-4,
            lora_rank=32,
        )
        assert cfg.dataset_ids == ["ds_001", "ds_002"]
        assert cfg.learning_rate == 2e-4
        assert cfg.lora_rank == 32

    def test_dataset_info(self):
        """DatasetInfo stores metadata correctly."""
        from aros.models import DatasetInfo
        ds = DatasetInfo(
            id="ds_001",
            name="CodeAlpaca",
            type="coding",
            num_examples=25000,
            quality_score=8.7,
            novelty_score=7.2,
            overlap_pct=18.0,
            topics=["Python", "Algorithms", "Debugging"],
            avg_length=1200,
            source="Synthetic Agent Data",
        )
        assert ds.quality_score == 8.7
        assert "Python" in ds.topics
        assert not ds.is_synthetic

    def test_experiment_feedback(self):
        """ExperimentFeedback stores evaluation results."""
        from aros.models import ExperimentFeedback, BenchmarkResult
        feedback = ExperimentFeedback(
            experiment_id="exp_001",
            training_loss=0.91,
            validation_loss=1.07,
            benchmark_results=[
                BenchmarkResult(benchmark_name="coding", score=74.8, delta=2.7),
                BenchmarkResult(benchmark_name="planning", score=71.1, delta=-1.3),
            ],
            forgetting_index=0.5,
            gpu_hours=2.1,
        )
        assert len(feedback.benchmark_results) == 2
        assert feedback.benchmark_results[0].score == 74.8
        assert feedback.benchmark_results[1].delta == -1.3


# =========================================================================
# Phase 2: Database
# =========================================================================

class TestDatabase:
    def test_create_tables(self, tmp_path):
        """Database creates tables on init."""
        from aros.config import Config
        from aros.database import ExperimentDB
        cfg = Config()
        cfg.database.path = str(tmp_path / "test.db")
        db = ExperimentDB(cfg)
        # Should not raise
        assert db.db_path.endswith("test.db")

    def test_create_and_retrieve_experiment(self, tmp_path):
        """Can create an experiment and read it back."""
        from aros.config import Config
        from aros.database import ExperimentDB
        from aros.models import Experiment, ExperimentConfig, ExperimentStatus

        cfg = Config()
        cfg.database.path = str(tmp_path / "test.db")
        db = ExperimentDB(cfg)

        exp = Experiment(
            id="exp_test_001",
            config=ExperimentConfig(learning_rate=1e-4, lora_rank=8),
        )
        db.create_experiment(exp)

        retrieved = db.get_experiment("exp_test_001")
        assert retrieved is not None
        assert retrieved.id == "exp_test_001"
        assert retrieved.status == ExperimentStatus.PROPOSED
        assert retrieved.config.learning_rate == 1e-4
        assert retrieved.config.lora_rank == 8

    def test_experiment_lifecycle(self, tmp_path):
        """Full experiment lifecycle: proposed → approved → queued → running → evaluating → completed."""
        from aros.config import Config
        from aros.database import ExperimentDB
        from aros.models import Experiment, ExperimentStatus

        cfg = Config()
        cfg.database.path = str(tmp_path / "test.db")
        db = ExperimentDB(cfg)

        exp = Experiment(id="exp_lifecycle_001")
        db.create_experiment(exp)

        transitions = [
            ExperimentStatus.APPROVED,
            ExperimentStatus.QUEUED,
            ExperimentStatus.RUNNING,
            ExperimentStatus.EVALUATING,
            ExperimentStatus.COMPLETED,
        ]
        for status in transitions:
            assert db.transition_experiment("exp_lifecycle_001", status), f"Failed transition to {status}"

        final = db.get_experiment("exp_lifecycle_001")
        assert final.status == ExperimentStatus.COMPLETED
        assert final.completed_at is not None

    def test_list_experiments_filtered(self, tmp_path):
        """Can list experiments filtered by status."""
        from aros.config import Config
        from aros.database import ExperimentDB
        from aros.models import Experiment, ExperimentStatus

        cfg = Config()
        cfg.database.path = str(tmp_path / "test.db")
        db = ExperimentDB(cfg)

        db.create_experiment(Experiment(id="e1"))
        e2 = Experiment(id="e2")
        e2.transition(ExperimentStatus.APPROVED)
        db.create_experiment(e2)

        proposed = db.list_experiments(status=ExperimentStatus.PROPOSED)
        assert len(proposed) == 1
        assert proposed[0].id == "e1"

    def test_hypothesis_crud(self, tmp_path):
        """Can save and retrieve hypotheses."""
        from aros.config import Config
        from aros.database import ExperimentDB
        from aros.models import Hypothesis

        cfg = Config()
        cfg.database.path = str(tmp_path / "test.db")
        db = ExperimentDB(cfg)

        hyp = Hypothesis(
            id="hyp_test_001",
            description="Test hypothesis",
            target_region={"lr": "1e-4 to 5e-4"},
            confidence=0.85,
            reasoning="Testing CRUD operations.",
        )
        db.save_hypothesis(hyp)

        retrieved = db.get_hypothesis("hyp_test_001")
        assert retrieved is not None
        assert retrieved.description == "Test hypothesis"
        assert retrieved.confidence == 0.85

    def test_dataset_registration(self, tmp_path):
        """Can register and query datasets."""
        from aros.config import Config
        from aros.database import ExperimentDB
        from aros.models import DatasetInfo

        cfg = Config()
        cfg.database.path = str(tmp_path / "test.db")
        db = ExperimentDB(cfg)

        ds = DatasetInfo(
            id="ds_test_001",
            name="Test Dataset",
            type="coding",
            num_examples=1000,
            quality_score=8.0,
            novelty_score=6.0,
            overlap_pct=10.0,
        )
        db.register_dataset(ds)

        datasets = db.list_datasets()
        assert len(datasets) == 1
        assert datasets[0].quality_score == 8.0

    def test_synthetic_flag(self, tmp_path):
        """Synthetic datasets are flagged correctly."""
        from aros.config import Config
        from aros.database import ExperimentDB
        from aros.models import DatasetInfo

        cfg = Config()
        cfg.database.path = str(tmp_path / "test.db")
        db = ExperimentDB(cfg)

        ds = DatasetInfo(
            id="ds_synth_001",
            name="Synth Data",
            type="synthetic",
            num_examples=500,
            quality_score=7.0,
            novelty_score=8.0,
            overlap_pct=5.0,
            is_synthetic=True,
        )
        db.register_dataset(ds)

        datasets = db.list_datasets()
        assert datasets[0].is_synthetic == True

    def test_benchmark_results(self, tmp_path):
        """Can save and retrieve benchmark results."""
        from aros.config import Config
        from aros.database import ExperimentDB
        from aros.models import Experiment, BenchmarkResult

        cfg = Config()
        cfg.database.path = str(tmp_path / "test.db")
        db = ExperimentDB(cfg)

        db.create_experiment(Experiment(id="exp_bench_001"))
        result = BenchmarkResult(benchmark_name="coding", score=85.0, delta=3.5)
        db.save_benchmark_result("exp_bench_001", result)

        results = db.get_benchmark_results("exp_bench_001")
        assert len(results) == 1
        assert results[0].score == 85.0
        assert results[0].delta == 3.5


# =========================================================================
# Phase 3: Component Instantiation
# =========================================================================

class TestComponents:
    def test_research_agent_instantiation(self, tmp_path):
        """Research Agent instantiates with correct model."""
        from aros.config import Config
        from aros.database import ExperimentDB
        from aros.research_agent import ResearchAgent

        cfg = Config()
        cfg.database.path = str(tmp_path / "test.db")
        db = ExperimentDB(cfg)
        agent = ResearchAgent(cfg, db)
        assert agent.model == "gemma-4-12b"
        assert agent.ollama_url == "http://localhost:11434/api/chat"

    def test_search_engine_instantiation(self, tmp_path):
        """Search Engine instantiates with correct algorithm."""
        from aros.config import Config
        from aros.database import ExperimentDB
        from aros.search_engine import SearchEngine
        from aros.models import Hypothesis

        cfg = Config()
        cfg.database.path = str(tmp_path / "test.db")
        db = ExperimentDB(cfg)
        engine = SearchEngine(cfg, db)
        assert engine.algorithm == "bayesian_optimization"
        # Pass a real hypothesis to expand
        hyp = Hypothesis(id="test", description="test", target_region={}, confidence=0.5, reasoning="test")
        configs = engine.expand_hypothesis(hyp)
        assert len(configs) == cfg.search_engine.experiments_per_hypothesis

    def test_evaluator_instantiation(self, tmp_path):
        """Evaluator instantiates with benchmark lists."""
        from aros.config import Config
        from aros.database import ExperimentDB
        from aros.evaluator import Evaluator

        cfg = Config()
        cfg.database.path = str(tmp_path / "test.db")
        db = ExperimentDB(cfg)
        evaluator = Evaluator(cfg, db)
        assert "coding" in evaluator.visible_benchmarks

    def test_dataset_registry_instantiation(self, tmp_path):
        """Dataset Registry instantiates and validates."""
        from aros.config import Config
        from aros.database import ExperimentDB
        from aros.dataset_registry import DatasetRegistry

        cfg = Config()
        cfg.database.path = str(tmp_path / "test.db")
        db = ExperimentDB(cfg)
        registry = DatasetRegistry(cfg, db)
        stats = registry.get_summary_stats()
        assert stats["total"] == 0

    def test_generalization_monitor_instantiation(self, tmp_path):
        """Generalization Monitor instantiates."""
        from aros.config import Config
        from aros.database import ExperimentDB
        from aros.generalization_monitor import GeneralizationMonitor

        cfg = Config()
        cfg.database.path = str(tmp_path / "test.db")
        db = ExperimentDB(cfg)
        monitor = GeneralizationMonitor(cfg, db)
        assert not monitor.should_pause_autonomy()

    def test_full_loop_instantiation(self, tmp_path):
        """AROSLoop instantiates all sub-components."""
        from aros.config import Config
        from aros.loop import AROSLoop

        cfg = Config()
        cfg.database.path = str(tmp_path / "test.db")
        loop = AROSLoop(cfg)
        assert loop.research_agent is not None
        assert loop.search_engine is not None
        assert loop.experiment_runner is not None
        assert loop.evaluator is not None
        assert loop.dataset_registry is not None
        assert loop.generalization_monitor is not None


# =========================================================================
# Phase 8: Loop Execution (One Cycle)
# =========================================================================

class TestLoopExecution:
    def test_one_cycle_runs(self, tmp_path):
        """A single research loop cycle runs without errors."""
        from aros.config import Config
        from aros.loop import AROSLoop

        cfg = Config()
        cfg.database.path = str(tmp_path / "test_cycle.db")
        loop = AROSLoop(cfg)

        # Monkey-patch to avoid real LLM call in tests
        original_propose = loop.research_agent.propose_hypotheses
        loop.research_agent.propose_hypotheses = lambda *a, **kw: []

        result = loop.run_one_cycle()
        # Returns None since hypotheses list is empty
        assert result is None or isinstance(result, str)

        # Restore
        loop.research_agent.propose_hypotheses = original_propose

    def test_loop_does_not_crash(self, tmp_path):
        """Running multiple cycles doesn't crash."""
        from aros.config import Config
        from aros.loop import AROSLoop

        cfg = Config()
        cfg.database.path = str(tmp_path / "test_cycles.db")
        cfg.loop.sleep_seconds = 0  # Don't actually sleep in tests
        loop = AROSLoop(cfg)

        # Monkey-patch to avoid real LLM call in tests
        original_propose = loop.research_agent.propose_hypotheses
        loop.research_agent.propose_hypotheses = lambda *a, **kw: []

        for _ in range(3):
            loop.run_one_cycle()

        experiments = loop.db.list_experiments()
        assert isinstance(experiments, list)

        loop.research_agent.propose_hypotheses = original_propose
