"""SQLite experiment database for AROS."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from .models import (
    Experiment,
    ExperimentConfig,
    ExperimentFeedback,
    ExperimentStatus,
    BenchmarkResult,
    Hypothesis,
    DatasetInfo,
)
from .config import Config


class ExperimentDB:
    """SQLite-backed experiment database."""

    def __init__(self, config: Config):
        db_path = Path(config.database.path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_path)
        self.echo = config.database.echo
        self.create_tables()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _log(self, msg: str):
        if self.echo:
            print(f"[DB] {msg}")

    def create_tables(self):
        """Create all required tables if they don't exist."""
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'proposed',
                    hypothesis_id TEXT,
                    config_json TEXT,
                    results_json TEXT DEFAULT '{}',
                    metrics_json TEXT DEFAULT '{}',
                    resource_usage_json TEXT DEFAULT '{}',
                    feedback_json TEXT,
                    config_fingerprint TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    human_comment TEXT
                );

                CREATE TABLE IF NOT EXISTS hypotheses (
                    id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    target_region_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reasoning TEXT NOT NULL,
                    is_exploration INTEGER DEFAULT 0,
                    budget_spent INTEGER DEFAULT 0,
                    budget_total INTEGER DEFAULT 10,
                    best_score REAL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    num_examples INTEGER NOT NULL,
                    quality_score REAL NOT NULL,
                    novelty_score REAL NOT NULL,
                    overlap_pct REAL NOT NULL,
                    topics_json TEXT DEFAULT '[]',
                    avg_length INTEGER DEFAULT 0,
                    source TEXT DEFAULT 'unknown',
                    is_synthetic INTEGER DEFAULT 0,
                    path TEXT,
                    metadata_json TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS benchmark_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    benchmark_name TEXT NOT NULL,
                    score REAL NOT NULL,
                    delta REAL,
                    is_hidden INTEGER DEFAULT 0,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );

                CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
                CREATE INDEX IF NOT EXISTS idx_experiments_hypothesis ON experiments(hypothesis_id);
                CREATE INDEX IF NOT EXISTS idx_benchmarks_experiment ON benchmark_results(experiment_id);
                CREATE INDEX IF NOT EXISTS idx_experiments_fingerprint ON experiments(config_fingerprint);
            """)
            self._log("Tables created/verified")

    # --- Experiment CRUD ---

    def create_experiment(self, experiment: Experiment) -> str:
        """Insert a new experiment. Returns the ID."""
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO experiments
                   (id, status, hypothesis_id, config_json, results_json,
                    metrics_json, resource_usage_json, feedback_json,
                    created_at, updated_at, completed_at, human_comment, config_fingerprint)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    experiment.id,
                    experiment.status.value,
                    experiment.hypothesis_id,
                    json.dumps(experiment.config.to_dict()) if experiment.config else None,
                    json.dumps(experiment.results),
                    json.dumps(experiment.metrics),
                    json.dumps(experiment.resource_usage),
                    json.dumps(experiment.feedback.to_dict()) if experiment.feedback else None,
                    experiment.created_at.isoformat(),
                    experiment.updated_at.isoformat(),
                    experiment.completed_at.isoformat() if experiment.completed_at else None,
                    experiment.human_comment,
                    experiment.config.fingerprint() if experiment.config else None,
                ),
            )
            self._log(f"Created experiment: {experiment.id}")
            return experiment.id

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Retrieve an experiment by ID."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_experiment(row)

    def update_experiment(self, experiment: Experiment) -> bool:
        """Update an existing experiment. Returns True if updated."""
        with self._conn() as conn:
            cursor = conn.execute(
                """UPDATE experiments SET
                   status = ?, config_json = ?, results_json = ?,
                   metrics_json = ?, resource_usage_json = ?,
                   feedback_json = ?, updated_at = ?,
                   completed_at = ?, human_comment = ?, config_fingerprint = ?
                   WHERE id = ?""",
                (
                    experiment.status.value,
                    json.dumps(experiment.config.to_dict()) if experiment.config else None,
                    json.dumps(experiment.results),
                    json.dumps(experiment.metrics),
                    json.dumps(experiment.resource_usage),
                    json.dumps(experiment.feedback.to_dict()) if experiment.feedback else None,
                    experiment.updated_at.isoformat(),
                    experiment.completed_at.isoformat() if experiment.completed_at else None,
                    experiment.human_comment,
                    experiment.config.fingerprint() if experiment.config else None,
                    experiment.id,
                ),
            )
            updated = cursor.rowcount > 0
            if updated:
                self._log(f"Updated experiment: {experiment.id}")
            return updated

    def transition_experiment(self, experiment_id: str, new_status: ExperimentStatus) -> bool:
        """Transition an experiment's status. Returns True if valid."""
        exp = self.get_experiment(experiment_id)
        if exp is None:
            return False
        if exp.transition(new_status):
            return self.update_experiment(exp)
        return False

    def list_experiments(self, status: Optional[ExperimentStatus] = None, limit: int = 100) -> list[Experiment]:
        """List experiments, optionally filtered by status."""
        with self._conn() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM experiments WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status.value, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [self._row_to_experiment(r) for r in rows]

    def _row_to_experiment(self, row: sqlite3.Row) -> Experiment:
        """Convert a SQLite row to an Experiment object."""
        exp = Experiment(
            id=row["id"],
            status=ExperimentStatus(row["status"]),
            hypothesis_id=row["hypothesis_id"],
            config=self._parse_config_json(row["config_json"]),
            results=json.loads(row["results_json"]) if row["results_json"] else {},
            metrics=json.loads(row["metrics_json"]) if row["metrics_json"] else {},
            resource_usage=json.loads(row["resource_usage_json"]) if row["resource_usage_json"] else {},
            feedback=self._parse_feedback_json(row["feedback_json"]),
            config_fingerprint=row["config_fingerprint"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            human_comment=row["human_comment"],
        )
        return exp

    def _parse_config_json(self, raw: Optional[str]) -> Optional[ExperimentConfig]:
        if not raw:
            return None
        data = json.loads(raw)
        return ExperimentConfig(
            dataset_ids=data.get("dataset_ids", []),
            dataset_weights=data.get("dataset_weights", {}),
            learning_rate=data.get("learning_rate", 5e-5),
            batch_size=data.get("batch_size", 16),
            epochs=data.get("epochs", 3),
            lora_rank=data.get("lora_rank", 16),
            lora_alpha=data.get("lora_alpha", 32),
            weight_decay=data.get("weight_decay", 0.01),
            warmup_ratio=data.get("warmup_ratio", 0.1),
            extras=data.get("extras", {}),
        )

    def _parse_feedback_json(self, raw: Optional[str]) -> Optional[ExperimentFeedback]:
        if not raw:
            return None
        data = json.loads(raw)
        benchmarks = [BenchmarkResult(**b) for b in data.get("benchmark_results", [])]
        return ExperimentFeedback(
            experiment_id=data["experiment_id"],
            training_loss=data["training_loss"],
            validation_loss=data["validation_loss"],
            benchmark_results=benchmarks,
            forgetting_index=data.get("forgetting_index", 0.0),
            gpu_hours=data.get("gpu_hours", 0.0),
            notes=data.get("notes", ""),
        )

    # --- Hypothesis CRUD ---

    def save_hypothesis(self, hypothesis: Hypothesis) -> str:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO hypotheses
                   (id, description, target_region_json, confidence, reasoning, is_exploration,
                    budget_spent, budget_total, best_score, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    hypothesis.id,
                    hypothesis.description,
                    json.dumps(hypothesis.target_region),
                    hypothesis.confidence,
                    hypothesis.reasoning,
                    int(hypothesis.is_exploration),
                    hypothesis.budget_spent,
                    hypothesis.budget_total,
                    hypothesis.best_score,
                    hypothesis.created_at.isoformat(),
                ),
            )
            return hypothesis.id

    def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM hypotheses WHERE id = ?", (hypothesis_id,)
            ).fetchone()
            if row is None:
                return None
            return Hypothesis(
                id=row["id"],
                description=row["description"],
                target_region=json.loads(row["target_region_json"]),
                confidence=row["confidence"],
                reasoning=row["reasoning"],
                is_exploration=bool(row["is_exploration"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )

    def list_hypotheses(self, limit: int = 50) -> list[Hypothesis]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM hypotheses ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [
                Hypothesis(
                    id=r["id"],
                    description=r["description"],
                    target_region=json.loads(r["target_region_json"]),
                    confidence=r["confidence"],
                    reasoning=r["reasoning"],
                    is_exploration=bool(r["is_exploration"]),
                    created_at=datetime.fromisoformat(r["created_at"]),
                )
                for r in rows
            ]

    # --- Dataset CRUD ---

    def register_dataset(self, dataset: DatasetInfo) -> str:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO datasets
                   (id, name, type, num_examples, quality_score, novelty_score,
                    overlap_pct, topics_json, avg_length, source, is_synthetic, metadata_json, path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    dataset.id,
                    dataset.name,
                    dataset.type,
                    dataset.num_examples,
                    dataset.quality_score,
                    dataset.novelty_score,
                    dataset.overlap_pct,
                    json.dumps(dataset.topics),
                    dataset.avg_length,
                    dataset.source,
                    int(dataset.is_synthetic),
                    json.dumps(dataset.metadata),
                    dataset.path,
                ),
            )
            return dataset.id

    def list_datasets(self, dataset_type: Optional[str] = None) -> list[DatasetInfo]:
        with self._conn() as conn:
            if dataset_type:
                rows = conn.execute(
                    "SELECT * FROM datasets WHERE type = ? ORDER BY quality_score DESC",
                    (dataset_type,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM datasets ORDER BY quality_score DESC"
                ).fetchall()
            return [self._row_to_dataset(r) for r in rows]

    def _row_to_dataset(self, row: sqlite3.Row) -> DatasetInfo:
        return DatasetInfo(
            id=row["id"],
            name=row["name"],
            type=row["type"],
            num_examples=row["num_examples"],
            quality_score=row["quality_score"],
            novelty_score=row["novelty_score"],
            overlap_pct=row["overlap_pct"],
            topics=json.loads(row["topics_json"]) if row["topics_json"] else [],
            avg_length=row["avg_length"],
            source=row["source"],
            is_synthetic=bool(row["is_synthetic"]),
            path=row["path"],
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
        )

    # --- Benchmark Results ---

    def save_benchmark_result(self, experiment_id: str, result: BenchmarkResult, is_hidden: bool = False):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO benchmark_results
                   (experiment_id, benchmark_name, score, delta, is_hidden)
                   VALUES (?, ?, ?, ?, ?)""",
                (experiment_id, result.benchmark_name, result.score, result.delta, int(is_hidden)),
            )

    def get_benchmark_results(self, experiment_id: str) -> list[BenchmarkResult]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM benchmark_results WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchall()
            return [
                BenchmarkResult(
                    benchmark_name=r["benchmark_name"],
                    score=r["score"],
                    delta=r["delta"],
                )
                for r in rows
            ]

    def get_experiments_by_fingerprint(self, fingerprint: str) -> list[Experiment]:
        """Find experiments with a matching config fingerprint (for dedup)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM experiments WHERE config_fingerprint = ?",
                (fingerprint,),
            ).fetchall()
            return [self._row_to_experiment(r) for r in rows]
