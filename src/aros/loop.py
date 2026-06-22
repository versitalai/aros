"""Main research loop — orchestrates the full AROS pipeline.

The loop follows a fixed sequence:
    Observe → Hypothesize → Search → Execute → Evaluate → Store → Learn → Repeat
"""

import time
import logging
from typing import Optional

from .config import Config, load_config
from .database import ExperimentDB
from .research_agent import ResearchAgent
from .search_engine import SearchEngine
from .experiment_runner import ExperimentRunner
from .evaluator import Evaluator
from .dataset_registry import DatasetRegistry
from .generalization_monitor import GeneralizationMonitor
from .models import Experiment, ExperimentConfig, ExperimentStatus, DatasetInfo

logger = logging.getLogger("aros")


class AROSLoop:
    """Main orchestrator for the AROS research loop."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.db = ExperimentDB(self.config)
        self.research_agent = ResearchAgent(self.config, self.db)
        self.search_engine = SearchEngine(self.config, self.db)
        self.experiment_runner = ExperimentRunner(self.config, self.db)
        self.evaluator = Evaluator(self.config, self.db)
        self.dataset_registry = DatasetRegistry(self.config, self.db)
        self.generalization_monitor = GeneralizationMonitor(self.config, self.db)
        self._iteration = 0

        # Seed some default datasets if registry is empty
        self._seed_datasets()

    def _seed_datasets(self):
        """Register default datasets if the registry is empty."""
        datasets = self.db.list_datasets()
        if len(datasets) > 0:
            return

        defaults = [
            DatasetInfo(
                id="ds_coding_001", name="CodeAlpaca", type="coding",
                num_examples=25000, quality_score=8.7, novelty_score=7.2, overlap_pct=18.0,
                topics=["Python", "Algorithms", "Debugging"], avg_length=1200,
                source="Synthetic Agent Data", is_synthetic=True,
            ),
            DatasetInfo(
                id="ds_coding_002", name="HumanEval Pack", type="coding",
                num_examples=5000, quality_score=9.2, novelty_score=6.5, overlap_pct=12.0,
                topics=["Python", "Function Completion"], avg_length=800,
                source="OpenAI", is_synthetic=False,
            ),
            DatasetInfo(
                id="ds_planning_001", name="PlanBench", type="planning",
                num_examples=12000, quality_score=8.0, novelty_score=8.5, overlap_pct=8.0,
                topics=["Task Decomposition", "Long-horizon", "Reasoning"], avg_length=2000,
                source="Synthetic", is_synthetic=True,
            ),
            DatasetInfo(
                id="ds_agent_001", name="AgentTraces", type="agent_traces",
                num_examples=8000, quality_score=7.8, novelty_score=9.0, overlap_pct=5.0,
                topics=["Tool Use", "Multi-step", "Navigation"], avg_length=3000,
                source="Synthetic Agent Data", is_synthetic=True,
            ),
            DatasetInfo(
                id="ds_qa_001", name="MMLU Subset", type="general_qa",
                num_examples=15000, quality_score=9.0, novelty_score=5.0, overlap_pct=25.0,
                topics=["STEM", "Humanities", "Social Sciences"], avg_length=500,
                source="Standard Benchmarks", is_synthetic=False,
            ),
        ]
        for ds in defaults:
            self.db.register_dataset(ds)
        logger.info(f"Seeded {len(defaults)} default datasets")

    def run_one_cycle(self) -> Optional[str]:
        """Execute one full research loop cycle.

        Returns the experiment ID if one was completed, None otherwise.
        """
        self._iteration += 1
        logger.info(f"=== AROS Cycle {self._iteration} ===")
        print(f"\n{'='*50}")
        print(f"  AROS Cycle {self._iteration}")
        print(f"{'='*50}")

        # 1. OBSERVE — analyze experiment history
        print("\n[1/7] Observe — analyzing experiment history...")
        recent = self.research_agent.analyze_history(limit=10)
        if recent:
            print(f"  Found {len(recent)} completed experiments")
            for exp in recent[-3:]:
                if exp.feedback:
                    best = max(exp.feedback.benchmark_results, key=lambda r: r.score, default=None)
                    if best:
                        print(f"    {exp.id}: {best.benchmark_name}={best.score:.1f}")
        else:
            print("  No completed experiments yet — this is the first cycle")

        # 2. HYPOTHESIZE — generate research hypotheses via LLM
        print("\n[2/7] Hypothesize — calling Research Agent (llama3.2:3b)...")
        hypotheses = self.research_agent.propose_hypotheses(recent)
        if not hypotheses:
            print("  No hypotheses generated (LLM returned nothing meaningful)")
            return None

        print(f"  Generated {len(hypotheses)} hypotheses:")
        for i, hyp in enumerate(hypotheses, 1):
            exp_type = "🔬 Exploration" if hyp.is_exploration else "🎯 Exploitation"
            print(f"  {i}. [{exp_type}] {hyp.description}")
            print(f"     Confidence: {hyp.confidence:.2f} | Target: {hyp.target_region}")
            print(f"     Reasoning: {hyp.reasoning[:120]}...")

        # 3. SEARCH — convert hypotheses to experiment configs
        print(f"\n[3/7] Search — expanding hypotheses into experiments...")
        all_configs = []
        for hyp in hypotheses:
            self.db.save_hypothesis(hyp)
            configs = self.search_engine.expand_hypothesis(hyp)
            for cfg in configs:
                exp = Experiment(
                    id=f"exp_{self._iteration}_{hyp.id[:8]}_{len(all_configs)}",
                    hypothesis_id=hyp.id,
                    config=cfg,
                )
                self.db.create_experiment(exp)
                all_configs.append(exp)

        print(f"  Created {len(all_configs)} experiment configurations")

        for exp in all_configs:
            exp.transition(ExperimentStatus.APPROVED)
            self.db.update_experiment(exp)

        # 4. EXECUTE — run the experiments (simulated)
        print(f"\n[4/7] Execute — running {len(all_configs)} experiments (simulated)...")
        for exp in all_configs:
            exp.transition(ExperimentStatus.RUNNING)
            self.db.update_experiment(exp)

            success = self.experiment_runner.run_training(exp.id)

            # Re-fetch from DB after runner writes metrics
            exp = self.db.get_experiment(exp.id)

            if exp.status != ExperimentStatus.RUNNING:
                # Runner may have reset status; fix it
                exp.transition(ExperimentStatus.RUNNING)
                self.db.update_experiment(exp)
                exp = self.db.get_experiment(exp.id)

            if success:
                ok = exp.transition(ExperimentStatus.EVALUATING)
                if not ok:
                    exp.status = ExperimentStatus.EVALUATING
            else:
                exp.transition(ExperimentStatus.FAILED)
            self.db.update_experiment(exp)

            train_loss = exp.metrics.get("training_loss", 0.0)
            forgetting = exp.metrics.get("forgetting_index", 0.0)
            print(f"  {exp.id}: train_loss={float(train_loss):.4f}, forgetting={float(forgetting):.3f}")

        # 5. EVALUATE — run benchmarks
        print(f"\n[5/7] Evaluate — benchmarking...")
        evaluating = self.db.list_experiments(status=ExperimentStatus.EVALUATING)
        for exp in evaluating:
            feedback = self.evaluator.evaluate(exp.id)
            exp.feedback = feedback
            exp.transition(ExperimentStatus.COMPLETED)
            self.db.update_experiment(exp)

            scores = {r.benchmark_name: r.score for r in feedback.benchmark_results}
            print(f"  {exp.id}: {scores}")

        # 6. STORE — already done by evaluator
        print(f"\n[6/7] Store — all results saved to database")

        # 7. LEARN — check generalization
        print(f"\n[7/7] Learn — checking generalization...")
        completed = self.db.list_experiments(status=ExperimentStatus.COMPLETED, limit=5)
        for exp in completed:
            alerts = self.generalization_monitor.check_alert_conditions(exp.id)
            if alerts:
                for alert in alerts:
                    print(f"  ⚠️  Alert: {alert}")
        print(f"  No generalization issues detected")

        if evaluating:
            return evaluating[-1].id
        return None

    def run(self, max_iterations: Optional[int] = None):
        """Run the research loop continuously."""
        max_iter = max_iterations or self.config.loop.max_iterations
        iteration = 0

        while True:
            iteration += 1
            if max_iter > 0 and iteration > max_iter:
                print("\nReached max iterations")
                break

            self.run_one_cycle()
            sleep_time = self.config.loop.sleep_seconds
            if sleep_time > 0:
                print(f"\nSleeping {sleep_time}s until next cycle...")
                time.sleep(sleep_time)

        # Print summary
        print("\n" + "=" * 50)
        print("  CYCLE SUMMARY")
        print("=" * 50)
        completed = self.db.list_experiments(status=ExperimentStatus.COMPLETED)
        hypotheses = self.db.list_hypotheses()
        print(f"  Total completed experiments: {len(completed)}")
        print(f"  Total hypotheses generated: {len(hypotheses)}")
        if hypotheses:
            print(f"\n  Hypotheses:")
            for h in hypotheses:
                print(f"    • {h.description[:80]}... ({h.confidence:.0%} confidence)")


def main():
    """Entry point for the AROS CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="AROS — Autonomous Research Operating System")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--iterations", type=int, default=0, help="Max iterations (0 = unlimited)")
    parser.add_argument("--db", type=str, help="Path to database file (overrides config)")

    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.db:
        cfg.database.path = args.db

    loop = AROSLoop(cfg)

    if args.once:
        exp_id = loop.run_one_cycle()
        if exp_id:
            print(f"\nCompleted experiment: {exp_id}")
        else:
            print("\nNo experiments this cycle")
    else:
        loop.run(max_iterations=args.iterations)


if __name__ == "__main__":
    main()
