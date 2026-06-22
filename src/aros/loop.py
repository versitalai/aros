"""Main research loop — orchestrates the full AROS pipeline.

The loop follows a fixed sequence:
    Observe → Hypothesize → Search → Execute → Evaluate → Store → Learn → Repeat
"""

import time
import logging
from typing import Optional

from .config import Config
from .database import ExperimentDB
from .research_agent import ResearchAgent
from .search_engine import SearchEngine
from .experiment_runner import ExperimentRunner
from .evaluator import Evaluator
from .dataset_registry import DatasetRegistry
from .generalization_monitor import GeneralizationMonitor
from .models import Experiment, ExperimentStatus

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

    def run_one_cycle(self) -> Optional[str]:
        """Execute one full research loop cycle.

        Returns the experiment ID if one was completed, None otherwise.
        """
        self._iteration += 1
        logger.info(f"=== AROS Cycle {self._iteration} ===")

        # 1. OBSERVE — analyze experiment history
        recent = self.research_agent.analyze_history()
        logger.info(f"Observed {len(recent)} recent experiments")

        # 2. HYPOTHESIZE — generate research hypotheses
        hypotheses = self.research_agent.propose_hypotheses(recent)
        if not hypotheses:
            logger.info("No hypotheses generated this cycle")
            return None

        # 3. SEARCH — convert hypotheses to experiments
        for hyp in hypotheses:
            self.db.save_hypothesis(hyp)
            configs = self.search_engine.expand_hypothesis(hyp)

            for cfg in configs:
                exp = Experiment(
                    id=f"exp_{self._iteration}_{hyp.id}_{len(configs)}",
                    hypothesis_id=hyp.id,
                    config=cfg,
                )
                self.db.create_experiment(exp)

        # 4. EXECUTE — run approved experiments
        pending = self.db.list_experiments(status=ExperimentStatus.PROPOSED)
        for exp in pending:
            exp.transition(ExperimentStatus.RUNNING)
            self.db.update_experiment(exp)

            success = self.experiment_runner.run_training(exp.id)
            if success:
                exp.transition(ExperimentStatus.EVALUATING)
            else:
                exp.transition(ExperimentStatus.FAILED)
            self.db.update_experiment(exp)

        # 5. EVALUATE — run benchmarks
        evaluating = self.db.list_experiments(status=ExperimentStatus.EVALUATING)
        for exp in evaluating:
            feedback = self.evaluator.evaluate(exp.id)
            exp.feedback = feedback
            exp.transition(ExperimentStatus.COMPLETED)
            self.db.update_experiment(exp)

        # 6. STORE — benchmark results already stored by evaluator

        # 7. LEARN — check generalization
        completed = self.db.list_experiments(status=ExperimentStatus.COMPLETED, limit=5)
        for exp in completed:
            alerts = self.generalization_monitor.check_alert_conditions(exp.id)
            if alerts:
                logger.warning(f"Generalization alerts for {exp.id}: {alerts}")

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
                logger.info("Reached max iterations")
                break

            self.run_one_cycle()
            sleep_time = self.config.loop.sleep_seconds
            logger.info(f"Sleeping {sleep_time}s until next cycle")
            time.sleep(sleep_time)


def main():
    """Entry point for the AROS CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="AROS — Autonomous Research Operating System")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--iterations", type=int, default=0, help="Max iterations (0 = unlimited)")

    args = parser.parse_args()

    config = Config()
    if args.config:
        from .config import load_config
        config = load_config(args.config)

    loop = AROSLoop(config)

    if args.once:
        exp_id = loop.run_one_cycle()
        if exp_id:
            print(f"Completed experiment: {exp_id}")
        else:
            print("No experiments this cycle")
    else:
        loop.run(max_iterations=args.iterations)


if __name__ == "__main__":
    main()
