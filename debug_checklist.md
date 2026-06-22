# AROS Debug Checklist

Run `python -m pytest tests/ -v` after each phase. Every test MUST pass before moving
to the next phase.

## Phase 1 — Project Scaffold ✅ / ❌
- [ ] `src/aros/` package exists with `__init__.py`
- [ ] `config.py` loads defaults without YAML file
- [ ] `config.py` reads YAML config files
- [ ] `config.py` supports `AROS_*` env var overrides
- [ ] `models.py` defines all dataclasses: Hypothesis, SearchSpace, ExperimentConfig, Experiment, DatasetInfo, BenchmarkResult, ExperimentFeedback
- [ ] `models.py` defines `ExperimentStatus` enum with all 8 states
- [ ] `models.py` `Experiment.transition()` validates state changes
- [ ] All dataclasses have `to_dict()` methods
- [ ] `pyproject.toml` is valid PEP 621

## Phase 2 — Experiment Database ✅ / ❌
- [ ] Tables created on init (experiments, hypotheses, datasets, benchmark_results)
- [ ] `create_experiment()` inserts and returns ID
- [ ] `get_experiment()` retrieves by ID
- [ ] `update_experiment()` updates fields
- [ ] `transition_experiment()` with valid state change
- [ ] `transition_experiment()` rejects invalid state change
- [ ] `list_experiments()` with status filter
- [ ] `save_hypothesis()` and `get_hypothesis()`
- [ ] `register_dataset()` and `list_datasets()`
- [ ] `save_benchmark_result()` and `get_benchmark_results()`

## Phase 3 — Dataset Registry ✅ / ❌
- [ ] `register()` stores dataset metadata
- [ ] `query()` filters by type and quality
- [ ] `get_summary_stats()` returns aggregate stats
- [ ] `validate_mixture()` checks synthetic cap and quality

## Phase 4 — Research Agent ✅ / ❌
- [ ] `analyze_history()` returns recent experiments
- [ ] `propose_hypotheses()` returns list of `Hypothesis` from LLM
- [ ] Calls Ollama (`llama3.2:3b`) and parses JSON response
- [ ] Handles ````json``` markdown fences in LLM output
- [ ] Returns empty list on LLM error gracefully
- [ ] `explain_reasoning()` returns text from DB hypothesis
- [ ] *Integration:* Running `aros --once` produces real hypotheses with reasoning

## Phase 5 — Search Engine ✅ / ❌
- [ ] `expand_hypothesis()` returns list of `ExperimentConfig`
- [ ] `get_baseline_config()` returns config with defaults
- [ ] `register_search_space()` accepts formal search space
- [ ] `suggest_next_parameters()` returns dict (for BO loop)
- [ ] *Integration:* Search Engine generates N configs per hypothesis

## Phase 6 — Evaluator ✅ / ❌
- [ ] `evaluate()` returns `ExperimentFeedback`
- [ ] `evaluate_visible()` runs visible benchmarks only
- [ ] `evaluate_hidden()` runs hidden benchmarks only
- [ ] `store_hidden_results()` saves without exposing to agent
- [ ] *Integration:* Evaluator loads model checkpoint and runs actual benchmarks

## Phase 7 — Experiment Runner ✅ / ❌
- [ ] `assemble_dataset()` returns dataset assembly metadata
- [ ] `setup_training()` returns training config dict
- [ ] `run_training()` executes training, updates experiment status
- [ ] `get_checkpoint_path()` returns path or None
- [ ] `estimate_cost()` returns cost estimate
- [ ] *Integration:* Runner does actual LoRA training

## Phase 8 — Generalization Monitor ✅ / ❌
- [ ] `compute_generalization_score()` returns float
- [ ] `check_alert_conditions()` returns list of alert strings
- [ ] `should_pause_autonomy()` returns bool
- [ ] *Integration:* Monitor triggers alerts on visible/hidden divergence

## Phase 9 — Integration ✅ / ❌
- [ ] `AROSLoop.run_one_cycle()` executes full loop
- [ ] `AROSLoop.run()` runs continuously with sleep
- [ ] CLI entry point works: `aros --once`
- [ ] CLI entry point works: `aros --iterations 5`
- [ ] All 20+ tests pass with `python -m pytest tests/ -v`
