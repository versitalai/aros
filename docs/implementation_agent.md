# Implementation Agent — Code Generation from Hypotheses

## The Idea

The Research Agent (small local model, ~3B) proposes high-level ideas. A **larger, more capable model** then translates those ideas into actual code changes. This bridges the gap between "what to try" and "how to implement it."

## Why

Hyperparameter search is useful, but the real breakthroughs come from architectural changes, new loss functions, dataset strategies, and training methods. A small model can spot the pattern ("our agent keeps failing on multi-turn tool use — we should change how context is structured"), but a bigger model is better equipped to actually write the code.

## Flow

```
Hypothesis (from Research Agent, 3B)
    │
    ▼
Implementation Agent (bigger model)
    │  Reads: hypothesis text + relevant source files
    │  Writes: code diff / patch
    │
    ▼
Code Review (optional) → Apply Patch
    │
    ▼
Experiment Runner executes with new code
    │
    ▼
Evaluator measures if the change helped
    │
    ▼
Feedback goes back to Research Agent
```

## Design Notes

- The Implementation Agent should receive the full context: which file to modify, the current code, and the hypothesis reasoning.
- Patches should be testable in isolation before full experiment runs.
- The agent should be sandboxed — generated code runs in the experiment runner, not directly on the host.
- This could live in a new `implementation_agent.py` component between the Research Agent and the Search Engine in the loop.

## Models

- **Research Agent**: llama3.2:3b (current) — cheap enough to run every cycle
- **Implementation Agent**: A larger model (Gemma 4 27B? Qwen3 32B?) called on-demand when a hypothesis requires code changes
- Could route through Ollama for local or OpenRouter for cloud
