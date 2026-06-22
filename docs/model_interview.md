# Model Interview — Qualitative Evaluation via Agent-Model Conversations

## The Idea

After training a new LoRA checkpoint, the Research Agent doesn't just look at benchmark scores — it has a **structured conversation** with the trained model to understand *what* it learned, *how* it thinks, and *where* it still struggles.

Benchmarks tell you *what score* the model got. Interviews tell you *why*.

## Flow

```
Experiment Runner → LoRA checkpoint saved
                           ↓
Evaluator: runs benchmarks (quantitative scores)
                           ↓
Evaluator: loads checkpoint, opens chat session
                           ↓
Research Agent interviews the trained model
    "Solve this planning problem step by step."
    "Why did you choose that approach?"
    "What's your confidence on this type of task?"
    "Generate an example of a problem you find difficult."
                           ↓
Interview transcript + analysis → stored with experiment feedback
                           ↓
Agent uses qualitative findings + benchmark scores for next hypothesis
```

## What This Enables

- **Deeper insights** — a model might score well on HumanEval but reveal in conversation that it's memorizing patterns rather than reasoning. The agent detects this.
- **Failure mode discovery** — the trained model can articulate what it finds hard, guiding the next experiment more precisely than a score delta.
- **Qualitative trends across cycles** — over multiple experiments, the agent spots shifts in how the model explains its reasoning.
- **Natural language as a metric** — confidence, coherence, self-awareness, and reasoning quality all become trackable signals.

## Technical Notes

- The trained model runs through the same Ollama endpoint, just loaded from the LoRA checkpoint.
- No extra infrastructure — the chat API is already there.
- Interview questions should be structured and consistent across experiments so comparisons are meaningful.
- This fits naturally as an extension of the **Evaluate** phase in the loop.

## Todo

- [ ] Add an "interview prompt template" to the Research Agent config
- [ ] Wire checkpoint loading into the Evaluator so it can spin up the trained model
- [ ] Store interview transcripts alongside benchmark results in the experiment DB
- [ ] Feed qualitative signals back into the hypothesis generation context
