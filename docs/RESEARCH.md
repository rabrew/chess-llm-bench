# Research Notes — Stripe Young Scientist 2027

---

## Research Question

> **"Can AI language models trained on text learn to reason spatially, as measured by chess performance?"**

---

## Strengthened Hypothesis

> *"LLMs with reasoning-optimized architectures will demonstrate stronger spatial reasoning in chess than parameter count alone predicts, suggesting emergent spatial cognition rather than pure pattern matching."*

---

## What Makes This Strong for Competitions

- **Measurable and quantitative** — centipawn loss and move accuracy give clear, objective results
- **Novel** — very little peer-reviewed work on LLM chess benchmarking at this scale
- **5.8M positions across 19 models** is unusually rigorous for a student project, approaching publishable-level scale
- **Connects to broader AI questions** judges care about: emergent reasoning, scaling laws, and whether LLMs truly "understand" or just pattern match
- **The deeper question isn't really about chess** — it's about whether LLMs genuinely *reason* or pattern match. Chess is a clean probe because every position has an objectively correct answer

---

## Results You Can Pull From the Data

### Model comparisons
- Which LLM plays the best chess overall?
- Do bigger models (70B) actually play better than smaller ones (7B)?
- Which model has the lowest centipawn loss on average?

### Prompt format impact
- Does giving the model move history (PGN+FEN) help vs just the position (FEN only)?
- Does chain-of-thought prompting improve move quality?

### Difficulty scaling
- Do models struggle more on tactical puzzles vs endgames?
- Which models hold up on "extreme" difficulty vs falling apart?

### Correction loop
- When told their move was wrong, do models correct themselves?
- Are some models better at self-correction than others?

### Position type breakdown
- Do models play openings better than endgames?
- Are certain themes (forks, pins, checkmates) harder for LLMs?

### Model families
- Does Qwen outperform Llama at the same size?
- Do reasoning models (deepseek-r1) play better than standard chat models?

---

## Addressing the Training Data Contamination Argument

Judges **will** ask: *"Aren't LLMs just memorizing chess games from their training data?"*

### What 5.8M positions does for you
- Statistically, the sheer volume makes it unlikely every position appeared in training data
- The *distribution* of performance across difficulty tiers is more revealing than any single position — a model that memorized openings would collapse on endgames

### The stronger methodological arguments
- **Centipawn loss** measures quality of reasoning, not just right/wrong — a memorized move would score perfectly, but reasoning errors show up in the *magnitude* of mistakes
- **The correction loop is your best defense** — if a model adjusts its move after being told it was wrong, that's active reasoning, not recall
- **Difficulty-scaled performance curves** follow a pattern consistent with reasoning limits, not memory gaps

### How to phrase it for judges
> *"While training data contamination cannot be fully ruled out, the correction loop methodology, difficulty-scaled performance curves, and centipawn loss distributions collectively suggest the models are reasoning over positions rather than recalling memorized games."*

This is more defensible than claiming 5.8M positions alone rules it out — judges will respect the nuance.

---

## Timeline

| Milestone | Target |
|-----------|--------|
| Benchmark run complete | March 2026 |
| Initial analysis & plots | April 2026 |
| Re-run with newer models (as released) | Late 2026 |
| Write-up complete | Early 2027 |
| Stripe Young Scientist submission | 2027 |

**Note:** The 2027 timeline gives you the opportunity to re-run the benchmark against newer models as they release, potentially showing how spatial reasoning scales over time across model generations.
