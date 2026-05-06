# Chess LLM Benchmark — Findings

**Ryan Brew · Chess LLM Benchmark · May 2026**  
**Dataset:** 526,662 evaluations across 22 models × 4 difficulty tiers × ~6,000 positions per cell  
**Ground truth:** Stockfish 17 (depth 20)  
**Positions:** Lichess puzzle database (stratified by puzzle rating into easy / medium / hard / extreme)

---

## Overview

This benchmark tests whether locally-run open-source LLMs understand chess — not as a proxy for intelligence, but as a direct test of spatial reasoning in a fully formal, closed domain where ground truth is computable. Each model was tested on three tasks per position:

| Task | What it measures |
|------|-----------------|
| **T1 — Evaluation** | Can the model estimate who is winning and by how much (centipawns)? |
| **T2 — Move selection** | Can the model select a legal, high-quality move? |
| **T3 — Explanation** | Can the model explain why a position favours one side? |

---

## Main Conclusion

**Local LLMs cannot play chess, evaluate positions, or reliably produce legal moves — and this does not improve with model size.**

All 22 models, spanning 2B to 70B parameters, perform at or near chance on move legality and position evaluation. Scaling up parameter count shows no consistent benefit. The findings suggest these models have learned the surface syntax of chess notation (SAN/FEN) from training data but have not developed the underlying spatial reasoning required to understand chess positions.

---

## Finding 1: One-third of moves are illegal — at every scale

Across all 526,662 evaluations, models produced a legal chess move **64.6%** of the time. All 22 models cluster tightly between 60–67%, regardless of size:

| Model size | Legal move rate |
|-----------|----------------|
| 2B–4B (small) | 63–66% |
| 7B–12B (mid) | 60–66% |
| 14B (mid-large) | 60–67% |
| 26B–35B (large) | 66–67% |
| 47B–70B (XL) | 60–66% |

The 70B model (llama3.3) achieves 66.4% — essentially identical to the 4B model (gemma3) at 65.8%. Scale buys nothing here.

**The retry experiment makes this worse.** Models that produced an illegal move were re-prompted with the complete list of legal moves for that position and asked to pick one. Even with the answer set presented explicitly, only **67.6%** selected a valid option — meaning roughly a third of models still failed to choose from a given list. This rules out notation/formatting as the primary failure mode; models genuinely do not know which moves are legal.

---

## Finding 2: Move quality is catastrophically bad — worse than informed guessing

For legal moves, performance is measured in **centipawn loss (CPL)** — the difference in position evaluation between the model's move and Stockfish's best move. Lower is better.

| Reference point | Approximate CPL |
|----------------|----------------|
| Stockfish (perfect play) | 0 |
| Magnus Carlsen (peak) | ~5 |
| Strong club player (~2000 Elo) | ~50–100 |
| Complete beginner | ~200–400 |
| **Best model in this study (gemma4:31b)** | **3,930** |
| **Median model in this study** | **~4,800** |
| **Worst model (gemma4:e2b)** | **5,238** |

Every model in this study plays at a level indistinguishable from near-random move selection in these positions. The Gemma4 26B and 31B models are the best performers (Google's newest architecture), but still average 3,930–4,140 CPL — roughly 40× worse than a beginner human.

---

## Finding 3: Models cannot determine who is winning

T1 asks models to estimate the Stockfish centipawn evaluation for a position. The key sub-metric is whether the model at least gets the **direction** right — does it correctly identify whether White is winning, Black is winning, or the position is equal?

- Overall direction accuracy across all models and positions: **46.3%**
- Random baseline for a 3-class problem: **33%**
- Best model: deepseek-r1:14b — **53.0%**
- Worst model: llama3.3:70b — **34.3%**

Models are barely above random chance at identifying who is winning — and the largest model in the study (70B) is near the bottom of this ranking.

The absolute centipawn error follows a pattern that initially looks like improvement with difficulty (easy: 1,722 cp → extreme: 311 cp) but this is a measurement artefact: easy Lichess positions tend to have very high Stockfish evaluations (mean ±1,652 cp — decisive advantages), while extreme positions are more balanced (mean ±211 cp), so a model guessing near zero racks up huge absolute error on easy positions regardless of any actual reasoning. When normalised by position magnitude (**relative error = |model − truth| / max(|truth|, 100)**), the error is essentially **flat across all difficulty tiers at ~1.37** — meaning model calibration does not change with position complexity.

---

## Finding 4: Verbal explanation partially decorrelates from chess ability

T3 scores the quality of positional explanations on a 0–1 scale (two criteria: correct side identification, correct theme/reasoning).

| Model | T3 Score | T2 CPL |
|-------|----------|--------|
| solar:10.7b | **0.550** | 4,463 |
| llama3.3:70b | **0.537** | 5,051 |
| gemma4:26b | 0.513 | 4,140 |
| gemma4:31b | 0.507 | **3,930** |
| codellama:34b | **0.253** | 4,702 |

Solar-10.7B produces the best positional explanations of any model in the study but plays with mediocre move quality (4,463 CPL, 12th of 22). Llama3.3-70B is the second-best explainer but has the worst move quality of all models (5,051 CPL, last). CodeLlama-34B, specifically trained on structured/logical code, produces the worst explanations (0.253 T3 score) — suggesting that proficiency in formal symbolic domains does not transfer to chess reasoning.

This divergence between verbal description of chess concepts and mechanical chess ability implies models have absorbed **chess vocabulary and heuristics** from training text without acquiring the underlying spatial reasoning those heuristics describe.

---

## Finding 5: Model size does not predict chess performance

Sorting all models by parameter count reveals no monotonic relationship with any metric:

| Params | Model | T2 CPL | T3 Score | T1 Direction |
|--------|-------|--------|----------|-------------|
| 7B | mistral:7b | **4,307** | 0.349 | 48.0% |
| 8B | llama3.1:8b | 5,194 | 0.335 | 45.4% |
| 11B | solar:10.7b | 4,463 | **0.550** | 48.2% |
| 14B | deepseek-r1:14b | 4,840 | 0.430 | **53.0%** |
| 31B | gemma4:31b | **3,930** | 0.507 | 38.8% |
| 32B | qwen2.5:32b | 5,102 | 0.386 | 40.9% |
| 70B | llama3.3:70b | 5,051 | 0.537 | 34.3% |

Notable inversions:
- Mistral-7B (4,307 CPL) outperforms Llama3.3-70B (5,051 CPL) on move quality — a 10× smaller model wins decisively.
- Qwen2.5-32B (5,102 CPL) is the second-worst on move quality — larger than all but two models.
- Direction accuracy actually *decreases* for the three largest models (32B, 31B, 70B).

**Architecture and training methodology appear to matter far more than scale.** The Gemma4 family (Google's 2025 models) achieves the best move quality at 26B–31B. The DeepSeek-R1 chain-of-thought models achieve the best direction accuracy at both 7B and 14B sizes — suggesting that explicit reasoning chains help with the classification problem even if they do not improve move quality.

---

## Finding 6: The DeepSeek-R1 anomaly

The DeepSeek-R1 models (7B and 14B) show a distinctive trade-off:

- **Best T1 direction accuracy in the study** (deepseek-r1:14b: 53%, deepseek-r1:7b: 50%)
- **Lowest legal move rates** (both around 60%)

The chain-of-thought reasoning process improves the model's ability to classify who is winning — a problem it can approach as a textual inference task ("the queen on d5 is dominant, the king is exposed...") — but the extended reasoning output makes it more likely to produce a move in a non-standard format that fails legality parsing. This suggests reasoning chains help with coarse-grained evaluation but not with the mechanical precision required to produce valid notation.

---

## Summary Table

| Metric | Best model | Score | Worst model | Score | All-model avg |
|--------|-----------|-------|-------------|-------|--------------|
| T1 direction accuracy | deepseek-r1:14b | 53.0% | llama3.3:70b | 34.3% | 46.3% |
| T2 legal move rate | qwen2.5:32b | 66.6% | deepseek-r1:7b | 60.5% | 64.6% |
| T2 CPL (lower=better) | gemma4:31b | 3,930 | gemma4:e2b | 5,238 | ~4,800 |
| T3 explanation score | solar:10.7b | 0.550 | codellama:34b | 0.253 | 0.409 |

---

## Interpretation for a Research Audience

### What this is evidence for

The uniform failure across all model sizes on T2 (move legality and quality) is the strongest result. Move legality is a hard constraint — there is no ambiguity. A model either knows which squares a knight can reach from e4 or it does not. The fact that **35% of moves are illegal regardless of whether the model has 2B or 70B parameters**, and that **explicitly providing the legal move list still fails 32% of the time**, strongly indicates that these models do not maintain an internal representation of the board state. They are generating text that looks like a chess move without verifying it against any spatial model of the position.

This is consistent with the hypothesis that LLMs are **syntactic mimics** in formal game domains: they produce outputs that are distributionally similar to chess notation in training data, but those outputs are not grounded in the underlying game semantics.

### What this is not evidence for

This study does not show that LLMs cannot reason in any formal domain — only that chess, as tested here, is not solved by current local models up to 70B parameters. Commercial models (GPT-4, Claude) are not included and may perform differently. The benchmark also uses three fixed prompt formats; it is possible that heavily engineered prompting or fine-tuning on chess data would change the picture significantly.

### The scale question

The parameter-scaling result is striking but should be interpreted carefully. All models tested here are **general-purpose models trained on broad internet data**. Scale in general-purpose pre-training does not obviously translate to emergent chess ability when chess games represent a small fraction of the training corpus. A fair test of the scaling hypothesis would require models specifically trained or fine-tuned on chess data — which is outside the scope of this study.

### Next steps

The most natural extension is comparison with commercial models (Claude, GPT-4o) using the same evaluation pipeline. A spec for Anthropic API integration is already written (`specs/commercial-models.md`). The relative cost of ~$13 for Claude Haiku makes this feasible as an immediate comparison.

---

*All data: `results/evaluations.jsonl` (526,662 records). Metrics computed by `scripts/generate_plots.py`. Plots: `results/plots/`. Raw metric CSVs: `results/metrics/`.*
