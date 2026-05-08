# Chess LLM Benchmark — Findings

**Ryan Brew · Chess LLM Benchmark · May 2026**
**Dataset:** 526,662 evaluations across 22 models × 4 difficulty tiers × ~6,000 positions per cell
**Ground truth:** Stockfish 17 (depth 22)
**Positions:** Lichess puzzle database (stratified by puzzle rating into easy / medium / hard / extreme)

> **Revision note (2026-05-07):** During post-hoc review, three measurement artefacts in the original metric definitions were identified and fixed. The qualitative findings below — uniform failure across scale, no scaling benefit, gemma4 family on top, llama3.3:70b near the bottom — all survive the correction. What changed is the **magnitude** of certain numbers. See "Methodology and metric artefacts" at the end of this document for the full audit.

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

**Local LLMs cannot play chess to a strong-club-player level — and this does not improve with model size.**

All 22 models, spanning 2B to 70B parameters, perform poorly on move legality, position evaluation, and move quality. Scaling up parameter count shows no consistent benefit. The findings suggest these models have learned the surface syntax of chess notation (SAN/FEN) from training data but have not developed the underlying spatial reasoning required to play chess at a competent level.

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

## Finding 2: Move quality is poor — uniform across architectures

For legal moves, performance is measured in **centipawn loss (CPL)** — the difference in position evaluation between the model's move and Stockfish's best move. CPL is reported under the standard Lichess convention with both endpoints clamped to ±1000 cp before subtraction (see *Methodology and metric artefacts* below for why; previous versions of this document reported the unclamped centipawn loss, which was inflated by mate-encoded scores).

| Reference point | Approximate clamped CPL |
|----------------|----------------|
| Stockfish (perfect play) | 0 |
| Magnus Carlsen (peak) | ~5 |
| Strong club player (~2000 Elo) | ~50–100 |
| Complete beginner | ~200–400 |
| **Best model in this study (gemma4:31b)** | **634** |
| **Median model in this study** | **~705** |
| **Worst model (gemma4:e2b)** | **727** |

Every model in this study plays at a level a few times worse than a complete beginner, and **the spread between the best and worst model is only ~14%** (634 → 727). The Gemma 4 26B and 31B models are the best performers (Google's newest architecture), but still average 634–650 CPL — roughly 6× worse than a beginner human and ~10× worse than a strong club player.

The same ranking holds under the alternative win-probability-loss metric (Δ win-probability × 1000), where models cluster between 283 (gemma4:31b, best) and 321 (gemma4:e2b, worst) milli-WP per move. Either way, the dispersion across 2B → 70B is small, and all models are bad.

---

## Finding 3: Models cannot determine who is winning

T1 asks models to estimate the Stockfish centipawn evaluation for a position. The key sub-metric is whether the model at least gets the **direction** right — does it correctly identify whether White is winning, Black is winning, or the position is equal?

- Overall direction accuracy across all models and positions: **46.3%**
- Random baseline for a 3-class problem: **33%**
- Best model: deepseek-r1:14b — **53.0%**
- Worst model: llama3.3:70b — **34.3%**

Models are barely above random chance at identifying who is winning — and the largest model in the study (70B) is near the bottom of this ranking.

The absolute centipawn error follows a pattern that initially looks like improvement with difficulty (easy: 1,722 cp → extreme: 311 cp) but this is an encoding artefact: easy Lichess positions tend to have decisive Stockfish evaluations, often mate-in-N, which Stockfish encodes internally as ~±10,000 cp. The model's eval output is itself clamped to ±2,000 cp, so on mate-truth positions the model's "error" is automatically ~8,000 cp regardless of any reasoning. **Excluding mate-truth positions, the mean T1 absolute error is 143 cp / median 67 cp** — much closer to real chess interpretability.

Under the magnitude-invariant relative-error metric (`|model − truth| / max(|truth|, 100)`), the error is **flat across all difficulty tiers at ~1.37** — meaning model calibration does not change with position complexity.

---

## Finding 4: Verbal explanation partially decorrelates from chess ability

T3 scores the quality of positional explanations on a 0–2 scale (two binary criteria: correct side identification, correct theme/reasoning). Reported using the corrected camelCase-aware theme matcher (the original matcher missed 70% of theme classes — see methodology).

| Model | T3 Score (v2 matcher) | T2 clamped CPL |
|-------|----------|--------|
| gemma4:26b | **0.776** | 650 |
| gemma4:31b | 0.668 | **634** |
| solar:10.7b | 0.632 | 692 |
| llama3.3:70b | 0.602 | 713 |
| phi4:14b | 0.593 | 715 |
| codellama:34b | **0.310** | 701 |

The Gemma 4 family dominates on both axes (best on move quality and best on explanation quality), so the verbal/mechanical decoupling is weaker than the original report suggested. However, it is still present further down the ranking: solar:10.7b, llama3.3:70b, and phi4:14b all explain well but play at near-median CPL, and CodeLlama-34B (specifically trained on structured/logical code) produces the worst explanations despite having strong T2 stats — suggesting that proficiency in formal symbolic domains does not transfer to chess reasoning.

---

## Finding 5: Model size does not predict chess performance

Sorting all models by parameter count reveals no monotonic relationship with any metric:

| Params | Model | T2 clamped CPL | T3 Score (v2) | T1 Direction |
|--------|-------|--------|----------|-------------|
| 7B | mistral:7b | **691** | 0.579 | 48.0% |
| 8B | llama3.1:8b | 724 | 0.367 | 45.4% |
| 11B | solar:10.7b | 692 | 0.632 | 48.2% |
| 14B | deepseek-r1:14b | 701 | 0.571 | **53.0%** |
| 31B | gemma4:31b | **634** | 0.668 | 38.8% |
| 32B | qwen2.5:32b | 725 | 0.503 | 40.9% |
| 70B | llama3.3:70b | 713 | 0.602 | 34.3% |

Notable inversions:
- Mistral-7B (691 CPL) outperforms Llama3.3-70B (713 CPL) on move quality — a 10× smaller model is slightly better.
- Qwen2.5-32B (725 CPL) is the second-worst on move quality — larger than all but two models.
- Direction accuracy actually *decreases* for the three largest models (32B, 31B, 70B).

**Architecture and training methodology appear to matter far more than scale.** The Gemma 4 family (Google's 2025 models) achieves the best move quality at 26B–31B and the best explanation quality at 26B. The DeepSeek-R1 chain-of-thought models achieve the best direction accuracy at both 7B and 14B sizes — suggesting that explicit reasoning chains help with the classification problem even if they do not improve move quality.

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
| T2 clamped CPL (lower=better) | gemma4:31b | 634 | gemma4:e2b | 727 | 702 |
| T2 win-prob loss (×1000, lower=better) | gemma4:31b | 283 | gemma4:e2b | 321 | 311 |
| T3 explanation score (v2) | gemma4:26b | 0.776 | codellama:34b | 0.310 | 0.520 |

---

## Interpretation for a Research Audience

### What this is evidence for

The uniform failure across all model sizes on T2 (move legality) is the strongest result. Move legality is a hard constraint — there is no ambiguity. A model either knows which squares a knight can reach from e4 or it does not. The fact that **35% of moves are illegal regardless of whether the model has 2B or 70B parameters**, and that **explicitly providing the legal move list still fails 32% of the time**, strongly indicates that these models do not maintain an internal representation of the board state.

The clamped-CPL spread of 634–727 across all 22 models is also notable: the gap between best and worst is small, and even the best-performing architecture remains far below strong-club-player level. Combined with the legality result, this supports the hypothesis that LLMs are **syntactic mimics** in formal game domains — they produce outputs that are distributionally similar to chess notation in training data, but those outputs are not grounded in the underlying game semantics.

### What this is not evidence for

This study does not show that LLMs cannot reason in any formal domain — only that chess, as tested here, is not solved by current local models up to 70B parameters. Commercial models (GPT-4, Claude) are not included and may perform differently. The benchmark also uses six fixed prompt formats; it is possible that heavily engineered prompting or fine-tuning on chess data would change the picture significantly.

### The scale question

The parameter-scaling result is striking but should be interpreted carefully. All models tested here are **general-purpose models trained on broad internet data**. Scale in general-purpose pre-training does not obviously translate to emergent chess ability when chess games represent a small fraction of the training corpus. A fair test of the scaling hypothesis would require models specifically trained or fine-tuned on chess data — which is outside the scope of this study.

### Next steps

The most natural extension is comparison with commercial models (Claude, GPT-4o) using the same evaluation pipeline. A spec for Anthropic API integration is already written (`specs/commercial-models.md`). The relative cost of ~$13 for Claude Haiku makes this feasible as an immediate comparison.

---

## Methodology and metric artefacts

This section documents three measurement artefacts identified during post-hoc review, and the corrections that produce the numbers reported above. The raw data (`results/evaluations.jsonl`, 526,662 records) was unchanged; only the postprocessing was updated. All corrections are unit-tested in `tests/test_artefact_fixes.py`.

### A1 — CPL inflation by mate-encoded scores

The Stockfish wrapper encodes mate-in-N as a centipawn value: `±10000 - mate_in_plies × 10`. For long mating sequences this produces values up to ±16,000 cp in the dataset. Without clamping, those values leak into the CPL difference whenever either endpoint of the move is mate-encoded:

| | % of evaluations |
|---|---|
| Mate-encoded *start* position | 4.3% |
| Mate-encoded *eval-after-move* | 20.0% |
| Either endpoint mate-encoded | 22.0% |

In particular, **20% of model moves walk into a position Stockfish sees as a forced mate**, and that single case dominates the inflation. The raw uncapped mean was 4,786 CPL; under the standard Lichess convention (both endpoints clamped to ±1000 cp before subtraction), the mean is **702 CPL**. The model rankings are essentially unchanged but the spread tightens (1.33× best/worst → 1.15×).

The `t2_cpl` column in the raw JSONL is preserved for transparency. The headline metric in this report is `t2_cpl_clamped`, with `t2_wp_loss` (Δ win-probability × 1000, bounded [0, 1000]) reported as a robustness check.

### A2 — T1 absolute error inflated by mate-truth rows

The model's eval output is clamped to ±2000 cp (`src/evaluator.py:81`). The Stockfish ground-truth eval is not clamped and ranges to ±16,000 cp on mate-encoded positions. So whenever the truth is mate-encoded, the model's error is ~9,966 cp on average, **regardless of any reasoning**.

| Truth type | Mean abs error | Median |
|---|---|---|
| Mate truth (`|sf| ≥ 9000`) | 9,966 | 9,999 |
| Non-mate truth | 143 | 67 |

Reported numbers above use `t1_abs_error_excl_mate` for the headline, and the relative-error metric (already in the dataset) for the cross-difficulty comparison.

### A3 — T3 theme matcher missed 70% of theme classes

Lichess puzzle theme labels are camelCase (`advancedPawn`, `kingsideAttack`, `backRankMate`, `bishopEndgame`). The original synonym dictionary in `src/evaluator.py` was keyed by snake_case strings (`passed_pawn`, `back_rank`), and the fallback `theme.lower().replace("_", " ")` produced gibberish like `"advancedpawn"` and `"kingsideattack"` that no human writes. Resulting theme-component match rates on the original metric:

| Theme | n | Original match rate |
|---|---|---|
| advancedPawn | 37,957 | 0.0% |
| kingsideAttack | 15,093 | 0.0% |
| backRankMate | 11,914 | 0.0% |
| bishopEndgame | 5,279 | 0.0% |
| crushing | 194,214 | 0.2% |
| advantage | 151,872 | 22.3% |

The original T3 score was therefore mostly the side-identification component (P1) plus accidental matches on `advantage` and `mate`. Models that wrote about the theme using its English-equivalent phrase (e.g. "advanced pawn", "kingside attack") got no credit for it.

The corrected `THEME_SYNONYMS` table is keyed by exact Lichess label and includes natural-English synonym lists; the fallback now splits camelCase to space-separated lowercase words. This produces a meaningfully different ranking on T3 — gemma4:26b moves from 5th place (0.51) to 1st (0.78), reflecting that Gemma actually engages with the puzzle theme rather than rambling about "advantage".

### A4 — H1 / H2 hypothesis tests under corrected metrics

Both pre-registered hypotheses (T1 error increases with difficulty; T2 CPL increases with difficulty) come out as **not supported** under the corrected primary metrics. The values are essentially flat across difficulty tiers:

| | Easy | Medium | Hard | Extreme |
|---|---|---|---|---|
| T1 relative error | 1.36 | 1.37 | 1.37 | 1.42 |
| T2 clamped CPL | 698 | 715 | 705 | 688 |
| T2 WP-loss (×1000) | 303 | 315 | 315 | 309 |

The original "easy → extreme" reversal in raw absolute CPL (5,507 → 4,070) was the same encoding artefact as A1 — easy Lichess puzzles are mate puzzles, so they have more mate-encoded eval-after positions, inflating their CPL. **Performance is flat across difficulty tiers.** This is a stronger result than the original direction-flipped finding: models do not get worse on harder positions because they were never engaging with the spatial structure of any of them.

### Other audited issues (no inflation, but worth recording)

- **35% parse-failure rate.** A third of all jobs failed to produce a parseable answer for at least one of the three tasks; the three failure rates are nearly identical (35.4% / 34.0% / 34.5%), suggesting a single root cause (when format breaks, all three fields go null). Failed-parse rows are excluded from averages, as expected.
- **side_claimed bias toward "White".** Models claim White is winning 55.5% of the time, Black 37.1%, Equal 7.4% — a ~5pp directional bias. Real ground-truth split is ~50/50. Inflates side-correctness on White-favourable positions slightly.
- **Bimodal eval distribution.** 92% of Stockfish ground-truth evals are within ±300 cp; 7% are mate-encoded (±9,000+); the 300–9,000 cp band is essentially empty. Lichess puzzles are tactical, so depth-22 Stockfish either has not yet "seen" the tactic (eval ≈ 0) or has resolved it to mate. There is almost no middle ground.
- **No duplicate records.** 0 duplicate `(model, position_id, prompt_format)` keys — retry/correction jobs are not double-counted.

---

*All data: `results/evaluations.jsonl` (526,662 records). Metrics computed by `scripts/generate_plots.py`. Corrected metric definitions in `src/metrics.py` (clamped CPL, WP-loss, mate-aware T1 error, v2 theme matcher). Plots: `results/plots/`. Raw metric CSVs: `results/metrics/`. Spec for the artefact corrections: `specs/artefact-fixes.md`.*
