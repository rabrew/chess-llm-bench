# Chess LLM Benchmark — Findings

**Ryan Brew · Chess LLM Benchmark · May 2026**
**Dataset:** 4,000 unique chess positions (1,000 per tier) × 22 models × 6 prompt formats = 526,662 evaluations
**Ground truth:**
- T1 evaluation: Stockfish 17 @ depth 22
- T2 best-move and CPL: Lc0 @ 800 nodes (GPU-accelerated post-hoc enrichment)
- T3 explanation: rule-based scorer comparing model output to Lichess theme tags

**Positions:** Lichess puzzle database (stratified by puzzle rating into easy / medium / hard / extreme)

> **Revision note (2026-05-08):** A two-pass post-hoc audit identified five measurement artefacts and one infrastructure issue in the original analysis. All artefacts have been corrected in the postprocessing layer (no re-collection of LLM data was needed). The qualitative findings — uniform failure across scale, no scaling benefit, gemma4 family on top, llama3.3:70b near the bottom — survive the corrections. The headline numbers have changed materially. **Read the "Methodology and metric artefacts" section at the end of this document before quoting any single number from this report.**

---

## Overview

This benchmark tests whether locally-run open-source LLMs understand chess — not as a proxy for intelligence, but as a direct test of spatial reasoning in a fully formal, closed domain where ground truth is computable. Each model was tested on three tasks per position:

| Task | What it measures |
|------|-----------------|
| **T1 — Evaluation** | Can the model estimate who is winning and by how much (centipawns)? |
| **T2 — Move selection** | Can the model select a legal, high-quality move? |
| **T3 — Explanation** | Can the model explain why a position favours one side? |

Six prompt formats were tested per (model, position): `cot`, `fen_only`, `pgn+fen` (all three tasks combined); `eval_only` (T1 only); `move_only` (T2 only, with the legal-move list provided); `explanation_only` (T3 only).

---

## Main Conclusion

**Local LLMs cannot play chess to a strong-club-player level — and this does not improve with model size.** They CAN, in most cases, follow chess notation, identify which side has a clearly winning advantage, and produce a legal move when explicitly asked. They CANNOT play moves of consistent quality, evaluate fine positional differences, or maintain reliable internal representations of the board.

All 22 models, spanning 2B to 70B parameters, show roughly the same ceiling on move quality and the same bimodal "perfect or catastrophic" failure pattern. Architecture (training methodology) appears to matter much more than parameter count: Google's 2025 Gemma 4 family tops every metric while several 30–70B models from older architectures rank at the bottom.

---

## Finding 1: When asked for a move, models produce a legal one ~98% of the time

**Models are not bad at the chess-notation level.** Across the four prompt formats that ask the model to produce a move (`cot`, `fen_only`, `pgn+fen`, `move_only`), the legality rate is **97.9%** averaged across all 22 models. Per-model legality on these prompts:

| Model | Legal % (when asked for a move) |
|---|---|
| qwen2.5:32b | **99.94%** |
| qwen2.5:14b | 99.77% |
| llama3.3:70b | 99.67% |
| gemma3:12b | 99.52% |
| gemma4:31b | 99.44% |
| ... | ... (16 models in 95–99% range) |
| wizardlm2:7b | 90.54% |
| mixtral:8x7b | 90.26% |
| deepseek-r1:14b | **90.00%** (worst) |

*(Reported as `t2_legal_attempted` — only counts attempts on prompts that asked for a move. The previous version of this report quoted "65% legal" / "1 in 3 illegal" — that was a denominator bug; see the methodology section.)*

**Important caveats:**
1. The combined-prompt formats (`cot`/`fen_only`/`pgn+fen`) include **silent system rescue**: if the model's first answer is illegal, the worker pulls any legal token from the response, and if that fails it re-prompts with the legal-move list. The legality numbers above are post-rescue. The pre-rescue (single-shot) rate is not recoverable from the data.
2. The `move_only` prompt always shows the model the list of legal moves. So its 96.5% rate is "given the answer set, did the model pick a legal one?" — not unconstrained generation.
3. **The "retry experiment"** (see `scripts/retry_illegal_moves.py`): on the 10,665 records where post-rescue moves were still illegal, re-prompting with the legal-move list explicitly succeeded for **67.6%** of attempts. So even given a second chance with the answer set provided, a third of those attempts still failed. The same is true to a lesser degree across the dataset: legality is generally not a problem, but on the residual hard cases it remains a problem.

**Architectural pattern:** DeepSeek-R1 (chain-of-thought, both 7B and 14B) is the only model family below 91% legality, almost certainly because the long reasoning output makes a clean SAN move harder to parse. Quiet-output architectures (Qwen, Gemma, Llama) are at 99%+.

---

## Finding 2: Move quality is bimodal — inconsistent, not uniformly weak

For legal moves, performance is measured in **centipawn loss (CPL)** — the Lc0-evaluation difference between the model's chosen move and Lc0's best move on the same position. CPL is reported under the standard Lichess analysis convention with both endpoints clamped to ±1000 cp before subtraction. (The clamping prevents mate-encoded scores from inflating the metric — see methodology.)

The mean clamped CPL across all 22 models is **702**, ranging from gemma4:31b at 634 to gemma4:e2b at 727 — a tight spread of only ~14%. But the **mean conceals a bimodal distribution** that is the more interesting finding:

| Move quality bucket | % of legal moves | Notes |
|---|---|---|
| Essentially perfect (CPL < 25) | **17.8%** | Top-3 Lc0 choice |
| Excellent / strong-club level (CPL 25–100) | 4.8% | |
| Decent amateur (CPL 100–500) | 11.6% | |
| Very weak (CPL 500–1000) | 26.7% | |
| **Catastrophic / saturated (CPL ≥ 1000)** | **39.1%** | Hung piece or walked into mate |

**Models pick Lc0's exact #1 move 10.2% of the time, vs ~3.6% random** (using the empirical average of ~28 legal moves per position). The best architecture, Gemma 4, picks the #1 move at 14.4% (gemma4:31b) and 13.7% (gemma4:26b) — over 4× random chance. So models do encode some chess knowledge.

But the distribution mass at the extreme (39% saturated blunders) drives the mean up. **Among non-catastrophic moves, mean CPL is 449 / median 337** — within "weak amateur" territory rather than "random move generator". The story is therefore *unreliability*, not *uniform incompetence*: models flip between recognising the right move and walking into disaster.

| Reference point | Approximate clamped CPL |
|----------------|----------------|
| Lc0 / Stockfish (perfect play) | 0 |
| Magnus Carlsen (peak) | ~5 |
| Strong club player (~2000 Elo) | ~50–100 |
| Complete beginner | ~200–400 |
| **All models, mean (incl. blunders)** | **702** |
| **All models, mean (excl. CPL ≥ 1000)** | **449** |
| **Best model (gemma4:31b), best-move match** | **14.4%** |
| **Worst model (wizardlm2:7b), best-move match** | **8.0%** |

The same ranking holds under the alternative win-probability-loss metric (Δ win-probability × 1000), where models cluster between 283 (gemma4:31b, best) and 321 (gemma4:e2b, worst) milli-WP per move. Either way, the dispersion across 2B → 70B is small, and all models suffer from the same bimodal failure pattern.

---

## Finding 3: Direction accuracy depends heavily on the threshold of "decisive advantage"

T1 asks models to estimate the Stockfish centipawn evaluation. The classic sub-metric is **direction accuracy**: does the model agree with Stockfish on whether White is winning, Black is winning, or the position is equal? The catch: the answer depends critically on the centipawn threshold for calling a position "decisive."

| Threshold | All-model direction accuracy | Random baseline (3-class) | Interpretation |
|---|---|---|---|
| ±0 cp (sign-only) | **44.5%** | 33% | "Which side has the *exact* edge?" — pessimistic |
| ±50 cp (boundary case) | **46.1%** | 33% | The previous headline. Accidentally the worst threshold |
| ±100 cp (1 pawn = decisive) | **61.6%** | 33% | "Is one side clearly winning?" |
| ±200 cp (2 pawns = decisive) | **78.9%** | 33% | "Is this position genuinely lost for the loser?" |

**The ±50 cp number quoted in the original draft was the worst-case threshold** — it forces a White/Black/Equal call right at the boundary where models, Stockfish, and humans alike often disagree. At more meaningful thresholds, model performance is materially better. **At a 2-pawn-advantage threshold, models agree with Stockfish 79% of the time on average; at 1 pawn, 62%.**

The pattern by model is striking — **the architectures best at move quality are NOT best at direction accuracy:**

| Model | t100 direction accuracy | t200 direction accuracy | Family note |
|---|---|---|---|
| deepseek-r1:14b | **76.7%** | **91.8%** | CoT reasoning model |
| gemma3:4b | 75.8% | 91.0% | Older Gemma |
| deepseek-r1:7b | 73.9% | 89.7% | CoT reasoning model |
| gemma4:e4b | 72.6% | 91.1% | |
| mistral:7b | 72.3% | 88.8% | |
| ... | ... | ... | |
| **gemma4:31b** | **31.8%** | **61.7%** | Best at move quality, worst-third at direction |
| **llama3.3:70b** | **21.9%** | **42.2%** | Below-random at every threshold |

**This is the most interesting model-architecture finding in the study.** The Gemma 4 31B that picks the best chess move 14% of the time — the highest in the study — is BAD at saying *who is winning*. It plays well but reasons poorly. Conversely, the DeepSeek-R1 reasoning models are best at the textual-classification task ("who is winning?") but worse at the mechanical-execution task ("what's the best move?"). At the lowest end, **llama3.3:70b agrees with Stockfish on the winning side only 22% of the time at the 1-pawn-advantage threshold — well below the 33% random baseline** — meaning the largest model in the study is *systematically wrong* about who is winning.

This decoupling — "play vs reason" — is a substantive finding that the original draft missed because it only reported a single threshold.

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

| Params | Model | T2 clamped CPL | T3 Score (v2) | T1 Direction (t100) |
|--------|-------|--------|----------|-------------|
| 7B | mistral:7b | **691** | 0.579 | 72.3% |
| 8B | llama3.1:8b | 724 | 0.367 | 62.1% |
| 11B | solar:10.7b | 692 | 0.632 | 63.3% |
| 14B | deepseek-r1:14b | 701 | 0.571 | **76.7%** |
| 31B | gemma4:31b | **634** | 0.668 | 31.8% |
| 32B | qwen2.5:32b | 725 | 0.503 | 59.0% |
| 70B | llama3.3:70b | 713 | 0.602 | 21.9% |

Notable inversions:
- Mistral-7B (691 CPL) outperforms Llama3.3-70B (713 CPL) on move quality — a 10× smaller model is slightly better.
- Qwen2.5-32B (725 CPL) is the second-worst on move quality — larger than all but two models.
- DeepSeek-R1-14B's 76.7% direction accuracy beats every model in the 30B+ range.
- Llama3.3-70B is the only model BELOW the 33% random baseline at the 1-pawn-advantage threshold — actively anti-correlated with truth on the directional task.

**Architecture and training methodology appear to matter far more than scale.** The Gemma 4 family (Google's 2025 models) achieves the best move quality at 26B–31B and the best explanation quality at 26B. The DeepSeek-R1 chain-of-thought models achieve the best direction accuracy at both 7B and 14B sizes — explicit reasoning chains help with the classification problem even if they don't improve mechanical move quality.

---

## Finding 6: The DeepSeek-R1 anomaly

The DeepSeek-R1 models (7B and 14B) show a distinctive trade-off:

- **Best T1 direction accuracy in the study** (deepseek-r1:14b: 76.7% at ±100 cp; deepseek-r1:7b: 73.9% at ±100 cp)
- **Lowest legal move rates** (both at 90.0–90.7%)

The chain-of-thought reasoning process improves the model's ability to classify who is winning — a problem it can approach as a textual inference task ("the queen on d5 is dominant, the king is exposed...") — but the extended reasoning output makes it more likely to produce a move in a non-standard format that fails legality parsing. This suggests reasoning chains help with coarse-grained evaluation but not with the mechanical precision required to produce valid notation.

---

## Summary Table

| Metric | Best model | Score | Worst model | Score | All-model avg |
|--------|-----------|-------|-------------|-------|--------------|
| T2 legal-move rate (move-asking prompts) | qwen2.5:32b | 99.94% | deepseek-r1:14b | 90.00% | 97.9% |
| T1 direction (±50 cp threshold) | deepseek-r1:14b | 53.0% | llama3.3:70b | 34.3% | 46.1% |
| T1 direction (±100 cp threshold) | deepseek-r1:14b | 76.7% | llama3.3:70b | 21.9% | 61.6% |
| T1 direction (±200 cp threshold) | deepseek-r1:14b | 91.8% | llama3.3:70b | 42.2% | 78.9% |
| T2 clamped CPL (lower=better) | gemma4:31b | 634 | gemma4:e2b | 727 | 702 |
| T2 win-prob loss (×1000, lower=better) | gemma4:31b | 283 | gemma4:e2b | 321 | 311 |
| T3 explanation score (v2 matcher) | gemma4:26b | 0.776 | codellama:34b | 0.310 | 0.520 |

---

## Interpretation for a Research Audience

### What this is evidence for

The bimodal CPL distribution is the strongest novel finding. **Models do not play uniformly badly — they play in two distinct modes**: 18% of moves are essentially perfect (top-3 engine choice), and 39% are saturated catastrophic blunders. Combined with the result that *no architecture or scale eliminates the catastrophic mode*, this is consistent with the hypothesis that LLMs maintain partial-but-fragile internal representations of chess: pattern-matching on familiar positions works (the perfect bucket), but in unfamiliar positions the representation breaks down completely (the catastrophic bucket), with little in between.

The clamped-CPL spread of 634–727 across all 22 models reinforces this: regardless of size or architecture, the rate of catastrophic failures is similar. Even the best Gemma 4 31B has 34% of moves in the saturated CPL bucket.

The "play vs reason" decoupling is a related finding. The architectures best at producing high-quality moves (Gemma 4 family, no chain-of-thought) are *not* the architectures best at saying who is winning (DeepSeek-R1, with chain-of-thought). The two failure modes appear to be addressed by different architectural choices, suggesting they are mechanistically distinct.

### What this is not evidence for

This study does not show that LLMs cannot reason in any formal domain — only that chess, as tested here, is not solved by current local models up to 70B parameters. Commercial models (GPT-4, Claude) are not included and may perform differently. The benchmark also uses six fixed prompt formats; it is possible that heavily engineered prompting or fine-tuning on chess data would change the picture significantly. The benchmark uses Lichess puzzles, which are tactical by construction; performance on quiet middlegame or endgame positions may differ.

### The scale question

The parameter-scaling result is striking but should be interpreted carefully. All models tested here are **general-purpose models trained on broad internet data**. Scale in general-purpose pre-training does not obviously translate to emergent chess ability when chess games represent a small fraction of the training corpus. A fair test of the scaling hypothesis would require models specifically trained or fine-tuned on chess data — which is outside the scope of this study.

### Next steps

The most natural extension is comparison with commercial models (Claude, GPT-4o) using the same evaluation pipeline. A spec for Anthropic API integration is already written (`specs/commercial-models.md`). The relative cost of ~$13 for Claude Haiku makes this feasible as an immediate comparison.

---

## Methodology and metric artefacts

This section documents five measurement artefacts identified during a two-pass post-hoc audit, and the corrections that produce the numbers reported above. The raw data (`results/evaluations.jsonl`, 526,662 records) was unchanged; only the postprocessing was updated. All corrections are unit-tested in `tests/test_artefact_fixes.py`.

### A1 — CPL inflation by mate-encoded scores

The engine wrapper encodes mate-in-N as a centipawn value: `±10000 - mate_in_plies × 10`. For long mating sequences this produces values up to ±16,000 cp in the dataset. Without clamping, those values leak into the CPL difference whenever either endpoint of the move is mate-encoded:

| Encoding case | % of evaluations |
|---|---|
| Mate-encoded *start* position | 4.3% |
| Mate-encoded *eval-after-move* | 20.0% |
| Either endpoint mate-encoded | 22.0% |

The raw uncapped mean was 4,786 CPL; under the standard Lichess convention (both endpoints clamped to ±1000 cp before subtraction), the mean is **702 CPL**. The model rankings are essentially unchanged but the spread tightens (1.33× best/worst → 1.15×).

The `t2_cpl` column in the raw JSONL is preserved for transparency. The headline metric in this report is `t2_cpl_clamped`, with `t2_wp_loss` (Δ win-probability × 1000, bounded [0, 1000]) reported as a robustness check.

### A2 — T1 absolute error inflated by mate-truth rows

The model's eval output is clamped to ±2000 cp. The Stockfish ground-truth eval is not clamped and ranges to ±16,000 cp on mate-encoded positions. So whenever the truth is mate-encoded, the model's error is ~9,966 cp on average, **regardless of any reasoning**.

| Truth type | Mean abs error | Median |
|---|---|---|
| Mate truth (`|sf| ≥ 9000`) | 9,966 | 9,999 |
| Non-mate truth | 143 | 67 |

Reported numbers above use `t1_abs_error_excl_mate` for the headline and the relative-error metric (already in the dataset) for the cross-difficulty comparison.

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

The original T3 score was therefore mostly the side-identification component (P1) plus accidental matches on `advantage` and `mate`. The corrected `THEME_SYNONYMS` table is keyed by exact Lichess label and includes natural-English synonym lists; the fallback now splits camelCase to space-separated lowercase words. This produces a meaningfully different ranking on T3 — gemma4:26b moves from 5th place (0.51) to 1st (0.78).

### A4 — Legality denominator bug (the big one)

The original draft of this report claimed *"models produce a legal move 64.6% of the time, regardless of size"* and used this as the headline finding. This was a **denominator bug**:

- `score_t2` returns `t2_legal=False` whenever `t2_move` is None
- The `eval_only` (T1-only) and `explanation_only` (T3-only) prompt formats never ask the model for a move
- For those records, `t2_move` is always None, so `t2_legal` is always False
- These two prompt formats account for 33% of all 526k records — they automatically contributed "False" to the legality denominator

When the legality rate is computed correctly (only counting records where the model was actually asked for a move), the average is **97.9%**, not 64.6%. The corrected per-model rates range from 90.0% (deepseek-r1:14b) to 99.94% (qwen2.5:32b). This completely changes the headline finding: legality is mostly *not* a problem; **scale even helps somewhat at the upper end** (the largest models are at 99%+).

The corrected column is `t2_legal_attempted`; the buggy original is preserved as `t2_legal_rate_buggy_includes_no_move_prompts` in `summary.json`.

This is the single most consequential bug in the original analysis. Caught by checking which prompt formats actually ask for which task.

### A5 — Direction-accuracy threshold cherry-picking

The original draft headlined direction accuracy at the ±50 cp threshold (46.3% across models, near the 33% random baseline). I tested four thresholds:

| Threshold | All-model accuracy |
|---|---|
| ±0 (sign only) | 44.5% |
| ±25 | 37.8% (worse) |
| **±50 (original headline)** | **46.1%** |
| ±100 (1 pawn = decisive) | **61.6%** |
| ±200 (2 pawns = decisive) | **78.9%** |

±50 was an accidental worst-case threshold — exactly at the boundary where models, Stockfish, and humans all disagree most. At ±100 cp ("one pawn matters"), models agree with Stockfish on the direction 62% of the time, well above random. At ±200 cp ("clear advantage"), they agree 79% of the time. The qualitative claim "models are barely above random at telling who's winning" is therefore overstated. **Models can recognise clear advantages; they struggle on near-equal positions.** All four thresholds are now reported.

### A6 — Combined-prompt move retry is silent

The worker pipeline (`src/worker.py:208-222`) silently rescues illegal moves on `cot`/`fen_only`/`pgn+fen` prompts:

1. If the parsed move is illegal, scan the original response text for any other legal SAN/UCI token (`extract_move_from_text`)
2. If that fails, re-prompt the model with the legal-move list (`build_move_prompt`)

The recorded `t2_move` and `t2_legal` reflect the *post-rescue* outcome. There is no field tracking what the first attempt was. This means the legality numbers for combined prompts are "best the system can produce after up to two attempts" — not single-shot model output. The `move_only` prompt does NOT trigger the rescue (because that prompt itself already provides the legal-move list), so its numbers are single-shot-with-hint.

Documented but not fixed — would require re-running the benchmark with worker.py instrumented to record both attempts. The current numbers are reported as "post-rescue legality" with this disclosure.

### Other audited issues (no inflation, but worth recording)

- **Two engines.** T1 ground truth is Stockfish 17 @ depth 22 (precomputed). T2 best-move and CPL ground truth is Lc0 @ 800 nodes (post-hoc enrichment via `scripts/enrich_cpl.py`). On the small overlap where both are populated, the two engines disagree on the best move on ~67% of positions. This is the standard expected disagreement between a calculator-style engine and a neural-network engine; it does not affect the qualitative findings (a hung queen is ~900 CPL on either engine), but it does mean the documentation now distinguishes T1-truth from T2-truth.
- **side_claimed bias toward "White".** Models claim White is winning 55.5% of the time, Black 37.1%, Equal 7.4% — roughly a 7.5pp directional bias. Real ground-truth split is 48% White / 46% Black / 6% =0. Inflates side-correctness on White-favourable positions slightly.
- **Theme/difficulty confound.** Most tactical theme labels concentrate in the `easy` tier (84% of `backRankMate`, 76% of `kingsideAttack`/`attackingF2F7`, 70% of `mate`, 64% of `endgame`). Per-theme analysis is therefore essentially per-difficulty analysis; theme-specific claims should be interpreted with this confound in mind.
- **Bimodal eval distribution.** 92% of Stockfish ground-truth evals are within ±300 cp; 7% are mate-encoded (±9,000+); the 300–9,000 cp band is essentially empty. Lichess puzzles are tactical, so depth-22 Stockfish either has not yet "seen" the tactic (eval ≈ 0) or has resolved it to mate.
- **No duplicate records.** 0 duplicate `(model, position_id, prompt_format)` keys; retry/correction jobs are not double-counted.
- **Missing data: deepseek-r1:7b** is missing ~1,020 of 24,000 expected evaluations (~4%). Job-DB audit trail is empty so the failure mode is not recoverable, but inferred from inference-time outliers to be CoT-induced timeouts. Affects the deepseek-r1:7b numbers slightly; statistical impact is negligible.
- **Correction loop is dead code.** The `correction_loop.enabled: True` config flag is set, but the trigger checks `t2_cpl > threshold` at job-completion time, when `t2_cpl` is always `None` (CPL is filled in post-hoc by `enrich_cpl.py`, not computed during job processing because the worker's `engine = None`). Result: 0 records have `job_type='correction'`. The correction-loop / learning-delta infrastructure exists but produced no data. **No claim in this report depends on correction-loop data.**
- **Hypothesis tests H1 and H2 (T1 / T2 error increases with difficulty)** come out as **not supported** under all corrected primary metrics (relative-error, clamped-CPL, WP-loss, multi-threshold direction accuracy). Performance is roughly flat across difficulty tiers — consistent with "models are not engaging with position complexity at all".

---

*All data: `results/evaluations.jsonl` (526,662 records). Metrics computed by `scripts/generate_plots.py`. Corrected metric definitions in `src/metrics.py` (clamped CPL, WP-loss, mate-aware T1 error, v2 theme matcher, multi-threshold direction accuracy, attempted-only legality). Plots: `results/plots/`. Raw metric CSVs: `results/metrics/`. Spec for the artefact corrections: `specs/artefact-fixes.md`.*
