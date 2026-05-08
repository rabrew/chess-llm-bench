# Spec: Chess LLM Benchmark — Overview

This is the **short overview** of the benchmark. For the full specification (parameters, edge cases, scoring details, hypothesis tests), see [benchmark-detailed.md](benchmark-detailed.md).

## Goal

Benchmark **22 open-source local LLMs** (2B–70B parameters, 12 architecture families) on chess move quality, evaluation, and explanation across **4,000 stratified Lichess puzzle positions** (1,000 each at four difficulty tiers). Primary research question: *"Can current general-purpose LLMs reason about a fully formal closed domain (chess) where ground truth is computable, and does scaling help?"*

## Inputs / Outputs

**Inputs:**
- `data/{easy,medium,hard,extreme}.json` — 1,000 Lichess puzzle positions per tier (4,000 total). Each carries a precomputed Stockfish-17 evaluation at depth 22. The full Lichess puzzle CSV (~5.8M positions) is sampled down to 1,000 per tier in `dataset_builder.py`.
- `config/config.yaml` — models, prompt formats, paths, thresholds, engine settings
- Ollama instance with all 22 models pulled and running

**Outputs:**
- `results/evaluations.jsonl` — one record per (model × position × prompt format) job. ~526k records.
- `results/evaluations_retried.jsonl` — output of `scripts/retry_illegal_moves.py` on the residual illegal-move rows
- `results/plots/*.png` — per-model and per-family visualisations
- `results/metrics/{by_model,by_difficulty,...}.csv` and `summary.json` — aggregated metrics

## Steps / Logic

1. **Build dataset** (`scripts/build_dataset.py`) — sample 1,000 puzzles per difficulty tier from the full Lichess CSV, validate, assign stable hash-based IDs.
2. **Stockfish ground truth** (`scripts/precompute_stockfish.py`) — depth-22 Stockfish-17 evaluations stored alongside each position. Used as T1 truth.
3. **Lc0 ground truth** (`scripts/enrich_cpl.py`) — Lc0 @ 800 nodes computes best-move and CPL post-hoc. Used as T2 truth. Runs after the LLM jobs complete.
4. **Pull models** (`scripts/pull_models.py`) — ensure all 22 models in `config.models` are available locally via Ollama.
5. **Generate jobs** (`scripts/generate_jobs.py`) — populate the SQLite job queue with one row per (position × model × prompt format). 4,000 × 22 × 6 = 528,000 jobs.
6. **Run workers** (`scripts/run_workers.py`) — parallel workers claim jobs, prompt the model, parse the response, score each task (T1/T2/T3), write to `evaluations.jsonl`.
7. **Generate plots & metrics** (`scripts/generate_plots.py --save-metrics`) — postprocess the JSONL into per-model and per-difficulty aggregations, compute hypothesis tests, render charts.

## Tasks scored per position

| Task | Question | Truth |
|---|---|---|
| **T1 — Eval** | What is the centipawn evaluation of this position? | Stockfish-17 @ depth 22 |
| **T2 — Move** | What is the best move? | Lc0 @ 800 nodes (post-hoc enrichment) |
| **T3 — Explanation** | Who stands better, and why? | Rule-based scorer comparing model output against the Lichess theme tag |

## Prompt Formats (six)

- `cot` — full three-task prompt with chain-of-thought scratchpad
- `fen_only` — full three-task prompt with just the FEN
- `pgn+fen` — full three-task prompt with PGN history + FEN
- `eval_only` — isolated T1 prompt
- `move_only` — isolated T2 prompt with the legal-move list shown to the model
- `explanation_only` — isolated T3 prompt

The **isolated prompts** (`eval_only`, `move_only`, `explanation_only`) exist to disentangle "the model can't do this task" from "the prompt format confused the model".

## Headline Metrics

- **T1**: direction accuracy at multiple thresholds (±0/50/100/200 cp), absolute error excl. mate-truth rows, magnitude-invariant relative error.
- **T2**: legality rate (computed only on move-asking prompts), clamped CPL (Lichess convention, ±1000 cp), win-probability loss (×1000, bounded).
- **T3**: 0/1/2 score combining side-correctness and theme-keyword match (using the v2 camelCase-aware matcher).

## What is NOT in this study

- **No commercial models.** Local Ollama only. The Anthropic API integration is specced in [commercial-models.md](commercial-models.md) but not yet run.
- **No correction-loop / learning-delta data.** The infrastructure exists (`src/feedback_loop.py`) but is dead code under the current worker because CPL is filled in post-hoc, not inline. The `correction_loop.enabled` config flag is set to `false` to reflect this.

## Reproducibility

- Random seed: `42` in `config/config.yaml` (controls puzzle sampling).
- Position IDs are SHA-256-derived from the FEN, stable across rebuilds with different `max_positions_per_tier`.
- All model versions pinned by Ollama tag.
- Engine versions and search budgets pinned in config (`stockfish.depth`, `lc0.nodes`).
- Python dependencies pinned in `requirements.txt`.

## Audit-log

This benchmark has gone through two post-hoc audit passes that fixed several measurement artefacts (mate-encoding inflation in CPL and T1, T3 theme matcher, legality denominator, direction-accuracy threshold cherry-picking). All corrections are postprocessing-only and unit-tested. See [`docs/FINDINGS.md` § Methodology and metric artefacts](../docs/FINDINGS.md) and [artefact-fixes.md](artefact-fixes.md) for the audit trail.
