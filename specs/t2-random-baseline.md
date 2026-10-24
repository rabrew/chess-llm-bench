# Spec: Best-Move Rate Against a Real Random Baseline (T2-3)

**Checklist item:** T2-3 in [exhibition-readiness.md](exhibition-readiness.md)
**Created:** 2026-10-24

---

## Goal

FINDINGS says models pick Lc0's #1 move 10.2% of the time, "vs ~3.6% random". The 3.6% is 1 ÷ 28,
using an average of 28 legal moves per position. But the chance of a uniform random legal move
being the best one is the **mean of 1 ÷ n_legal**, not 1 ÷ mean(n_legal). Because 1/x is convex,
the two differ, and positions with few legal moves (checks, endgames) pull the true figure up.

Compute the baseline from the real positions, **on the same rows as each model's own rate**, so that
the two numbers are directly comparable.

## Inputs / Outputs

**Input:** a `load_results_df()` DataFrame with `model`, `fen`, `t2_move` and `t2_best_move`.

**Rows:** move-asking rows (`MOVE_PROMPT_FORMATS`) with a non-null `t2_best_move`, which means the
move was legal and CPL-enriched.

**`compute_best_move_vs_random(df)`:** one row per model:

| Column | Meaning |
|---|---|
| `model`, `n` | |
| `best_move_rate` | share of rows where `t2_move == t2_best_move` |
| `random_best_move_rate` | mean over the same rows of 1 ÷ (number of legal moves in `fen`) |
| `mean_legal_moves` | mean number of legal moves over those rows (for context) |
| `ratio_vs_random` | `best_move_rate / random_best_move_rate` |

**Outputs:**
- `save_metrics()` writes `t2_best_move_vs_random.csv`.
- `summary.json["overall_metrics"]["t2_best_move_vs_random"]` holds the pooled
  `{n, best_move_rate, random_best_move_rate, ratio_vs_random}`, or `None`.

## Steps / Logic

1. Count legal moves per unique FEN once (`len(list(chess.Board(fen).legal_moves))`) and map the
   counts back to the rows.
2. Compare moves as SAN strings. This matches how `enrich_cpl.py` decides CPL = 0, which
   is `move == best_move`.

## Edge cases

| Case | Behaviour |
|---|---|
| A FEN with no legal moves (mate or stalemate) | Cannot have a best move; the row is dropped |
| An unparseable FEN | Dropped, with one warning giving the count |
| A model with no qualifying rows | Left out |
| No qualifying rows at all | Empty table; summary `None` |

## Dependencies

python-chess, pandas.

## Verification (`tests/test_t2_random_baseline.py`)

- A position with exactly 1 legal move gives a random rate of 1.0. A mix of positions with 20 and 1
  legal moves gives (1/20 + 1)/2, not 1/mean.
- `best_move_rate` counts exact SAN matches only
- Rows without `t2_best_move`, and rows from `eval_only`, are excluded
- A bad FEN is dropped, not raised
- The ratio equals the rate ÷ the random rate
- `save_metrics` writes the CSV and the summary key
