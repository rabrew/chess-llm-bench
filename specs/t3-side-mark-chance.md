# Spec: Chance-Corrected T3 Side Mark (T3-5)

**Checklist item:** T3-5 in [exhibition-readiness.md](exhibition-readiness.md)
**Builds on:** [direction-baseline-kappa.md](direction-baseline-kappa.md)
**Created:** 2026-10-24

---

## Goal

T3 part 1 gives a mark when the side the model names ("White", "Black" or "Equal") matches the
engine's eval at ±50 cp. Raw accuracy has no chance baseline:
- Most positions are not near level, and models almost never say "Equal" (1,128 of 78,170 claims in
  the base re-run).
- So "always name the side that is usually better" could score well with no reading at all.

This adds the same treatment T1-1 gave direction accuracy: the majority-class baseline, Cohen's κ and
balanced accuracy. That makes a side mark above chance visible, and one at chance clearly marked.

## Inputs / Outputs

**Input:** a `load_results_df()` DataFrame with `model`, `t3_side_claimed` and `t1_stockfish_eval`.

**Rows:** `t3_side_claimed` is one of `White` / `Black` / `Equal`, and `t1_stockfish_eval` is not
NaN.

**`compute_side_mark_agreement(df)`:** one row per model:

| Column | Meaning |
|---|---|
| `model`, `n` | |
| `accuracy` | share of claims matching the truth label |
| `majority_label`, `majority_share` | the most common truth label on these rows and its share (the "always say X" score) |
| `cohen_kappa` | κ between claim and truth labels |
| `balanced_accuracy` | mean per-class recall over the truth classes present |
| `equal_claim_share`, `equal_truth_share` | how often the model says Equal vs how often it is Equal |

**Outputs:**
- `save_metrics()` writes `t3_side_mark_agreement.csv`.
- `summary.json["overall_metrics"]["t3_side_mark"]` holds the pooled
  `{n, accuracy, majority_share, cohen_kappa}`, or `None`.

## Steps / Logic

1. Truth labels come from `_direction_labels(t1_stockfish_eval, SIDE_MARK_THRESHOLD_CP)` with
   threshold 50 (the threshold `score_t3` uses through `get_direction`). Claims map White → W,
   Black → B, Equal → E.
2. κ and balanced accuracy reuse `cohen_kappa` and `balanced_accuracy` from `src/metrics.py`.

## Edge cases

| Case | Behaviour |
|---|---|
| A claim that is None or another string | Row excluded |
| A model with no qualifying rows | Left out |
| All truth labels the same class | κ follows `cohen_kappa` (0 or NaN as defined there); the majority share is 1 |
| No qualifying rows | Empty table; summary `None` |

## Dependencies

numpy, pandas.

## Verification (`tests/test_t3_side_mark.py`)

- A perfect claimer: accuracy 1, κ 1
- "Always White" on a mix: accuracy equals the White share of the truth, and κ is 0
- Truth labels use ±50: eval 50 → E, 51 → W
- Rows with a None or invalid claim are excluded
- The share columns are correct
- `save_metrics` writes the CSV and the summary key; with no rows, the key is `None`
