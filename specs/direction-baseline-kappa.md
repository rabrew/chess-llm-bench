# Spec: Chance-Corrected T1 Direction Metrics

**Closes:** D1 and D2 step 3 in [metric-defect-remediation.md](metric-defect-remediation.md)
**Created:** 2026-10-20

---

## Goal

T1 direction accuracy labels both the model eval and the engine eval as
White / Black / Equal using the **same** threshold T. As T grows, more
positions fall into "Equal", so a model that always answers 0 cp scores
well even though it knows nothing about the position. FINDINGS compares
the results against a "33% random baseline", which only holds when the
three classes are equally common. They aren't: about 92% of engine evals
fall within ±300 cp.

This spec adds metrics that cannot be gamed by a constant answer:

1. **Majority-class baseline** per threshold. This is what "always answer
   the most common label" would score.
2. **Cohen's κ** per (model, threshold). This is agreement beyond what the
   label frequencies alone would produce. A constant answer gets κ = 0.
3. **Balanced accuracy** per (model, threshold). This is the mean recall
   over the true classes. A constant answer gets 1/k, where k is the number
   of classes present in the truth.
4. **Spearman ρ** per model between model eval and engine eval. It needs no
   threshold, so it is the primary measure of "does the model rank positions
   the way the engine does".

Raw threshold accuracy stays in the outputs for transparency, but it should
never again be shown without its baseline.

## Inputs / Outputs

**Inputs:** the results DataFrame from `load_results_df()`, which uses the
columns `model`, `t1_model_eval` (already clamped to ±2000) and
`t1_stockfish_eval`. Rows where either eval is missing are excluded.

**Outputs** (written by `save_metrics()` to `results/metrics/`):

| File | Grain | Columns |
|---|---|---|
| `direction_baselines.csv` | threshold | `threshold, n, pct_W, pct_B, pct_E, majority_class, majority_baseline_acc` |
| `direction_agreement.csv` | model × threshold | `model, threshold, n, accuracy, majority_baseline_acc, lift_over_baseline, cohen_kappa, balanced_accuracy` |
| `t1_rank_correlation.csv` | model | `model, n, spearman_rho` |

`summary.json["overall_metrics"]` gains:
- `t1_direction_baselines`: `{ "t0": {...}, ... }` with the same fields as the CSV
- `t1_direction_kappa_by_threshold`: pooled κ across all models, per threshold
- `t1_spearman_rho`: pooled ρ across all models

`majority_baseline_acc` in `direction_agreement.csv` is computed on **that
model's own rows**. Models have slightly different missing rows, so each one
is compared against the baseline for exactly the positions it answered.

## Steps / Logic

1. `_direction_labels(evals, threshold)` returns `"W"` if eval > T, `"B"` if
   eval < −T, and `"E"` otherwise. `_compute_direction_correct` reuses it, so
   the labelling rule lives in one place.
2. `cohen_kappa(pred, truth)`:
   - p_o = share of rows where pred == truth
   - p_e = Σ over classes c of P(pred = c) · P(truth = c)
   - κ = (p_o − p_e) / (1 − p_e)
   - If p_e == 1 (both sides are the same single constant label), κ is
     mathematically undefined. Return NaN.
3. `balanced_accuracy(pred, truth)`: for each class present in `truth`,
   recall = share of those rows where pred matches. Return the mean of
   those recalls.
4. `compute_direction_baselines(df)`: for each threshold, take the class
   shares of the engine labels over valid rows. The majority class is the
   largest share, and the baseline accuracy is that share.
5. `compute_direction_agreement(df)`: group by model, and for each
   threshold compute the columns in the table above.
   `lift_over_baseline = accuracy − majority_baseline_acc`.
6. `compute_eval_rank_correlation(df)`: Spearman ρ per model using
   `Series.corr(method="spearman")`. This is pandas-native, so no new
   dependency is needed. Mate-encoded truth rows stay in because rank
   correlation doesn't care about magnitude.

## Edge cases

| Case | Behaviour |
|---|---|
| Rows with a missing model or engine eval | Excluded from every metric |
| Constant prediction (e.g. always 0 cp) | κ = 0; balanced accuracy = 1/k; ρ = NaN (zero variance makes it undefined) |
| Pred and truth both the same single label | κ = NaN (undefined), accuracy = 1.0 |
| Empty input to `cohen_kappa` / `balanced_accuracy` | Raise `ValueError` rather than returning a silent NaN |
| Mismatched lengths | Raise `ValueError` |
| Model with no valid rows | Left out of the per-model tables, with a logged warning |
| Ties in the majority class | Pick the first in `W, B, E` order (deterministic) |

## Dependencies

numpy and pandas only. These are already in `requirements.txt`.

## Verification (tests in `tests/test_direction_baseline.py`)

- Baseline shares sum to 1 and the majority baseline equals the largest share
- A constant-0 predictor gets κ = 0 at every threshold, and accuracy ≤ majority baseline
- A perfect predictor gets κ = 1, balanced accuracy = 1 and ρ = 1
- A perspective-flipped predictor (sign inverted) gets κ < 0 and ρ = −1
- Balanced accuracy of a constant predictor is 1/k
- `cohen_kappa` raises on empty or mismatched input and returns NaN when p_e = 1
- The agreement table has one row per (model, threshold), and each model's
  baseline is computed on its own rows
- `save_metrics` writes all three CSVs, and `summary.json` carries the new keys
