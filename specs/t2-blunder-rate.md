# Spec: Blunder Rate and CPL Buckets as the Main T2 Quality Numbers (T2-1)

**Builds on:** [metric-defect-remediation.md](metric-defect-remediation.md) D5, [cpl-eval-before.md](cpl-eval-before.md)
**Checklist items:** T2-1, and the blunder-rate part of F-3
**Created:** 2026-10-24

---

## Goal

Clamped CPL is capped at ±1,000 cp per side of the subtraction, and a large share of legal moves sit at
or near that ceiling. The mean is therefore close to "1,000 × the blunder rate", while looking like a
continuous quality score, and the cross-model spread is tiny (D5). This spec reports instead:

1. **The blunder rate:** the share of legal moves with clamped CPL ≥ 1,000, per model, with a 95%
   bootstrap confidence interval. Models whose intervals overlap their neighbour in the ranking are
   flagged as tied.
2. **CPL buckets:** five buckets per model with counts and shares, so the "perfect or disaster" shape
   is visible directly.
3. **The mean and median clamped CPL**, kept next to these as secondary figures.

## Inputs / Outputs

**Input:** a `load_results_df()` DataFrame. The rows used are those with a non-NaN `t2_cpl_clamped`
(legal moves only, since CPL exists only for them).

**Buckets** (the same as FINDINGS Finding 2; lower bound inclusive, upper bound exclusive):

| Key | Clamped CPL |
|---|---|
| `perfect` | < 25 |
| `strong` | 25–100 |
| `decent` | 100–500 |
| `very_weak` | 500–1000 |
| `blunder` | ≥ 1000 |

**`compute_cpl_buckets(df)`:** one row per model: `model`, `n`, then `<bucket>_count` and
`<bucket>_share` for each of the five buckets.

**`compute_blunder_rates(df)`:** one row per model, sorted by `blunder_rate` ascending (best first):

| Column | Meaning |
|---|---|
| `model`, `n` | |
| `blunder_rate` | share with clamped CPL ≥ 1000 |
| `ci_low`, `ci_high` | 95% bootstrap interval for `blunder_rate` |
| `mean_cpl_clamped`, `median_cpl_clamped` | |
| `rank_tied` | True if this model's interval overlaps the interval of the model ranked directly above or below it |

**Outputs:**
- `save_metrics()` writes `cpl_buckets.csv` and `t2_blunder_rate.csv` through `_write_or_remove`.
- `summary.json["overall_metrics"]["t2_blunder_rate"]` holds `{n, rate, ci_low, ci_high}` pooled over
  all models, or `None` when there are no CPL rows.
- `compare_finetuning_results.py` adds the metric `t2_blunder_rate`, with `base_`/`ft_`/`delta_` like
  the rest.

## Steps / Logic

- **Bootstrap.** A blunder is a 0/1 value, so a nonparametric bootstrap of its mean is the same as
  drawing `Binomial(n, p̂) / n`. That form is used: it is exact and fast enough for about 77k rows ×
  22 models.
- **Settings.** `BOOTSTRAP_RESAMPLES = 10_000`, `BOOTSTRAP_SEED = 42`, and `CI_LEVEL = 0.95` (percentile
  interval). A fresh `np.random.default_rng(seed)` is used per model, so each model's interval doesn't
  depend on which other models are in the frame.
- **Ties.** After sorting, compare each model's interval with its neighbours; two intervals overlap
  when `a.ci_low ≤ b.ci_high` and `b.ci_low ≤ a.ci_high`.

## Edge cases

| Case | Behaviour |
|---|---|
| A model with no CPL rows | Left out of both tables |
| `n == 1` | Rate is 0 or 1, the interval collapses to that point, and `rank_tied` follows the rule |
| p̂ = 0 or 1 | A zero-width interval (a known property of the percentile bootstrap; noted in the write-up) |
| A single model | `rank_tied = False` |
| Clamped CPL exactly 1000 | Counts as a blunder (≥) |
| No CPL rows at all | Empty tables (any stale CSV is removed); summary key `None` |

## Dependencies

numpy and pandas only.

## Reproducibility

The seed is fixed, so the intervals reproduce exactly.

## Verification (`tests/test_t2_blunder_rate.py`)

- Bucket edges: 24.9 → `perfect`, 25 → `strong`, 999.9 → `very_weak`, 1000 → `blunder`
- Bucket counts sum to `n`; shares sum to 1
- `test_blunder_rate_matches_bucket_counts`: the rate equals the `blunder` share
- The interval contains p̂, is within [0, 1], and is identical across two calls
- Two models with far-apart rates → `rank_tied` False; two with the same data → True
- A model with no CPL rows is left out
- `save_metrics` writes both CSVs and `summary.json` has `t2_blunder_rate`; with no CPL rows, the key
  is `None` and no CSVs are written
- The comparison reports `base_`/`ft_`/`delta_t2_blunder_rate`
