# Spec: H3, H4 and the Family Table Use the Corrected Columns (T3-6)

**Checklist item:** T3-6 (D4) in [exhibition-readiness.md](exhibition-readiness.md)
**Created:** 2026-10-24

---

## Goal

H3, H4 and `by_model_family.csv` still read pre-audit columns:

- **T3:** the substring-matched `t3_score`. It credits "mate" in "material" (T3-1).
- **T2:** raw `t2_cpl`. It is dominated by mate-score encoding (D5).
- **Legality:** `t2_legal` over every row, which counts prompts that never ask for a move as illegal.
  This is the 64.6% figure.
- **T1:** raw absolute error. It rewards guessing near 0 (Issue 14).

Each should use the headline measure set by the audit, falling back to the legacy column only when the
derived column is missing (hand-built frames, old files).

| Task | Headline column | Legacy fallback |
|---|---|---|
| T1 | Spearman ρ per group (higher = better) | none: NaN if no valid T1 rows |
| T2 quality | blunder rate: share of `t2_cpl_clamped` ≥ 1000 (lower = better) | mean `t2_cpl` |
| T2 legality | `t2_legal_attempted` | `t2_legal` |
| T3 | `t3_score_v2` | `t3_score` |

## Inputs / Outputs

- **H3** ("T3 decreases with difficulty") uses the T3 headline column and records the column name in
  `metric`.
- **H4** ("larger models perform better within family"):
  - For each family with at least two sizes, compare the largest with the smallest on T1 ρ (higher),
    T2 blunder rate (lower) and T3 (higher).
  - The output records the metric names, the values per size, and `better_t1/t2/t3/all_better`, as
    now.
- **`aggregate_by_model_family(df)`:** one row per (family, size), with the columns
  - `model_family`, `model_size_b`
  - `t1_spearman_rho`
  - `t1_abs_error_excl_mate` (where available)
  - `t2_blunder_rate`, `t2_cpl_clamped_mean`
  - `t2_legal_rate` (from `t2_legal_attempted`)
  - `t3_score_v2_mean`
  - `n_records`

  The legacy fallbacks keep the same output names, and a `columns_used` dict in the H4 output says
  which source was used.

## Steps / Logic

1. `_headline_columns(df)` picks the T2 and T3 sources once.
2. ρ per group reuses `_spearman` on `_valid_t1_rows`.
3. The blunder rate reuses `BLUNDER_CPL_CP`.
4. The merge guard in the family table (it raises if a row would merge two models) is kept.

## Edge cases

| Case | Behaviour |
|---|---|
| A family with one size | Skipped in H4, as now |
| No family has two sizes | H4 `supported` is False (nothing was shown), not vacuously True |
| A group with no valid T1 rows | ρ is NaN; H4 `better_t1` is False (it can't be shown) |
| No `t2_cpl_clamped` | The blunder rate falls back to the legacy mean CPL, and `columns_used` says so |
| An empty frame | Empty family table; H3/H4 as now |

## Dependencies

pandas, numpy.

## Verification (`tests/test_corrected_hypotheses.py`)

- H3 follows `t3_score_v2` when present, even when `t3_score` shows the opposite trend
- H4 uses ρ, blunder rate and v2 T3, and reports `columns_used`
- The family table has the corrected columns and no column built from raw `t2_legal`
- Without derived columns, it falls back to the legacy sources
- The merge guard still raises
