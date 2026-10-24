# Spec: Spearman ρ as the Headline T1 Number, and Plot Metadata From the Registry (T1-4)

**Checklist item:** T1-4 (D2) in [exhibition-readiness.md](exhibition-readiness.md), plus the M-4 follow-up in plots
**Created:** 2026-10-24

---

## Goal

1. **Headline T1 = ρ.** Direction accuracy depends on an arbitrary ±50 cp threshold and rewards
   guessing near 0 (Issue 14). Spearman ρ needs no threshold, and ρ ≈ 0 means "the eval does not follow
   the position", which is the honest T1 result. The by-model table, the summary and every T1 plot
   panel should lead with ρ. Direction accuracy at four thresholds stays in its own CSVs, as robustness
   analysis.
2. **Plot metadata.** `scripts/generate_plots.py` keeps its own hardcoded `MODEL_META` (family, size),
   which duplicates `MODEL_REGISTRY`. M-4 fixed the registry, but the plots still use the copy, so a
   fine-tuned tag (`*-board-ft`) gets family "Other" and size 0, which breaks the log-scale size axis.
   Family and size should come from `parse_model_info`. A display-name map keeps the current plot
   grouping (Mistral and Mixtral both shown under "Mistral").

## Inputs / Outputs

**`aggregate_by_model(df)`:** adds a `t1_spearman_rho` column per model, merged from
`compute_eval_rank_correlation`. It is NaN for a model with no valid T1 rows.

**`summary.json["overall_metrics"]["t1_headline"]`:**
`{"metric": "spearman_rho", "value": <pooled ρ>, "n": <rows>}`, or `None` when there are no valid T1
rows.

**`generate_plots.py`:**
- `load_model_df` sets `t1_headline_mean` to `t1_spearman_rho` when the column exists. Older metrics
  files fall back to `t1_direction_correct_mean`, and `t1_headline_label` says which is used.
- `family` and `size_b` come from `parse_model_info(model)` through `FAMILY_DISPLAY`. An unknown tag
  raises `ValueError`, matching the registry rule.
- `plot_overview_rankings`, `plot_family_scaling` and `plot_summary_heatmap` use `t1_headline_mean`,
  with a reference line at 0 labelled "No relation (ρ = 0)" and labels formatted as `ρ=0.12`. The
  overview axis allows negative values.

## Steps / Logic

- `FAMILY_DISPLAY` maps each registry family to the existing display label: `llama3` → "LLaMA 3", …,
  `mixtral` → "Mistral".
- A registry family missing from `FAMILY_DISPLAY` raises `KeyError`, which a test guards against.

## Edge cases

| Case | Behaviour |
|---|---|
| An old `by_model.csv` without `t1_spearman_rho` | Falls back to direction accuracy, with the label saying so |
| All ρ values negative | The overview axis spans from the minimum to 0 |
| A `-board-ft` tag | Same family and size as its base |
| An unknown tag | `ValueError` naming the tag |

## Dependencies

matplotlib, seaborn and pandas (already used by the plots).

## Verification (`tests/test_generate_plots.py`)

- `aggregate_by_model` has `t1_spearman_rho` and it matches `compute_eval_rank_correlation`
- The summary has `t1_headline` with metric `spearman_rho`
- `load_model_df` uses ρ as the headline when present and falls back otherwise
- Family and size come from the registry for a base tag and its `-board-ft` tag; an unknown tag raises
- Every registry family has a display name
- A smoke test: the overview, family-scaling and heatmap plots write PNGs from a small frame that
  includes negative ρ (Agg backend)
