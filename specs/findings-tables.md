# Spec: Findings Tables Generated From the Metrics (W-8, P-2)

**Checklist items:** W-8 (regenerate every document from `summary.json`), P-2 ("never type doc numbers by hand")
**Created:** 2026-10-24

---

## Goal

Every number in FINDINGS was typed by hand, and the audit found several of them stale or wrong
(98% legal, 3.6% random, 702 CPL). From now on, the tables in the write-up are **generated** from the
metrics files. A check mode fails when a document no longer matches the data, so the write-up can't
silently drift from the results.

## Inputs / Outputs

**Inputs:**
- `--metrics` (default `results/metrics`): the base benchmark's `summary.json`, `t1_rank_correlation.csv`,
  `t1_error_vs_zero.csv`, `t2_blunder_rate.csv`, `t2_best_move_vs_random.csv`, `cpl_buckets.csv`,
  `puzzle_meta_metrics.csv`, `t3_side_mark_agreement.csv`, `by_model.csv`, `by_model_family.csv`
- `--rerun-metrics` (default `results/finetuning_experiment/base_rerun/metrics`): `summary.json` and
  `t2_move_source.csv` (first-answer legality exists only there)
- `--comparison` (default `results/finetuning_experiment/comparison.csv`)
- `--board-test` (default `results/finetuning_experiment/board_t0_summary.csv`)
- `--doc`, repeatable (default `docs/FINDINGS.md` and `docs/SIMPLE_SUMMARY.md`)

**Output:** each document is rewritten in place. Only the text between
`<!-- table:NAME -->` and `<!-- /table:NAME -->` markers is replaced; everything else is untouched.

**Tables (`NAME`):**

| Name | Content |
|---|---|
| `headline` | the pooled key numbers: records, models, positions, T1 ρ, T1 MAE ÷ zero-answer MAE, final legality, first-answer legality (re-run), blunder rate + CI, best-move rate vs random, setup-move rate, side mark (accuracy, majority, κ), tactical theme mark |
| `models` | one row per model, sorted by blunder rate: ρ, MAE ÷ zero, legality, blunder rate [CI], best-move ×random, setup-move rate, side-mark κ, T3 (v2) |
| `buckets` | the pooled share of legal moves in each CPL bucket |
| `first_answer` | per model, first-answer legality on combined prompts vs final legality (re-run) |
| `family` | `by_model_family.csv` corrected columns |
| `finetune` | per model, base → fine-tuned: board-reading recall and perfect-board rate, ρ, legality, blunder rate, T3 |
| `finetune_legal` | per model, base → pieces + legal moves: legal-move recall on held-out positions (criterion 0), first-answer legality (criterion 1), final legality (2), blunder rate (3) and T1 ρ (4). Inputs: `--legal-comparison` (default `results/finetuning_experiment/comparison_legal.csv`) and `--legal-board-test` (default `results/finetuning_experiment/board_legal_t0_training_prompt_summary.csv`); "not available" until both exist |
| `difficulty` | per tier, from `summary.hypothesis_tests`: T1 relative error (H1), clamped CPL (H2), T3 v2 (H3), each with its "supported" verdict |
| `formats` | per prompt format, from `by_prompt_format.csv`: T1 ρ, final legality, blunder rate, T3 v2 (W-6) |

**New metrics table (W-6):** `compute_by_prompt_format(df)` in `src/metrics.py` gives one row per
`prompt_format`, pooled over models: `n`, `t1_spearman_rho` (rows with both evals), `t2_legal_rate`
(mean `t2_legal_attempted`, NaN for formats that don't ask for a move), `t2_blunder_rate` (rows with a
clamped CPL), and `t3_score_v2_mean`. `save_metrics` writes `by_prompt_format.csv`. The test is in
`tests/test_render_findings.py`.

**`--check`:** the documents are not written. Exit 1, listing each table name that differs from the
fresh render.

## Steps / Logic

1. Load each input. A missing *required* file (the base `summary.json`) exits 1. A missing optional
   input (re-run, comparison, board test) renders that table as a one-line note, "not available".
2. Each table is a pure function from DataFrames or dicts to a Markdown string, using fixed formats:
   - rates as percentages, 1 decimal place
   - ρ and κ to 2 decimal places
   - ratios to 2 decimal places
3. Block replacement uses a regex on the exact marker pair. A document can contain any subset of
   tables. An unknown table name, or an unterminated marker, raises `ValueError` naming it.

## Edge cases

| Case | Behaviour |
|---|---|
| A marker pair with no table of that name | `ValueError` |
| An opening marker without a closing one | `ValueError` |
| NaN values | Shown as "—" |
| A document with no markers | Left unchanged (reported) |

## Dependencies

pandas.

## Verification (`tests/test_render_findings.py`)

- Replacing a block changes only the text between the markers; unknown names and unterminated markers
  raise
- The table builders produce the expected rows and formats from small fixtures, including "—" for NaN
- `--check` exits 1 on a stale document and 0 after a render
- A missing optional input produces the "not available" note; a missing summary exits 1
