# Spec: T1 Zero-Answer Baseline, Honest Fine-Tune Comparison, Raw Replies

**Builds on:** [direction-baseline-kappa.md](direction-baseline-kappa.md), [finetuned-benchmark-run.md](finetuned-benchmark-run.md)
**Created:** 2026-10-22

---

## Goal

The first fine-tuned results (llama3.2:3b-board-ft, gemma3:4b-board-ft) seemed to show better T1 eval
error after fine-tuning. That was an artefact:

- Spearman ρ between model eval and Stockfish is about 0 for both base and fine-tuned models in every
  format. Their evals do not follow the position at all.
- Most dataset positions are near level: the median |Stockfish eval| on non-mate rows is about 50 cp.
  So answering a small constant scores a low error without reading the board.
- The fine-tuned models answer small numbers (median |eval| of 3–4), so they "win" on error by guessing
  closer to 0.

Direction metrics already have chance correction (κ, majority baseline, ρ). **Eval error has no
baseline**, and the fine-tune comparison script reports neither ρ nor a baseline. This spec covers
three changes:

1. **Zero-answer error baseline.** Report each model's T1 error next to the error of a model that
   always answers 0 cp, on the same rows.
2. **A comparison script that can be trusted.** Rewrite `compare_finetuning_results.py` so that it:
   - pairs records properly
   - reads the real fields
   - reports ρ, κ and error against the zero baseline for T1
   - computes T2 legality only over prompts that ask for a move
3. **Save raw model replies** in result records, so parsing and unit questions can be answered from the
   data instead of by re-querying live models.

### Defects fixed in `compare_finetuning_results.py`

| Defect | Effect |
|---|---|
| Reads `t1_correct`, which no record has | T1 accuracy is always 0 for both sides, so ΔT1 is always 0 |
| Default fine-tuned path is `finetuning_experiment/evaluations.jsonl` | The run writes `evaluations_ft.jsonl`, so the script exits with "not found" |
| No pairing, despite the docstring | 526k base records (other runs, corrections) are averaged against the fine-tuned subset |
| T2 legality is counted over every row | `eval_only` / `explanation_only` rows count as illegal |
| T0 is computed from `response` / `raw_response` with `except Exception: return None` | The error is swallowed silently. Once raw replies are stored (change 3), benchmark answers would be scored as board descriptions, giving meaningless numbers |

T0 (board reading) is already measured by `board_reading_test.py`, so it is removed from this script.

## Inputs / Outputs

### 1. Zero-answer baseline (`src/metrics.py`)

**Input:** a `load_results_df()` DataFrame with the columns `model`, `t1_model_eval` and
`t1_stockfish_eval`.

**Rows:** both evals are present and |truth| < `MATE_SCORE_THRESHOLD_CP`. This is the same mate
exclusion as `t1_abs_error_excl_mate`. A mate truth gives about 10,000 cp of error to any answer and
would swamp both sides.

**Function:** `compute_eval_error_vs_zero(df)` returns one row per model:

| Column | Meaning |
|---|---|
| `model`, `n` | |
| `mae` | mean \|model − truth\| |
| `zero_mae` | mean \|truth\|, i.e. the MAE of always answering 0 |
| `mae_ratio` | `mae / zero_mae`; < 1 beats the zero answer, ≥ 1 does not. NaN when `zero_mae == 0` |
| `median_abs_error` | median \|model − truth\| |
| `zero_median_abs_error` | median \|truth\| |

**Outputs:**
- `save_metrics()` writes `t1_error_vs_zero.csv` through `_write_or_remove`.
- `summary.json["overall_metrics"]["t1_zero_baseline"]` holds the pooled `{n, mae, zero_mae, mae_ratio}`,
  or `None` when there are no valid rows.

### 2. `fine-tuning/scripts/compare_finetuning_results.py`

**Inputs:**
- `--base-config` (default `config/config.yaml`) and `--ft-config` (default
  `config/config_finetuned.yaml`). Each results path is read from `paths.results_file`.
- Explicit `--base` / `--ft` paths override the configs.
- `--output` defaults to `results/finetuning_experiment/comparison.csv`.

**Output:** a CSV with one row per fine-tuned model that has a base partner, and the same table logged
to the console. The columns are:

- `base_model, ft_model, n_pairs`
- for each metric: `base_<m>`, `ft_<m>`, `delta_<m>` (fine-tuned − base)

| Metric `<m>` | Definition (on paired rows only) |
|---|---|
| `t1_spearman` | ρ between model eval and Stockfish eval (rows with both evals) |
| `t1_kappa_t50` | Cohen's κ for W/B/E labels at ±50 cp |
| `t1_mae` | MAE excluding mate truth |
| `t1_mae_ratio` | `t1_mae / zero_mae` on that side's own valid rows |
| `t2_legal_rate` | share of move-asking rows (`fen_only, pgn+fen, cot, move_only`) whose final `t2_move` is legal |
| `t2_mean_cpl_clamped` | mean CPL capped at 1000, over rows with a CPL |
| `t3_mean` | mean `t3_score` over rows with one |

`zero_mae` is also written once per row. It is the zero-answer MAE on the rows where **both** sides gave
an eval, so the reader has one common reference figure.

### 3. Raw replies (`src/result_writer.py`, `src/worker.py`)

`build_result_record(..., raw_responses=None)` copies each non-`None` entry of `raw_responses` into the
record. The worker passes:
- `raw_response`: the text of the main call for every format.
- `raw_move_retry_response`: the text of the isolated move call. This is only present when the
  combined-format retry actually made that call.

Existing records are unchanged, and readers must treat both fields as optional.

## Steps / Logic

**Zero baseline.**
1. Take the rows from `_valid_t1_rows(df)`.
2. Drop mate-truth rows.
3. Group by model.
4. Compute the columns above with plain numpy.

**Comparison.**
1. Load both JSONL files with `load_results_df` so the derived columns (clamping, `t2_cpl_clamped`)
   match the main metrics.
2. Keep only `job_type == "standard"` rows. Correction and control jobs are separate experiments.
3. Map each fine-tuned tag to its base with `removesuffix("-board-ft")`.
4. Inner-join on `(base model, position_id, prompt_format)`.
5. If a key appears more than once on either side, keep the first occurrence and log a warning with the
   duplicate count.
6. For each pair set, compute each side's metrics. Reuse `compute_eval_rank_correlation`,
   `compute_direction_agreement` and `compute_eval_error_vs_zero` from `src/metrics.py`, so that
   nothing is defined twice.

**Raw replies.** The worker passes `llm_result["response"]` (and the retry's response) into
`build_result_record`.

## Edge cases

| Case | Behaviour |
|---|---|
| Constant model eval | ρ = NaN, κ = 0, and `mae_ratio` shows whether it beats 0 cp |
| `zero_mae == 0` (all truths 0) | `mae_ratio` = NaN, not a divide error |
| A model with no valid T1 rows | Left out of the zero-baseline table, with a logged warning (same as direction agreement) |
| A fine-tuned model with no base records, or no overlapping keys | Skipped, with a logged warning. If no model pairs at all, exit 1 |
| A metric that is undefined on one side (e.g. no move-asking rows) | `base_`/`ft_` is NaN and `delta_` is NaN; it is never reported as 0 |
| A results file is missing or a config has no `paths.results_file` | Log an error naming the path and exit 1 |
| A raw reply is an empty string | Stored as `""`; only `None` is left out |
| Combined format, move rescued from the first reply | No retry call is made, so `raw_move_retry_response` is absent |

## Rollout constraint

`bin/run_all.sh` starts a new `run_workers.py` process for every tier, and that process imports
`src/worker.py` from disk. Merging change 3 while a run is in progress would switch the record schema
partway through the run. So the work is done on branch `t1-baselines` in a separate worktree, and it is
merged into the working checkout **only after the current fine-tuned run has finished**. Changes 1 and
2 are read-side only and can be run against existing results straight from the worktree.

## Storage

The fine-tuned run produces 120k records at about 0.3–1.6 KB of reply text each (`num_predict` is 400).
That is at most about 200 MB, which is acceptable. Note that the root partition was 98% full on
2026-10-22.

## Dependencies

numpy, pandas and pyyaml, which are already in `requirements.txt`. No scipy (ρ uses average ranks, as
in the κ spec).

## Verification

**`tests/test_t1_zero_baseline.py`:**
- A constant-0 predictor has `mae_ratio == 1` exactly
- A perfect predictor has `mae == 0` and `mae_ratio == 0`
- Mate-truth rows are excluded from `n` and from both MAEs
- `zero_mae == 0` gives a `mae_ratio` of NaN
- One row per model; a model with no valid rows is left out
- `save_metrics` writes `t1_error_vs_zero.csv`, and `summary.json` has `t1_zero_baseline`

**`fine-tuning/tests/test_compare_finetuning.py`:**
- Only keys present on both sides are compared, and `n_pairs` is correct
- Correction and control rows are ignored
- Duplicate keys keep the first record
- `t2_legal_rate` ignores `eval_only` / `explanation_only` rows
- A fine-tuned model that only shrinks its evals toward 0 shows a lower `t1_mae` but `delta_t1_spearman ≈ 0`.
  This is the regression test for the artefact above.
- `t1_spearman` matches `compute_eval_rank_correlation`
- An undefined metric gives NaN deltas, never 0
- Results paths come from the configs, and an explicit path overrides the config
- A missing file or no overlapping pairs exits with status 1

**`tests/test_result_writer.py` and `tests/test_worker.py`:**
- `raw_responses` entries are copied into the record; `None` entries are not, and an empty string is kept
- Without `raw_responses`, the record has no raw fields
- The worker stores `raw_response` for each of the six formats
- The combined format with a retry stores `raw_move_retry_response`; a rescue without a retry does not
