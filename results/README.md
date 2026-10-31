# Results

The raw answer-by-answer records are too large for the repository. They are on the release
[`data-2026-10-31`](https://github.com/rabrew/chess-llm-bench/releases/tag/data-2026-10-31); see the
main README, section 6. Everything else is here.

## Base benchmark (22 models, 4,000 positions, 6 prompt formats)

| Path | What it holds |
|---|---|
| `metrics/summary.json` | Headline numbers, hypothesis tests and provenance (git commit, data hash, engine) |
| `metrics/by_model.csv` | Every measure per model |
| `metrics/by_model_family.csv` | Per family and size (the scaling question) |
| `metrics/by_prompt_format.csv` | Per prompt format (chain of thought, isolated prompts) |
| `metrics/by_difficulty.csv`, `by_phase.csv`, `by_source.csv` | Splits by tier, game phase and source |
| `metrics/t1_rank_correlation.csv` | T1: Spearman ρ against the engine |
| `metrics/t1_error_vs_zero.csv` | T1: error compared with always answering 0 |
| `metrics/direction_agreement.csv`, `direction_baselines.csv` | T1: "who is winning" agreement, κ and the baselines |
| `metrics/t1_by_side_to_move.csv` | T1: perspective check |
| `metrics/t2_blunder_rate.csv` | T2: blunder rate with bootstrap 95% CIs |
| `metrics/cpl_buckets.csv` | T2: the "perfect or disaster" distribution |
| `metrics/t2_best_move_vs_random.csv` | T2: best-move rate against a random legal move |
| `metrics/t2_move_source.csv` | T2: first-answer legality and how each final move was obtained |
| `metrics/t3_side_mark_agreement.csv` | T3: side mark, majority baseline and κ |
| `metrics/puzzle_meta_metrics.csv` | Setup-move rate and tactical-theme rate |
| `metrics/hallucination_rate.csv` | Legacy: share of records without a legal move, including prompts that never ask for one. Superseded by `t2_move_source.csv` (Issue 16) |
| `metrics/engine_strength.csv`, `engine_strength_summary.json` | Engine-strength check (800 vs 8,000 nodes) |
| `plots/` | Charts generated from the metrics |
| `engine_strength/` | The engine-strength check's raw engine output |
| `evaluations_retried.jsonl` | Moves obtained by re-asking after an illegal answer |
| `t3_examples.md` | Real explanation phrasings, used to choose theme synonyms |
| `logs/` | Run logs |

## Fine-tuning (`finetuning_experiment/`)

| Path | What it holds |
|---|---|
| `comparison.csv` | Base vs pieces-only fine-tune, paired on the same positions |
| `comparison_legal.csv` | Base vs pieces + legal moves fine-tune, judged against the pre-registered criteria |
| `base_rerun/` | Metrics, plots and logs for the base re-run of the 5 models |
| `metrics/`, `plots/`, `logs/` | The pieces-only fine-tune |
| `legal_ft/` | The pieces + legal moves fine-tune |
| `board_t0*.jsonl`, `board_legal_t0*.jsonl` and their `_summary.csv` | Board-reading tests before and after training |
| `ft_move_failure_probe.jsonl` | Probe of why fine-tuned models gave illegal moves |
| `evaluations_*_retried.jsonl` | Re-asked moves for each run |

## Notation check (`plain_board/`)

Running. When it finishes, `plain_board/metrics/` will hold the usual metrics plus
`plain_board_comparison.csv` and `plain_board_summary.json` (the pre-registered N1–N4 and the verdict).
