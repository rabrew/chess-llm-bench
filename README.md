# Do LLMs Reason? Benchmarking Logical Performance via Chess

**Ryan Brew · Project PR09426 · Stripe Young Scientist Award 2027**

Do open-source language models reason about a chess position, or do they pattern-match on familiar
chess text? This repository holds the benchmark, every experiment, all results and the audit trail.

| Section | What's in it |
|---|---|
| [1. The project in brief](#1-the-project-in-brief) | Question, method and headline result in one page |
| [2. Where to start reading](#2-where-to-start-reading) | A reading order for a reviewer |
| [3. The experiments](#3-the-experiments) | Each experiment with its spec, config, results and write-up |
| [4. How the benchmark works](#4-how-the-benchmark-works) | Positions, models, tasks, scoring |
| [5. Repository map](#5-repository-map) | What each folder is for |
| [6. Raw data](#6-raw-data) | The full answer-by-answer records and how to get them |
| [7. Reproducibility](#7-reproducibility) | What can and can't be reproduced |
| [8. Running it yourself](#8-running-it-yourself) | Commands for the pipeline, tests and checks |

---

## 1. The project in brief

- **What was tested:** 22 open-source models (2B–70B parameters, 12 families) run locally through Ollama.
- **On what:** 4,000 Lichess puzzle positions, 1,000 in each of four rating tiers.
- **Three tasks per position:** evaluate it (T1), choose a move (T2), explain who stands better and why (T3).
- **Against what:** the Lc0 chess engine as ground truth, plus chance and trivial-answer baselines.
- **Scale:** 6 prompt formats, 526,662 evaluations in the base benchmark.

**Headline result.** The models' own first moves are legal about 1 time in 10; their move choices are
1.16× better than a random legal move; their evaluations are worse than always answering "0"; and
size doesn't help except within Google's Gemma models. Fine-tuning small models to read the board
taught them to read the board, but not to play better. All of it fits pattern matching better than
reasoning. Full detail: [docs/FINDINGS.md](docs/FINDINGS.md).

---

## 2. Where to start reading

1. **[docs/SIMPLE_SUMMARY.md](docs/SIMPLE_SUMMARY.md)**: the whole project in plain language.
2. **[docs/FINDINGS.md](docs/FINDINGS.md)**: every result. Its tables are generated from the metrics,
   and `scripts/render_findings.py --check` fails if they go stale.
3. **[docs/issues-log.md](docs/issues-log.md)**: every measurement problem found, and how it was fixed.
   The section "How the audit changed the numbers" in FINDINGS summarises it.
4. **[specs/](specs/)**: what each experiment set out to test, written before it ran. The folder's
   README groups them by topic.
5. **[results/](results/)**: the numbers themselves. The folder's README says which file holds what.

---

## 3. The experiments

| # | Experiment | Status | Spec (written first) | Config | Results | Write-up |
|---|---|---|---|---|---|---|
| 1 | **Base benchmark**: 22 models × 4,000 positions × 6 formats | Done | [specs/benchmark.md](specs/benchmark.md) | `config/config.yaml` | `results/metrics/` | FINDINGS 1–6 |
| 2 | **Base re-run**: 5 models again, saving raw replies (first-answer legality) | Done | [specs/t2-move-source.md](specs/t2-move-source.md) | `config/config_base_rerun.yaml` | `results/finetuning_experiment/base_rerun/` | FINDINGS 2, F |
| 3 | **Pieces-only fine-tune**: teach 5 models to read the board | Done | [fine-tuning/specs/chess-board-finetuning.md](fine-tuning/specs/chess-board-finetuning.md) | `config/config_finetuned.yaml` | `results/finetuning_experiment/` | FINDINGS F |
| 4 | **Pieces + legal moves fine-tune**: also teach the legal moves | Done (criteria fixed in advance) | [fine-tuning/specs/legal-moves-finetuning.md](fine-tuning/specs/legal-moves-finetuning.md) | `config/config_legal_ft.yaml` | `results/finetuning_experiment/legal_ft/` | FINDINGS F |
| 5 | **Engine-strength check**: does a 10× stronger engine change the ranking? | Done | [specs/engine-strength-check.md](specs/engine-strength-check.md) | — | `results/engine_strength/`, `results/metrics/engine_strength*` | FINDINGS, Limitations |
| 6 | **Notation check**: same positions as a plain-English board, no FEN | **Paused: 14/21; scoring pending** | [specs/plain-board-prompts.md](specs/plain-board-prompts.md) | `config/config_plain_board.yaml` | `results/plain_board/` | To be added |
| 7 | **Commercial models** (Claude and others via API) | Haiku/Opus scored; C1 pending; Sonnet not run | [registered spec](https://github.com/rabrew/chess-llm-bench/blob/claude-models/specs/commercial-models.md) | `config/config_claude.yaml` (claude-models) | [Claude summary](https://github.com/rabrew/chess-llm-bench/blob/1da3d14/results/claude/metrics/claude_summary.json) | [Conclusions §5](docs/CONCLUSIONS.md#5-claude-stronger-descriptive-competence-incomplete-move-quality-evidence) |

---

## 4. How the benchmark works

**Positions.** 4,000 Lichess puzzles, 1,000 per rating tier (easy < 1200, medium 1200–1800,
hard 1800–2400, extreme 2400+), sampled with seed 42.

**Prompt formats.** Three combined prompts ask all three questions at once (`fen_only`, `pgn+fen`, `cot`).
Three isolated prompts ask one question each (`eval_only`, `move_only`, `explanation_only`).

**Ground truth.** Lc0, chosen because it runs on the GPU:
- T1: a single forward pass of the network, converted to centipawns, capped at ±10,000.
- T2: Lc0 at 800 nodes for the best move and centipawn loss (CPL), clamped to ±1,000.
- Fields named `stockfish_*` are legacy names and hold Lc0 values.

**Scoring.**

| Task | What is measured | Headline metric |
|---|---|---|
| **T1: Evaluation** | The model's centipawn estimate | Spearman ρ against Lc0; error compared with always answering 0; chance-corrected direction agreement (κ) |
| **T2: Move** | A legal, good move | Blunder rate (clamped CPL ≥ 1,000, bootstrap CI); CPL buckets; best-move rate against a random legal move; legality on the first answer and after re-asking |
| **T3: Explanation** | Who stands better, and why | Side mark against a majority baseline (κ); whole-word theme mark |

**Models.** The full list with sizes and families is in [config/config.yaml](config/config.yaml).

**Hardware.** One PC: RTX 5080 (16 GB), 32 GB RAM. Models run through Ollama; Lc0 runs on the GPU.

---

## 5. Repository map

| Folder | What's there | Guide |
|---|---|---|
| `docs/` | Findings, plain summary, issues log, research notes, runbook | [docs/README.md](docs/README.md) |
| `specs/` | One spec per feature or experiment, written before the code | [specs/README.md](specs/README.md) |
| `results/` | Metrics, plots, logs and comparison tables for every run | [results/README.md](results/README.md) |
| `fine-tuning/` | The fine-tuning experiments: data, training, export, comparison | [fine-tuning/README.md](fine-tuning/README.md) |
| `src/` | The library: prompts, model client, scoring, metrics, job queue | — |
| `scripts/` | Command-line tools: build data, run workers, analyse, plot | [docs/SCRIPTS.md](docs/SCRIPTS.md) |
| `bin/` | Shell scripts that run the whole pipeline | [docs/SCRIPTS.md](docs/SCRIPTS.md) |
| `config/` | One YAML file per run: models, prompt formats, paths | — |
| `tests/` | pytest suite for `src/` and `scripts/` (fine-tuning tests are in `fine-tuning/tests/`) | — |
| `runs/` | One record per training run: config, git commit, seed, data hash, metrics | — |
| `dashboard/` | Live web dashboard used while runs were in progress | [specs/dashboard.md](specs/dashboard.md) |

Not in the repository: `data/` (the Lichess dataset, downloadable), `jobs/` (runtime job queues) and
the fine-tuned model weights.

---

## 6. Raw data

One JSON record per model answer: the prompt format, the position, the model's answers, the engine's
values and the scores. Later runs also keep the model's full reply. The files are 110–460 MB, over
GitHub's 100 MB limit, so they are attached xz-compressed to the release
[`data-2026-10-31`](https://github.com/rabrew/chess-llm-bench/releases/tag/data-2026-10-31), with
SHA-256 checksums.

| Release file | Goes to | What it is |
|---|---|---|
| `base_evaluations.jsonl.xz` | `results/evaluations.jsonl` | Base benchmark, 22 models, 526,662 records |
| `base_rerun_evaluations.jsonl.xz` | `results/finetuning_experiment/evaluations_base_rerun.jsonl` | Base re-run of the 5 fine-tuned models, with raw replies |
| `pieces_ft_evaluations.jsonl.xz` | `results/finetuning_experiment/evaluations_ft.jsonl` | Pieces-only fine-tune |
| `legal_ft_evaluations.jsonl.xz` | `results/finetuning_experiment/evaluations_legal_ft.jsonl` | Pieces + legal moves fine-tune |
| `plain_board_evaluations_PARTIAL_*.jsonl.xz` | `results/plain_board/evaluations.jsonl` | Notation check, **in progress** (snapshot) |
| `*_pre_audit_backup.jsonl.xz` | — | The same files before the October 2026 audit corrections, for comparison |

```bash
gh release download data-2026-10-31 --repo rabrew/chess-llm-bench --dir release_data
cd release_data && sha256sum -c SHA256SUMS.txt && xz -dk *.xz
```

---

## 7. Reproducibility

**Reproducible:** the dataset, the engine ground truth and the scoring.
- **Positions:** sampled with `random_seed: 42` (`config/config.yaml`). Position IDs are a hash of the
  FEN, so they are stable across rebuilds.
- **Engine truth:** Lc0, precomputed and stored. The node count is on every record (`t2_eval_nodes`).
- **Provenance:** every `summary.json` records the git commit, the results-file hash, the Lc0 weights
  hash and the config.
- **Dependencies:** pinned in `requirements.txt`; dev tools (ruff) in `requirements-dev.txt`.

**Not reproducible:** the model outputs. Five of the six prompt formats ran at Ollama's default
temperature, and given the identical prompt twice a model gave the same move 68.7% of the time
(Issue 23).

---

## 8. Running it yourself

```bash
bash bin/run_pipeline.sh                                 # the full benchmark pipeline
venv/bin/python -m pytest -q tests fine-tuning/tests     # full test suite
venv/bin/ruff check src scripts fine-tuning tests        # lint (config: ruff.toml)
venv/bin/python scripts/render_findings.py --check       # FINDINGS tables match the metrics
```

Step-by-step instructions: [docs/RUNBOOK.md](docs/RUNBOOK.md). Every change follows `CLAUDE.md`:
spec, then tests, then code, then commit.
