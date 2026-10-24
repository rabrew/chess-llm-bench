# Scripts Reference

Quick guide to every script in `scripts/` and the shell helpers in `bin/`.

---

## `bin/` — shell scripts

| Script | Purpose |
|---|---|
| `bin/run_pipeline.sh` | **Start here.** Creates `results/logs/`, runs `bin/run_all.sh` and tees output to a timestamped log. |
| `bin/run_all.sh` | Orchestrates the full benchmark: iterates every model × tier, generates jobs, runs workers, and wipes the DB between tiers. |
| `bin/run_forever.sh` | Crash-restart wrapper — relaunches `run_all.sh` automatically until it exits 0. |
| `bin/monitor_progress.sh` | Live terminal dashboard (refreshes every 5 s). Shows per-model progress bars, done/failed/active counts, and which tier is running. |
| `bin/monitor.sh` | Background watchdog that polls a tmux session for crash/error patterns and logs them to a file. Useful for overnight runs. |
| `bin/goodnight.sh` | Shuts down the machine and sets an RTC wake alarm for 6 am. |
| `bin/sleeptime.sh` | Pauses the benchmark for a given duration (kills Ollama, restarts after sleep). |
| `bin/pause_resume.sh` | Sends SIGSTOP/SIGCONT to a process group to pause/resume the pipeline. |
| `bin/restart.sh` | Kills the running pipeline and restarts it in the `chess-bench` tmux session. |
| `bin/auto_sync.sh` | Auto-commits and pushes tracked file changes every 30 minutes. |

---

## `scripts/`

### Running the benchmark

**`run_workers.py`**
Spawns N parallel worker processes that pull jobs from the SQLite queue, call Ollama, score results (T1/T2/T3), and write to `results/evaluations.jsonl`.
```
python scripts/run_workers.py --workers 6 --tier easy
```

**`generate_jobs.py`**
Populates the SQLite job queue for a given model/tier combination. Also accepts `--estimate` to preview job counts without writing anything.
```
python scripts/generate_jobs.py --tier easy --model mistral:7b
python scripts/generate_jobs.py --estimate
```

### Dataset & pre-computation

**`build_dataset.py`**
Builds the chess position dataset from source material and writes JSON files to `data/`. Run once before the benchmark, or to regenerate after changing position sources.
```
python scripts/build_dataset.py --output-dir data/
```

**`precompute_stockfish.py`**
Pre-computes Stockfish evaluations (best move + centipawn score) for every position in the dataset. Uses `ProcessPoolExecutor` for parallelism. Required before scoring T3 (move quality).
```
python scripts/precompute_stockfish.py --depth 20 --workers 8
```

**`precompute_lc0.py`**
Pre-computes Lc0 neural-net evaluations using the engine wrapper. GPU-accelerated via CUDA. Slower than the batch variants below but simpler.

**`precompute_lc0_batch.py`**
Faster Lc0 evaluation using ONNX Runtime directly (no UCI subprocess). Encodes positions to Lc0's 112-plane format and runs GPU inference in batches.

**`precompute_lc0_batch_v2.py`**
Optimised version of `precompute_lc0_batch.py` with pre-encoding for extra throughput. Prefer this over v1 for large position sets.

**`precompute_lc0_fast.py`**
Lc0 via UCI in `valuehead` mode with 1 node — trades accuracy for speed. Useful for quick sanity checks or large datasets where approximate evals are fine.

### Post-run tools

**`retry_illegal_moves.py`**
Re-queries models for every record in `evaluations.jsonl` where `t2_legal = False`, this time providing the full legal move list so the model must pick a valid move. Writes to `results/evaluations_retried.jsonl`. Use to get a ceiling on T2 legal-move rate.
```
python scripts/retry_illegal_moves.py --model phi4:14b
python scripts/retry_illegal_moves.py --dry-run
```

**`validate_responses.py`**
Re-scores every record in `evaluations.jsonl` through the current evaluator logic. Useful after changing scoring rules to verify old results are still consistent — or to spot regressions.

**`generate_plots.py`**
Reads `evaluations.jsonl`, aggregates metrics (by model, difficulty, phase, source, family), and writes charts to `results/plots/` and summary CSVs to `results/metrics/`.
```
python scripts/generate_plots.py
```

**`enrich_cpl.py`**
Fills `t2_cpl`, `t2_best_move` and the 800-node "before" eval (`t2_eval_before`, `t2_eval_nodes`) with
Lc0. `--backfill-eval-before` adds the "before" eval to older records: it backs up the file first
and resolves symlinks (specs/cpl-eval-before.md).
```
python scripts/enrich_cpl.py --results results/evaluations.jsonl
python scripts/enrich_cpl.py --results results/evaluations.jsonl --backfill-eval-before
```

**`build_puzzle_meta.py`**
Joins the Lichess CSV back to the benchmark FENs and writes `data/puzzle_meta.json`, with the full
themes, the setup move and the puzzle FEN (Issue 21, specs/puzzle-metadata-join.md).
```
python scripts/build_puzzle_meta.py --results results/evaluations.jsonl
```

**`find_t3_examples.py`**
Reports, for each plain-English theme cue, how many unmarked explanations it would credit (gain) and how
often it appears for other themes (leak). Writes `results/t3_examples.md` (specs/t3-example-finder.md).

**`render_findings.py`**
Regenerates the tables in `docs/FINDINGS.md` and `docs/SIMPLE_SUMMARY.md` from the metrics. `--check`
exits 1 if a table is stale (specs/findings-tables.md).
```
python scripts/render_findings.py
python scripts/render_findings.py --check
```

**`check_ft_run.py`** / **`pending_jobs.py`**
The pre-flight check that a fine-tuned config matches its base config, and a read-only count of pending
jobs for `run_all.sh`.

### Development / debugging

**`test_parsers_live.py`**
Sanity-check script: queries each configured model with all 3 prompt formats and verifies that `parse_response` extracts T1/T2/T3 fields without errors. **Run this before starting a full benchmark** to catch prompt or parsing bugs early.
```
python scripts/test_parsers_live.py
```

**`pull_models.py`**
Pulls all models listed in `config/config.yaml` from Ollama. Also accepts `--model` to pull a single model, and `--list` to show what's configured.
```
python scripts/pull_models.py
python scripts/pull_models.py --model phi4:14b
```

---

## `fine-tuning/scripts/`

| Script | Purpose |
|---|---|
| `generate_board_dataset.py` | Builds the training data. `--task board+legal --from-dataset data/board_dataset.jsonl` builds the pieces + legal moves version on the same positions |
| `finetune_model.py` | QLoRA training for one model, then GGUF export; writes a `runs/` record |
| `import_finetuned_to_ollama.py`, `verify_export.py` | Import the GGUF into Ollama, then check that it still reads boards |
| `run_batch.sh` | Train, import and verify each listed model; `FT_CONFIG=...` picks the config; exits 1 if any model fails |
| `board_reading_test.py` | F-9 board-reading test, base vs fine-tuned; `--task board+legal` also scores the legal-move list |
| `compare_finetuning_results.py` | Paired base vs fine-tuned comparison (ρ, legality, first-answer legality, blunder rate, T3) |
| `run_record.py backfill` | Writes `runs/` records for models trained before tracking existed |

---

## `src/` (library modules, not run directly)

| Module | Role |
|---|---|
| `llm_client.py` | Ollama HTTP client, prompt builders (`build_prompt`, `build_move_prompt`), response parser |
| `evaluator.py` | Scoring functions for T1 (position understanding), T2 (legal move), T3 (move quality) |
| `worker.py` | Single-worker loop: dequeues a job, calls LLM, scores, writes result |
| `job_queue.py` | SQLite-backed job queue (enqueue, dequeue, mark done/failed) |
| `job_generator.py` | Generates job records from positions × models × formats |
| `dataset_builder.py` | Loads and preprocesses raw position data into the benchmark dataset |
| `engine_wrapper.py` | Thin wrappers around Stockfish and Lc0 UCI engines |
| `position_generator.py` | Generates or filters chess positions by difficulty tier |
| `metrics.py` | DataFrame aggregations and metric calculations over `evaluations.jsonl` |
| `result_writer.py` | Thread-safe append writer for `evaluations.jsonl` |
| `data_loader.py` | Loads dataset files from `data/` |
| `feedback_loop.py` | (Experimental) Logic for iterative prompt improvement based on failure patterns |
| `utils.py` | Config loading, logging setup, directory helpers, `MODEL_REGISTRY` |
| `provenance.py` | Git, data and engine provenance for `summary.json` |
