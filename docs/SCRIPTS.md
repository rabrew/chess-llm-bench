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
Pre-computes Stockfish evaluations (best move + centipawn score) for every position in the dataset. Uses `ProcessPoolExecutor` for parallelism. An optional reference preprocessing path for T1; T3 scores explanations.
```
python scripts/precompute_stockfish.py --depth 20 --workers 8
```

**`precompute_lc0_batch_v2.py`**
Direct ONNX value evaluation with explicit reference provenance. This is the available Lc0 batch preprocessing script; older script variants documented here are absent. Runtime inference is a GPU operation and was not used for repair validation.

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
Reads audited diagnostic exports and renders direction, endpoint-quality availability and aligned heatmaps. Old chart filenames are retired historical artifacts. Use a new output directory; a chart directory is not evidence that all files in it are current.
```
python scripts/generate_plots.py
```

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

## `src/` (library modules, not run directly)

| Module | Role |
|---|---|
| `llm_client.py` | Ollama HTTP client, prompt builders (`build_prompt`, `build_move_prompt`), response parser |
| `evaluator.py` | Scoring functions for T1 (position understanding), T2 (legal move), T3 (explanation side/theme matching) |
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
| `utils.py` | Config loading, logging setup, directory helpers |

## Audited postprocessing
`scripts/generate_audit_metrics.py` requires an explicit output directory, hashes the input, records provenance uncertainty and generates corrected exports without model/engine calls. Run `--help` for current arguments. New charts use direction_diagnostics.csv and cpl_quality.csv. Legacy plots in results/plots are not refreshed or endorsed.

Shell helpers derive the repository root from their own location or CHESS_BENCH_ROOT. Model storage uses OLLAMA_MODELS or config.ollama.models_dir. qwen2.5:72b is postponed and is not one of the 22 evaluated models.
