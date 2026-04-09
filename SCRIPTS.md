# Scripts Reference

Quick guide to every script in `scripts/` and the top-level shell helpers.

---

## Top-level shell scripts

| Script | Purpose |
|---|---|
| `run_pipeline.sh` | **Start here.** `cd`s into the project root, creates `results/logs/`, and runs `run_all.sh` while tee-ing output to a timestamped log. |
| `run_all.sh` | Orchestrates the full benchmark: iterates every model × tier, generates jobs, runs workers, and wipes the DB between tiers. Called by `run_pipeline.sh`. |
| `monitor_progress.sh` | Live terminal dashboard (refreshes every 3 s). Shows per-model progress bars, done/failed/active counts, and which tier is running. Uses the shared SQLite job DB. |
| `monitor.sh` | Background watchdog that polls a tmux session for crash/error patterns and logs them to a file. Useful for overnight runs. |

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
| `utils.py` | Config loading, logging setup, directory helpers |
