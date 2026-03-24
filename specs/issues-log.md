# Issues Log — Chess LLM Benchmark
**Period:** March 16–19, 2026

This document covers every technical problem encountered during development and execution of the benchmark pipeline, how each was diagnosed, and what the fix was.

---

## Issue 1 — Wrong Model Name in Config
**Date:** March 16
**Symptom:** Pipeline failed at step 4 (model pull) — `ollama pull llama3.2:8b` returned an error because the model does not exist.
**Root cause:** Config file had `llama3.2:8b` — this model tag does not exist on Ollama's registry. The 8B variant of the llama3 family is released under `llama3.1`, not `llama3.2`.
**Fix:** Changed `llama3.2:8b` → `llama3.1:8b` in `config/config.yaml`.

---

## Issue 2 — Out-of-Memory Kill During Job Generation
**Date:** March 16
**Symptom:** Pipeline was OOM-killed mid-run with no output. System logs showed the process was killed by the kernel.
**Root cause:** `generate_standard_jobs()` in `src/job_generator.py` built the entire job list as a Python list before inserting anything into SQLite. With 19 models × 4 tiers × 1,000 positions × 3 prompt formats = 228,000 job dicts all held in RAM simultaneously, this exhausted available memory.
**Fix:** Replaced full in-memory generation with a streaming batch approach — generates and inserts 10,000 jobs at a time, keeping memory flat regardless of scale.

---

## Issue 3 — SQLite File Descriptor Exhaustion
**Date:** March 16
**Symptom:** `sqlite3.OperationalError: unable to open database file` after inserting a few thousand jobs.
**Root cause:** `insert_jobs()` called `insert_job()` in a loop. Each call to `insert_job()` opened and closed a new SQLite connection. With batches of 10,000 jobs this eventually hit the OS file descriptor limit.
**Fix:** Rewrote `insert_jobs()` to open a single connection for the entire batch, wrap all inserts in one transaction, and commit once at the end.

---

## Issue 4 — Dataset Builder Produced 0 Positions
**Date:** March 16
**Symptom:** Pipeline completed step 2 but all four dataset files (`easy.json`, `medium.json`, `hard.json`, `extreme.json`) contained 0 positions. Crash monitor flagged this immediately.
**Root cause:** Stockfish evaluation step ran but the position filtering logic had a bug — all positions were being filtered out before being saved. The validation step (5.8M positions across 24 CPU cores) completed successfully, but the downstream filtering wrote empty files.
**Fix:** Debugged the filtering condition in `src/dataset_builder.py`. The Stockfish evaluation was rerun, and the correct 4,000 positions (1,000 per tier) were saved.

---

## Issue 5 — System Freeze Under High Worker Load
**Date:** March 16–17
**Symptom:** Screen went black, mouse stopped responding, system was unresponsive and required a hard reboot. This happened during benchmark runs.
**Root cause:** The pipeline was configured with 16 parallel workers, each holding a connection to Ollama and running Stockfish simultaneously. Combined CPU + RAM pressure from 16 concurrent inference + evaluation threads overwhelmed the system.
**Fix:** Reduced worker count from 16 → 4 in `config/config.yaml`. Models are now run one at a time per tier (already the design), and within each run only 4 workers operate concurrently.

---

## Issue 6 — Ollama Model Storage — Disk Space Split
**Date:** March 17
**Symptom:** Could not download all 19 models — main partition (`/dev/nvme0n1p5`, 300GB) did not have enough space for the full ~250GB model set alongside the OS and project data.
**Root cause:** The 19 models range from 2GB (llama3.2:3b) to 47GB (qwen2.5:72b). The shared NTFS partition (`/dev/nvme0n1p6`, 338GB, labeled "Shared") had sufficient free space but Ollama was not pointed at it.
**Fix:** Moved model blobs to `/mnt/shared/ollama_models/`, created symlinks from the main Ollama models directory into the shared partition so Ollama sees all 19 models from a single path. The shared partition must be mounted before starting Ollama.

---

## Issue 7 — Ollama Not Running on Pipeline Start
**Date:** March 19
**Symptom:** Pipeline failed at step 4 with a curl error — `http://localhost:11434/api/tags` returned nothing. The `set -e` flag in `run_all.sh` caused an immediate exit.
**Root cause:** Ollama was not started before launching the pipeline. Because the service runs in the background, it is easy to forget.
**Fix:** Start Ollama manually first in a dedicated tmux session (`tmux new-session -d -s ollama-serve && tmux send-keys -t ollama-serve 'OLLAMA_MODELS=/mnt/shared/ollama_models ollama serve' Enter`), then launch the pipeline.

---

## Issue 8 — Shared Partition Not Mounted, Ollama Found 0 Models
**Date:** March 19
**Symptom:** Ollama started successfully but reported 0 available models. The pipeline tried to pull all 19 models from scratch.
**Root cause:** The shared NTFS partition (`/dev/nvme0n1p6`) was not mounted. Without it, the model directory at `/mnt/shared/ollama_models/` was empty. The partition is not in `/etc/fstab` and must be mounted manually each session.
**Fix:** `sudo mount /dev/nvme0n1p6 /mnt/shared` before starting Ollama, then start Ollama with `OLLAMA_MODELS=/mnt/shared/ollama_models`.
**Longer-term fix:** Add the partition to `/etc/fstab` to auto-mount on boot.

---

## Issue 9 — Test Suite Had a Wrong Assertion (False Negative)
**Date:** March 19
**Symptom:** `pytest` reported 1 failure in `test_evaluator.py::TestScoreT2::test_illegal_move`.
**Root cause:** The test asserted that `Qh4` is illegal in position `r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3`, but the move is actually legal — the Black queen can travel d8→e7→f6→g5→h4 with a clear path. The evaluator was correct; the test was wrong.
**Fix:** Replaced `Qh4` with `Qa5`, which is genuinely illegal because the path d8→c7 is blocked by Black's own pawn on c7.

---

## Issue 10 — Response Parser Dropped All Fields on Markdown Bold Labels
**Date:** March 19
**Symptom:** gemma3:4b produced "All three fields missing from response" errors for every job in the `cot` prompt format. T1, T2, and T3 all came back `None`.
**Root cause:** gemma3:4b wraps its numbered labels in markdown bold: `**1. Eval:** 0`. The parser stripped the numbered prefix (`1.`) first, then stripped `**` — but by then the numbered prefix was gone and the bold stripping left `1. Eval: 0` with the number still attached. Actually the issue was the opposite order: the bold strip came *after* the numbered-prefix strip, so `**1. Eval:**` didn't match `^\d+\.` (because it starts with `**`), and after the bold strip it became `1. Eval:` — which then couldn't be matched because the numbered prefix stripping had already run.
**Fix:** Swapped the order — strip markdown bold (`**`) first, then strip the numbered prefix. After this, `**1. Eval:** 0` → `1. Eval: 0` → `Eval: 0` correctly.

---

## Issue 11 — Parser Kept Backticks and Asterisks Inside Move Values
**Date:** March 19
**Symptom:** Some models output moves in inline code (`` `Nf6` ``) or bold (`**Rxc6**`). The parser stored the move with the formatting characters intact (e.g. `` `Nf6` `` or `**Rxc6**`), causing every subsequent legality check to fail since `chess.Board.parse_san()` cannot parse those strings.
**Root cause:** The move extraction took the first whitespace-delimited token and stripped trailing punctuation, but did not strip backtick or asterisk wrappers.
**Fix:** Added `move_str = move_str.strip("`")` after extracting the move, and the bold strip (`line.replace("**", "")`) already handled the asterisk case once the order fix in Issue 10 was applied.

---

## Issue 12 — Stuck `in_progress` Jobs After Ctrl-C
**Date:** March 19
**Symptom:** After interrupting the pipeline mid-run, restarting it caused workers to stall — the DB showed 4 jobs perpetually `in_progress` with no completions, even after 30+ seconds.
**Root cause:** When the pipeline was killed with Ctrl-C, the 4 worker subprocesses were holding jobs in `in_progress` state. When they were killed, those jobs were never marked `done` or `failed`. Restarted workers only claim `pending` jobs, so those 4 jobs blocked a slot permanently and the 2,280 pending jobs waited behind them.
**Fix:** Manually reset stuck jobs: `UPDATE jobs SET status='pending', worker_id=NULL WHERE status='in_progress'`, then restarted the pipeline.

---

## Issue 13 — Old Worker Process Continued Running With Pre-Fix Code
**Date:** March 19
**Symptom:** After fixing the parser (Issue 10) and restarting the pipeline, "All three fields missing" errors continued at the same rate. DB counts did not move for 30+ seconds.
**Root cause:** Python loads module code at import time and caches it in memory for the life of the process. The original worker process (PID 16221) was started before the parser fix was written to disk. Even after the file was saved, the running process continued using the old `parse_response` function from its in-memory module cache.
**Fix:** Force-killed the original worker process (`kill -9 16221`), then restarted the pipeline so a fresh process imported the fixed code.

---

## Summary Table

| # | Date | Component | Impact | Fix |
|---|------|-----------|--------|-----|
| 1 | Mar 16 | config.yaml | Pipeline failed at model pull | Corrected model tag |
| 2 | Mar 16 | job_generator.py | OOM kill | Streaming batch inserts |
| 3 | Mar 16 | job_queue.py | SQLite FD exhaustion | Single connection per batch |
| 4 | Mar 16 | dataset_builder.py | 0 positions saved | Fixed filtering bug |
| 5 | Mar 16–17 | System / workers | Full system freeze | Reduced workers 16 → 4 |
| 6 | Mar 17 | Ollama / storage | Models couldn't fit on disk | Symlink to shared partition |
| 7 | Mar 19 | Pipeline startup | Immediate exit at step 4 | Start Ollama before pipeline |
| 8 | Mar 19 | Shared partition | 0 models visible to Ollama | Mount partition first |
| 9 | Mar 19 | test_evaluator.py | False test failure | Fixed wrong move assumption |
| 10 | Mar 19 | llm_client.py parser | All fields None for gemma3 | Fixed strip order (bold before number) |
| 11 | Mar 19 | llm_client.py parser | Moves failed legality check | Strip backticks from move value |
| 12 | Mar 19 | jobs.db / workers | Workers stalled on restart | Reset in_progress → pending manually |
| 13 | Mar 19 | Worker process | Fix not applied to live run | Force-killed old process to reload code |
