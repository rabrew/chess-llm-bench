# Spec: run_all.sh Tier Tracking and Output Paths

**Builds on:** [finetuned-benchmark-run.md](finetuned-benchmark-run.md)
**Created:** 2026-10-23

## Goal

The fine-tuned run (finished 2026-10-23) completed all 20 model × tier combinations, each with 6,000
records. Even so, `jobs/db/skipped_tiers.txt` listed all 20 as "abandoned due to exhausted queue". This
spec fixes three defects in `bin/run_all.sh` and one deprecation in `scripts/run_workers.py`.

| # | Defect | Effect |
|---|---|---|
| 1 | After the workers return, the loop checks for pending jobs **before** checking whether the tier is complete. A finished tier also has 0 pending jobs. | Every completed tier is written to `skipped_tiers.txt`. `tier_done` trusts that file before it counts results, so a later run with a moved or deleted results file silently skips every tier. |
| 2 | The pending count is inline Python with `except Exception: print(0)` and `\|\| echo 0` | A locked, corrupt or missing DB reads as "0 pending", so the tier is marked abandoned instead of the run stopping |
| 3 | The closing summary prints `results/plots/` and `results/metrics/` | These are wrong for any config other than the base one |
| 4 | `datetime.utcnow()` in `run_workers.py` | DeprecationWarning on Python 3.12+ in every tier's log |

## Inputs / Outputs

`scripts/pending_jobs.py --db <path> --model <tag>` prints the number of jobs for that model whose
status is `pending` or `in_progress`, and exits 0.
- If the DB cannot be read, it prints the reason to stderr and exits 1.
- Its function `count_pending(db_path, model) -> int` raises `FileNotFoundError` for a missing DB and
  `sqlite3.Error` for an unreadable one.
- It opens the DB **read-only** (`file:<path>?mode=ro`), so it never creates an empty DB file. Opening
  with the `sqlite3` CLI did exactly that on 2026-10-23.

`bin/run_all.sh`:
- After each worker pass, if `tier_done` → `break`. Only then count pending jobs.
- If `pending_jobs.py` exits non-zero, print the error and `exit 1`. Do not mark the tier as skipped.
- The closing summary prints `cfg_path plots_dir` and `cfg_path metrics_dir`.

`scripts/run_workers.py` uses `datetime.now(timezone.utc)`. The stored formats stay the same: the run log
`timestamp` is `YYYY-MM-DDTHH:MM:SS.ffffffZ`, and `run_id` is `YYYYMMDD_HHMMSS`.

## Edge cases

| Case | Behaviour |
|---|---|
| Tier complete, 0 pending | Break. Not written to `skipped_tiers.txt` |
| Tier short of results, 0 pending (failures exhausted the queue) | Written to `skipped_tiers.txt`, as before |
| DB missing, locked or corrupt | Exit 1 with the reason. No empty DB is created |
| Jobs for other models in the DB | Not counted |
| `done` / `failed` jobs | Not counted |

## Dependencies

Python stdlib (`sqlite3`, `argparse`). No new packages.

## Verification (`tests/test_run_all_tier_tracking.py`)

- `count_pending` counts `pending` and `in_progress` jobs for the given model only
- A missing DB raises `FileNotFoundError` and no file is created
- A file that is not a database raises `sqlite3.Error`
- The CLI prints the count and exits 0; on a missing DB it exits 1
- In `run_all.sh`, inside the worker loop, `tier_done` is checked before the pending count, and before
  any write to `SKIPPED_TIERS`
- `run_all.sh` has no `except Exception` fallback to 0 and no hard-coded `results/plots/` or
  `results/metrics/`
- `run_workers.py` does not use `utcnow`, and `write_run_log` keeps the trailing-`Z` timestamp format

---

## Added 2026-10-31: tier target from the config, and one skip list per run

Two more defects, found before the plain-board run (`specs/plain-board-prompts.md`):

| # | Defect | Effect |
|---|---|---|
| 5 | `JOBS_PER_TIER=5900` is hard-coded for 1,000 positions × 6 formats | A run with fewer positions or formats (plain-board: 125 × 3 = 375 per tier) never reaches it, so every finished tier is recorded as abandoned |
| 6 | `skipped_tiers.txt` sits next to the jobs DB, shared by every config | An entry from one run (e.g. `llama3.2:3b::easy`) makes a later run with a different config skip that tier without running it |

**Fix**
- `scripts/tier_target.py --config <path>` prints the results needed per model per tier:
  `floor(positions per tier × number of prompt formats × 5,900 / 6,000)`. Positions per tier is
  `benchmark.sample_per_tier` when set, otherwise `benchmark.max_positions_per_tier`. The 5,900/6,000
  allowance (jobs that fail every attempt) is the one the full benchmark always used, so its target
  stays 5,900. A missing or invalid setting exits 1 with the reason; `run_all.sh` stops.
- `run_all.sh` sets `JOBS_PER_TIER` from that script, and keeps its skip list at
  `<jobs DB path without .db>_skipped_tiers.txt`, so each config has its own.

**Tests:** the target is 5,900 for the full benchmark and 368 for plain-board; `sample_per_tier`
wins over `max_positions_per_tier`; a missing `prompt_formats` or positions setting exits 1;
`run_all.sh` has no hard-coded `JOBS_PER_TIER=` number and no shared `skipped_tiers.txt` path.
