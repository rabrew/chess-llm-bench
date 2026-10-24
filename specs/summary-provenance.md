# Spec: Provenance Block in summary.json (P-2)

**Checklist item:** P-2 (D8) in [exhibition-readiness.md](exhibition-readiness.md)
**Created:** 2026-10-24

---

## Goal

Every number in the write-up should trace back to the exact code, data, engine and settings that
produced it. `summary.json` currently has no record of any of these. A provenance block is added, so a
reader (or Dr Brew) can tell which run a figure came from, and so stale summaries are detectable.

## Inputs / Outputs

**New module `src/provenance.py`:**

`build_provenance(config_path, results_file, repo_dir=".") -> dict` returns:

| Key | Value |
|---|---|
| `generated_at` | UTC ISO-8601 timestamp |
| `git_sha` | `git rev-parse HEAD` in `repo_dir`, or `None` (with a warning) if git is unavailable or this isn't a repo |
| `git_dirty` | True if `git status --porcelain` lists tracked changes; `None` if unknown |
| `results_file` | the path as given, plus `results_sha256` and `results_records` (line count) |
| `engine` | `{"name": "lc0", "binary", "weights", "weights_sha256", "nodes"}` from `config.lc0`; `weights_sha256` is `None` if the file is missing |
| `config_path`, `config` | the path and the parsed config dict (the snapshot) |
| `python`, `packages` | the Python version, and the versions of pandas, numpy and python-chess |

**`save_metrics(df, output_dir, provenance=None)`:** when given, writes it as
`summary["provenance"]`.

**`scripts/generate_plots.py --save-metrics`:** builds the provenance from `--config` and the results
file and passes it on.

## Steps / Logic

- Files are hashed in 1 MiB chunks, so the ~460 MB results file isn't loaded into memory.
- Git is run with `subprocess.run([...], capture_output=True, check=False, timeout=...)`. There is no
  shell, so no command injection.
- Failures (no git, missing weights) produce `None` plus a logged warning, never a crash, so a
  provenance problem can't block the metrics.
- `config` is the dict exactly as loaded. It holds paths and settings, no secrets; the repo keeps
  none in config.

## Edge cases

| Case | Behaviour |
|---|---|
| Not a git repo, or git missing | `git_sha` / `git_dirty` are `None` |
| A missing results file | `results_sha256` / `results_records` are `None` |
| No `lc0` section | `engine` is `None` |
| `provenance=None` | No key written; old callers are unchanged |

## Dependencies

The standard library (`hashlib`, `subprocess`, `platform`, `importlib.metadata`) plus `src.utils.load_config`.

## Verification (`tests/test_provenance.py`)

- The SHA and dirty flag in a temporary git repo, and `None` outside one
- The results hash and line count match a known file; a missing file gives `None`
- The engine block reads the config and hashes the weights; missing weights give a `None` hash
- `save_metrics(..., provenance=...)` writes `summary["provenance"]`, and without it there is no key
