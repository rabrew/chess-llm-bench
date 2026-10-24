# Spec: Shell Scripts Locate the Repo Themselves

**Created:** 2026-10-24 (quality gate)

## Goal

Six scripts in `bin/` hardcoded `/home/rabrew/Desktop/chess-llm-bench`. That is the **old** checkout,
not the working one (`~/Desktop/clb-analysis`). So `bin/run_pipeline.sh`, which the README gives as
the quick start, ran old code against old results. CLAUDE.md says never to hardcode paths.

## Inputs / Outputs

Every script in `bin/` that needs the repo sets:

    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

and uses `$REPO` for every path, so it works from any checkout and any working directory.

## Edge cases

- The scripts are run through a symlink or from another directory: `BASH_SOURCE` plus `pwd` resolve
  the script's own location.
- A path for a separate tool (for example a log outside the repo) must come from a variable or an
  argument, never a literal home directory.

## Dependencies

bash.

## Verification (`tests/test_bin_scripts.py`)

- No file in `bin/` contains a literal `/home/` path
- Every `bin/*.sh` that runs `cd` derives the path from `BASH_SOURCE` / `dirname "$0"`
