# Instructions for coding agents (Codex, Claude Code, ...)

Read [docs/HANDOFF.md](docs/HANDOFF.md) first: it has the current state of every run and what to do next.

## Hard rules (these have caused real damage before)

1. **Never reboot the PC, and never ask to,** unless Ryan is physically at it. He works remotely over
   Tailscale; if the PC goes down he can't bring it back.
2. **Never load a 70B model (~40 GB) on the PC.** llama3.3:70b OOM-killed the desktop once.
3. **Never edit a file a running job uses.** While `bin/run_all.sh` runs (tmux `plainboard`), don't
   touch `src/*`, `bin/run_all.sh`, `scripts/run_workers.py`, `scripts/generate_jobs.py` in the main
   checkout. New files and docs are fine. Never edit a bash script while it is running.
4. **Never `pkill -f` with a pattern that also matches your own ssh command.** Use PIDs or tmux.
5. **Never print, log or commit the Anthropic API key.** It lives in
   `~/.config/chess-llm-bench/anthropic.env` on the PC (chmod 600). Don't `cat` it.
6. **Ask before spending money** (API calls, cloud GPUs). The Claude budget is $22; $12.26 is spent.
7. **Don't merge `claude-models` into `ft-fixes`** (it edits `src/utils.py`) until the plain-board
   run is finished.

## How work is done here (from Ryan's CLAUDE.md)

- Order, no exceptions: **spec** (`specs/<name>.md`: goal, inputs/outputs, steps, edge cases,
  dependencies, reproducibility) → **tests** (pytest; happy path, edge cases, failures) → **code**
  → **commit and push**.
- Experiments with a verdict get their success criteria **in the spec, committed before results.**
- Commit messages: `<type>: <short description>` (types: feat, fix, test, spec, refactor, docs,
  chore), optional body, ending with a `Co-Authored-By:` line for the agent.
- No magic numbers (named constants), no hardcoded paths (config), no bare `except`, functions
  under ~40 lines, no dead code, seeds everywhere. No secrets in git.
- When you spot a problem, flag it as: **Code issue detected:** / **Problem:** / **Fix:**, then fix it.
- Before finishing: `venv/bin/python -m pytest -q -p no:cacheprovider tests fine-tuning/tests`
  (all green) and `venv/bin/ruff check src scripts fine-tuning tests` (clean).
- Numbers in `docs/FINDINGS.md` come from `scripts/render_findings.py`; run it with `--check`.
