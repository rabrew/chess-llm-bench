# Handoff: state of the project on 31 Oct 2026 (late evening)

Written at the end of a long Claude Code session for whoever picks this up next (Codex). Rules for
agents are in [../AGENTS.md](../AGENTS.md). Read both before doing anything.

## 1. The project

**Ryan Brew, PR09426, "Do LLMs Reason? Benchmarking Logical Performance via Chess"**, for the Stripe
Young Scientist Award 2027 (supervisor contact: Dr Brew, who reviews the GitHub repo).

Open-source LLMs (22 models, 2B–70B, via Ollama) are asked about Lichess puzzle positions: T1
evaluate (centipawns), T2 best move, T3 who stands better and why. Ground truth is Lc0 (one network
pass for T1, 800 nodes for T2). Headline so far: models behave like pattern matchers. First answers
legal 9.3%; best move 1.16× random; blunder rate 41.9%; T1 ρ 0.11; T3 side κ 0.07; size doesn't help
except Gemma; fine-tuning taught board reading but not play. Full detail: `docs/FINDINGS.md`.

## 2. Machines and access

- **Ryan's Mac** (where the agent session runs) → **Linux PC** `rabrew@100.115.244.117` over Tailscale.
  Always: `ssh -o ConnectTimeout=60 -o ConnectionAttempts=3 -o BatchMode=yes rabrew@100.115.244.117`
  (add `-o ServerAliveInterval=30 -o ServerAliveCountMax=40` for long waits).
- PC: RTX 5080 16 GB, 32 GB RAM, Python 3.14 venv (`venv/` is a symlink).
- **Main checkout:** `~/Desktop/clb-analysis`, branch `ft-fixes` (= `master` on GitHub; keep master
  fast-forwarded to ft-fixes so Dr Brew sees the latest). Remote `origin` = `rabrew/chess-llm-bench` (public).
- **Second checkout (git worktree):** `~/clb-claude`, branch `claude-models` (Claude API runner and
  results). `data` and `venv` there are symlinks to the main checkout.
- Ollama runs on the PC; `run_all.sh` restarts it per model.

## 3. What is running right now

| tmux session | What | State |
|---|---|---|
| `plainboard` | `BENCH_CONFIG=config/config_plain_board.yaml SKIP_INSTALL=1 bin/run_all.sh > ~/plainboard.log` | Finishing codellama:34b (model 14 of 21) |
| `pausewatch` | `~/pause_after_codellama.sh`: sends Ctrl-C to `plainboard` when the log shows `MODEL: yi:34b` | Waiting; Ryan asked to pause after codellama (he may reboot) |

**When paused**, 14 models are done; 7 remain: yi:34b, command-r:35b, gemma4:26b, gemma4:31b,
mixtral:8x7b, deepseek-r1:7b, deepseek-r1:14b (~2 h each for the 30B+, ~11 h total). The pause
note is appended to `~/plainboard_pause.log`.

**To resume** (it skips finished tiers and resets stale jobs itself). Note `>>`, not `>`:
```
cd ~/Desktop/clb-analysis && tmux new -s plainboard2
BENCH_CONFIG=config/config_plain_board.yaml SKIP_INSTALL=1 bin/run_all.sh >> ~/plainboard.log 2>&1; echo "EXIT $?" >> ~/plainboard.log
```
After all models, `run_all.sh` runs step 7 (`retry_illegal_moves.py` → a separate `_retried.jsonl`),
then you run `enrich_cpl.py` and `generate_plots.py --save-metrics` (check the log; see section 6).

## 4. Experiments and their state

| # | Experiment | Spec | State |
|---|---|---|---|
| 1 | Base benchmark (22 models × 4,000 × 6 formats, 526,662 records) | `specs/benchmark.md` | Done, written up |
| 2 | Base re-run of 5 models with raw replies | `specs/t2-move-source.md` | Done |
| 3 | Pieces-only fine-tune (5 models) | `fine-tuning/specs/chess-board-finetuning.md` | Done |
| 4 | Pieces + legal-moves fine-tune | `fine-tuning/specs/legal-moves-finetuning.md` | Done; criterion 0 failed (training didn't take) |
| 5 | Engine-strength check (800 vs 8,000 nodes) | `specs/engine-strength-check.md` | Done: ρ 0.97, ranking holds |
| 6 | **Plain-board / notation check** | `specs/plain-board-prompts.md` | **Running** (14/21 models) |
| 7 | **Claude via Batch API** | `specs/commercial-models.md` (branch `claude-models`) | Haiku + Opus done; Sonnet deferred |
| — | GPT via Codex, 12 positions | `results/codex_illustration/README.md` (branch `claude-models`) | Done; **illustration only** |

### Experiment 6: plain-board (notation) check
- 500 positions (125/tier, seed 42, drawn per tier by run_all), 21 models (no 70B), formats
  `fen_only` (control), `board_text` (diagram + piece lists, no FEN), `board_text+legal`.
- Pre-registered criteria N1–N4 and the decision rule are in the spec; `scripts/plain_board_compare.py
  --config config/config_plain_board.yaml` computes them and writes `results/plain_board/metrics/`.
- **Interim (13 models, no CPL yet):** first-answer legal FEN 16.9% → plain 24.6% (CI 23.3–25.9; N4
  bar is < 25%, so likely to fail → verdict probably **mixed**); T1 ρ 0.10 → 0.14 (N2 holds);
  side κ 0.08 → 0.09 (N3 holds); N1 blunder rate needs CPL. Wrong-side illegal moves 14% → 8%.
  qwen2.5:32b ρ 0.08 → 0.33 on plain board; gemma3:12b 0.35 → 0.08.

### Experiment 7: Claude (branch `claude-models`, checkout `~/clb-claude`)
- Code: `src/claude_batch.py`, `scripts/run_claude_batch.py`, `config/config_claude.yaml`,
  `tests/test_claude_batch.py` (33 tests). Claude tags are in `API_MODEL_FAMILIES` in `src/utils.py`
  (size `None`, never guessed). Same 500 positions as the plain-board run (verified identical).
- Batch API (half price), resumable (`responses.jsonl` = raw replies + resume cache; `batches.json`
  = open batches), budget guard. Spent **$12.26 of $22**.
- Variants run: `claude-haiku:5.5`, `claude-opus:5.5-low` (all 3 formats, 1,500 each, 0 failures).
  Sonnet (`-nothink`, `-think`) only in the 24-reply pilot; **deferred by Ryan** (spec "Pilot result":
  thinking-off Sonnet still reasons in its reply; nothink cap raised to 3,000; re-send its 24 pilot
  replies when it runs). C2/C3 not tested yet.
- **Interim (no CPL yet), FEN prompt:** Opus first-answer legal 93.8%, T1 ρ 0.58 (0.50–0.65), side κ
  0.36; Haiku 88.8%, 0.49, 0.26. Best local so far: 38% legal, ρ 0.35, κ 0.16. Plain board doesn't
  help Claude (Opus ρ 0.58 → 0.53, κ 0.36 → 0.29), so C4: notation is not a barrier for Claude.
- C1 (Claude vs gemma4:31b blunder rate) needs CPL and gemma4:31b's plain-board run.
- Story change for the write-up: open models up to 70B show no reasoning signal; **frontier models
  show real but limited competence**. Contamination caveat (public puzzles) applies more to them.

### GPT illustration
- `codex exec` (CLI 0.166.0, Ryan's ChatGPT login, model reported `gpt-6.1-sol`) on 12 seeded
  positions (3/tier) with **every tool disabled** (first test tried to run Stockfish on the Mac).
  First answer legal 10/12, final 12/12, side 6/12. Too few positions for ρ. Label as illustrative.
  Bulk runs through a subscription are not OK (terms, agent wrapper, limits).

## 5. Write-up material

Page sources are in `docs/writeup/` (see its README for the claude.ai links). Ryan's own artifacts
on claude.ai can't be updated by Codex; edit the HTML in the repo instead.
- `draft.html`: full paper draft with references [1]–[27]; CHECK flags; §11.11 and §12.2 are
  placeholders for the plain-board result.
- `notes.html`, `conclusions.html` (10 conclusions), `notation.html` (experiment 6 mock write-up),
  `project.html` (whole-project mock write-up with chart), `readiness.html` (checklist),
  `scoring.html`, `plain-guide.html`.
- Conclusion 5 in `conclusions.html` (cross-task correlations) is exploratory, not pre-registered.

## 6. Next steps, in order

1. Confirm the pause happened (`~/plainboard_pause.log`, `tail ~/plainboard.log`), GPU free.
2. Optional while paused (Ryan was asked, no answer yet): run `scripts/enrich_cpl.py` on the
   Claude results (from `~/clb-claude`, `--config config/config_claude.yaml --results
   results/claude/evaluations.jsonl`) and on the partial plain-board results for an early blunder rate.
   Check `enrich_cpl.py --help` first. Lc0 uses the GPU: never while Ollama has a model loaded.
3. Resume the plain-board run when Ryan says (command above). Wait for `EXIT` in the log.
4. Then: `enrich_cpl.py` + `generate_plots.py --save-metrics` (plain-board config), then
   `plain_board_compare.py`. Report N1–N4 and the verdict exactly as the decision rule says.
5. Claude: enrich CPL, write `scripts/claude_compare.py` (spec → tests → code) applying C1 and C4
   exactly as written, reusing `plain_board_compare.py`'s pairing and bootstrap.
6. After the plain-board run: merge `claude-models` into `ft-fixes`, run all tests + ruff, push,
   fast-forward `master`. Remove the `~/clb-claude` worktree only after merging.
7. Update `docs/FINDINGS.md` (new sections for experiments 6 and 7, limitations), the write-up pages,
   the checklist (`specs/exhibition-readiness.md`), and the README experiments table.
8. Later, only if Ryan wants: Sonnet thinking on/off (~$10 left); a 12B fine-tune on the PC.

## 7. Ryan's own open items
R-3 review with Dr Brew; R-5; A-1 turn off Tailscale key expiry; A-2 off-PC backup. Emails to AI
companies for credits (Gary gave a free Codex subscription, not API credits).
