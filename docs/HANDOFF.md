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
| `plainboard` | Plain-board run using `config/config_plain_board.yaml` | Paused after codellama:34b (14 of 21 models); do not resume without Ryan |
| `pausewatch` | `~/pause_after_codellama.sh` | Pause recorded at 23:00:09 GMT on 31 Oct 2026 |

**Paused state:** 14 models have finished their collection pass; 7 remain: yi:34b, command-r:35b, gemma4:26b, gemma4:31b,
mixtral:8x7b, deepseek-r1:7b, deepseek-r1:14b (~2 h each for the 30B+, ~11 h total). The pause
note is appended to `~/plainboard_pause.log`.

There are 20,991 saved plain-board responses: 13 models have 1,500 each and
WizardLM has 1,491. All CPL values are still null. Completion of a model's
collection pass therefore does not imply a complete analysis cohort.

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
  help Claude descriptively (Opus ρ 0.58 → 0.53, κ 0.36 → 0.29). **C4 is pending:**
  CPL is absent and the registered rule has an interpretation limitation: its
  absolute board-text ρ ≥ 0.30 branch would label this a "major barrier" despite
  a decline from FEN. Keep the rule unchanged and report that limitation next to
  the descriptive paired changes; do not claim a completed C4 result.
- C1 (Claude vs gemma4:31b blunder rate) needs CPL and gemma4:31b's plain-board run.
- Story change for the write-up: the tested local models show weak and unreliable
  chess competence; **frontier results provisionally suggest stronger competence**.
  These measures do not identify an internal reasoning mechanism. Contamination
  caveat (public puzzles) applies more to frontier models.

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

## Post-handoff correction notes (2026-10-31)

- The plain-board run remains paused after codellama; do not resume without Ryan.
- Analysis fixes follow `specs/analysis-review-corrections.md`. Historical CSVs,
  JSON summaries, responses, and datasets have not been regenerated or modified.
  Existing blunder CIs are row-based and need an explicitly requested rebuild
  before they can be described as position-cluster intervals.
- No-CPL/partial analysis must report pending measures and a pending verdict.
  Report measure-specific coverage, per-model correlations and legal-output
  selection alongside the registered pooled measures. Do not change N/C criteria.
- Inspect `bin/run_all.sh` before resumption: it currently invokes enrichment
  and plotting automatically after retries. Enrichment checks Ollama's loaded
  model status and refuses to launch Lc0 if models are loaded or state is unknown;
  it does not stop or unload anything. Arrange the GPU handover explicitly.
- Haiku has 41 token-capped initial replies among 1,500 saved records. "Zero
  failures" means no missing scored records, not an absence of truncated replies.


## Completion progress (1 November 2026)

- Plain-board remains paused at 14/21, awaiting Ryan's resume instruction.
- Claude enrichment is complete: all 3,048 legal moves have 800-node Lc0 CPL;
  original responses were preserved. No new API requests or charges.
- `claude-models` contains tested `scripts/claude_compare.py` and committed
  JSON/CSV in `results/claude/metrics/` (results commit `1da3d14`). C1 is pending
  Gemma4:31b; C2/C3 are not run. Both C4 verdicts are “notation was a major
  barrier” under the unchanged absolute-correlation rule, despite decreases in
  paired T1 correlation. This supersedes earlier no-CPL/pending-C4 notes above.
- Main includes optional experiment table rendering and a read-only Ollama idle
  wait before automatic enrichment. Successful completion requires terminal
  `EXIT 0` in the log. Do not merge branches until experiment 6 finishes.
- Full CPU checks: main 1,164 passed; Claude 1,105 passed, one existing missing
  fixture skip. Required Ruff and main renderer consistency checks passed.
- Updated numerical results and limitations are in `docs/CONCLUSIONS.md`.


## Overnight resumption (2026-11-01T00:41:20+00:00)

Ryan explicitly authorized “cap and resume”. The GPU power limit was changed
from 360 W to 250 W and verified. Experiment 6 resumed in tmux `plainboard2`
from the main checkout, with `SKIP_INSTALL=1` and append-only output to
`~/plainboard.log`; the wrapper appends `EXIT <status>` on termination.
The earlier paused status is superseded. No inference settings or registered
criteria changed. The limit remains 250 W until explicitly restored; do not
automatically raise it overnight. Runtime source files are now frozen while
the job runs. Successful collection/scoring is not yet established.
