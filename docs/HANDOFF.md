# Handoff: current state on 1 November 2026

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
- **Claude worktree:** merged into `ft-fixes` and removed after verification.
  Private backup: `~/.local/share/chess-llm-bench/worktree-backups/clb-claude-20261101T164357Z.tar.gz`.
  Claude code/results are now in the main checkout; do not use the removed path.
- Ollama runs on the PC; `run_all.sh` restarts it per model.

## 3. Current runtime state

No benchmark, pilot or training job was started by this handover after the
following completed runs:

- Plain-board pipeline: terminal **EXIT 0**, finished **13:42:45 GMT** on
  1 November. All 21 model passes finished: **31,487 answers**, 13 missing
  records disclosed, all **30,625 legal moves** have finite nonnegative CPL.
- OpenAI setup pilot: **40/40 requests complete**, estimated **$0.5612788**.
  Sol: 17/20 legal first answers, two caps. Luna: 2/20 legal, 18 caps.
- GPT-5.4 campaign: **running in tmux `gpt54`**, authorized by Ryan after the
  eligibility/exclusive-pool explanation. First 20 requests were independently
  audited and committed; the 1,500-request plan is continuing until its rolling
  quota guard stops it. Provider free-credit application remains unverified.
- GPU power limit remains **250 W**, set with Ryan's approval. Do not raise it
  automatically. No reboot or 70B loading was performed.

The prior pause/resume instructions are historical: Ryan explicitly authorized
“cap and resume”; execution used append-only `~/plainboard.log`, not truncation.
The final comparison is committed and the formal verdict is **mixed**.

## 4. Experiments and their state

| # | Experiment | Spec | State |
|---|---|---|---|
| 1 | Base benchmark (22 models × 4,000 × 6 formats, 526,662 records) | `specs/benchmark.md` | Done, written up |
| 2 | Base re-run of 5 models with raw replies | `specs/t2-move-source.md` | Done |
| 3 | Pieces-only fine-tune (5 models) | `fine-tuning/specs/chess-board-finetuning.md` | Done |
| 4 | Pieces + legal-moves fine-tune | `fine-tuning/specs/legal-moves-finetuning.md` | Done; criterion 0 failed (training didn't take) |
| 5 | Engine-strength check (800 vs 8,000 nodes) | `specs/engine-strength-check.md` | Done: ρ 0.97, ranking holds |
| 6 | **Plain-board / notation check** | `specs/plain-board-prompts.md` | Done: 21 models; registered verdict **mixed** |
| 7 | **Claude via Batch API** | `specs/commercial-models.md` (branch `claude-models`) | Haiku + Opus done; Sonnet deferred |
| — | GPT via Codex, 12 positions | `results/codex_illustration/README.md` (branch `claude-models`) | Done; **illustration only** |

### Experiment 6: completed notation check

500 positions, 21 models, FEN/plain-board/plain-board+legal. Registered paired
FEN → board measures: blunders **42.2542% → 41.8761%**, evaluation correlation
**0.1214 → 0.1672**, side κ **0.0802 → 0.0886**, first-answer legality
**21.1861% → 25.5149%**. N1–N3 hold, N4 fails; exact verdict **mixed**.
See `results/plain_board/metrics/plain_board_summary.json`; it records missing
responses, measure-specific coverage, provenance and terminal completion evidence.

### Experiment 7: completed Haiku/Opus comparison

Haiku and Opus each have 1,500 answers, 500 positions × three formats. Sonnet's
48 pilot replies remain excluded from completed-variant comparisons; **C2/C3
are not run**. All 3,048 legal Claude/pilot moves have CPL.

C1 on 498 matched legal/scored FEN pairs: Gemma4:31b blunders 34.5382%, Haiku
28.9157%, Opus 19.2771%. Opus meets **“frontier model plays clearly better”**;
Haiku has overlapping intervals and **“no clear difference”**. Both C4 labels
are “notation was a major barrier” via the absolute-rho rule, despite paired
correlation declines; report that limitation. Existing ledger spend $12.2555896.
No new Claude API requests were made during scoring/comparison/hardening.

### GPT illustration
- `codex exec` (CLI 0.166.0, Ryan's ChatGPT login, model reported `gpt-6.1-sol`) on 12 seeded
  positions (3/tier) with **every tool disabled** (first test tried to run Stockfish on the Mac).
  First answer legal 10/12, final 12/12, side 6/12. Too few positions for ρ. Label as illustrative.
  Bulk runs through a subscription are not OK (terms, agent wrapper, limits).

## 5. Write-up material

Page sources are in `docs/writeup/` (see its README for the claude.ai links). Ryan's own artifacts
on claude.ai can't be updated by Codex; edit the HTML in the repo instead.
- `draft.html`: paper draft with references [1]–[27] and retained reference CHECK
  flags; §11.11/§12.2 contain final notation results and a Claude section is added.
- `conclusions.html`: eleven conclusions, including final notation and frontier
  results. `notation.html` and `project.html` use final E6/E7 values and caveats.
- Other supporting sources: `notes.html`, `readiness.html`, `scoring.html` and
  `plain-guide.html`; the final source updates do not constitute submission sign-off.
- Conclusion 5 in `conclusions.html` (cross-task correlations) is exploratory, not pre-registered.

## 6. Next steps

1. Monitor the current GPT-5.4 campaign via `results/openai_gpt54/summary.json`
   and `run.log` / tmux `gpt54`. It uses a 200k input+output-token rolling limit
   inside the user-reported 250k pool. Do not edit running code/config/outputs.
   Local accounting cannot independently verify invoice/free-credit application.
2. The runner stops on quota exhaustion and records earliest resume time; there
   is no automatic daily scheduler. Resume the same plan only once permitted by
   the quota guard. Unknown/ambiguous request reservations need manual review,
   never automatic retries. No additional paid Luna expansion is approved.
3. Confirm the raw-data publication/backup plan. Final plain-board raw JSONL and
   original run logs remain local under existing data-release policy; metrics,
   figures, comparison reports and OpenAI pilot outputs are committed.
4. Review the preserved original checkout's uncommitted user changes separately
   on `preserve/local-master-work-20261101`; do not discard or silently publish.
5. Sonnet C2/C3 or more training remain optional and require a new explicit
   approval/budget. Unknown Claude submission reservations require manual
   provider reconciliation before considering a resubmission.
6. Complete Ryan's own review/backup/setup tasks below. Publication pages are
   updated; keep future experiments, pilot evidence and registered results distinct.

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

## OpenAI pilot preparation (1 November 2026)

- Ryan stored an OpenAI credential outside the repository. Its contents were
  not inspected or displayed; file permissions are owner-only.
- `specs/openai-pilot.md` and `config/config_openai_pilot.yaml` define a setup
  pilot: GPT-6.1 Sol and GPT-6 Luna, 20 shared positions per model (5/tier),
  plain-board prompts, no tools, no rescue, API default reasoning/sampling.
- `venv/bin/python scripts/openai_pilot.py` is an offline-only plan. It does not
  read credentials, contact services or create output files. Current plan: 40
  requests, $1.1682 conservative reservation total within a $2 token-spend
  ceiling. This is not a guarantee about provider billing or tax.
- Paid submission requires explicit `--execute` and Ryan's spending approval;
  approval of the proposed $2 pilot is still pending. No OpenAI requests have
  been submitted. New outputs will be isolated under `results/openai_pilot/`.
- An ambiguous request, missing/empty established ledger, lost completed ledger
  entries or a changed plan must stop for review, never automatically resend.
  This pilot tests setup and cost; it supplies no population-level verdict.
- Separate collection audit: experiment 6's log ends in `EXIT 0` (13:42:45 GMT).
  All 21 models have outputs: 31,487 records, with 19 models at 1,500, one at
  1,491 and one at 1,496. All 30,625 legal moves have finite nonnegative CPL.
  Saved metrics exist; the final registered comparison/provenance audit and
  branch integration are still outstanding. Earlier running status is obsolete.


## Final experiment comparisons and OpenAI preparation (1 November 2026)

This supersedes earlier pending experiment 6/7 and running-pilot notes:

- E6 has terminal `EXIT 0`, 31,487 answers and 30,625 fully scored legal moves;
  registered final verdict **mixed** (N1–N3 hold, N4 fails). Summary/CSV at
  `c4541da`, other metrics and twelve plots at `35be11c`.
- C1 is complete on 498 pairs: Opus meets “frontier model plays clearly better”;
  Haiku has overlapping intervals and “no clear difference”. C4 retains its
  absolute-correlation interpretation limitation. C2/C3 are not run.
- Claude branch merged at `4686a2c` after E6; post-merge suite 1,286 passed,
  required Ruff/renderer clean. No new Claude calls. A separate paid-submission
  safety correction is in progress under `specs/claude-submission-safety.md`.
- Approved OpenAI pilot finished 40 requests, estimated $0.5612788; replies and
  ledger are committed (`77f67fb`). Sol 17/20 legal, two caps; Luna 2/20 legal,
  18 caps. Treat as setup/cost testing, not a clean ability ranking.
- User's latest next-model choice: GPT-5.4, retain Luna, exclude mini/nano.
  New daily runner `scripts/openai_daily_benchmark.py` is prepared and tested
  (`5b83402`), default offline, 1,500 requests, 200k rolling-token ceiling.
  No GPT-5.4 campaign started: project free entitlement and exclusive shared-pool
  usage confirmation remain pending. Sol/Luna are not in the pasted free list.
- All current work is consolidated in `docs/PROJECT_RECORD.md`.
- The original master checkout's uncommitted edits remain unchanged on local
  `preserve/local-master-work-20261101`. Do not discard or publish them as part
  of the current completed research work. Master update/worktree cleanup must
  be recorded after final checks, not assumed complete from this note.


## Final verified integration state (1 November 2026)

The current sections above supersede historical appended snapshots. Claude
submission safety is complete (`ed6c4d4`): durable reservation before paidcreate,
no SDK retries, exclusive CLI lock, durable replies before ledger removal.
Independent review approved; **1,308 CPU tests passed**, required Ruff and
renderer checks clean. No live Claude submissions were used for validation.

`master` has been fast-forwarded and pushed; final publication commits are kept
synchronized with `ft-fixes`. Original local uncommitted edits remain unchanged
on `preserve/local-master-work-20261101`. The merged Claude worktree has been
removed only after a verified private backup at the path in section 2.

Final findings, simple summary, conclusions and four HTML write-up sources are
updated. Browser rendering was not checked. Raw final plain-board JSONL, retry
output and original run logs remain local; formal metrics/plots/comparisons and
OpenAI pilot replies are committed. GPT-5.4 remains prepared, not started.


## GPT-5.4 execution update (1 November 2026)

Ryan said “yes go with gpt 5.4” after the pending eligibility/exclusive-pool
explanation. Execution used the explicit operator-confirmation flags; billing
entitlement itself has not been queried or independently verified.

First 20 requests: 7 FEN, 7 plain board, 6 plain+legal, 14 legal first moves,
no caps/refusals/missing fields, 7,675 input+output tokens. Returned model was
`gpt-5.4-2026-03-05`. Estimated uncached list price $0.02985, distinct from
actual free-credit application. Checkpoint is committed at `68374c5`.

Continuation is running in tmux `gpt54`, sequential requests to the frozen
500-position/three-format plan, no tools or rescue. It stops before the next
reservation would exceed 200,000 tokens in a rolling 24 hours. Only completed
checkpoints are committed; live response/ledger exports will differ in the
working tree until the next safe checkpoint. The GPU remains idle/capped250W.
Engine scoring is pending a finished snapshot; never enrich live campaign files.
