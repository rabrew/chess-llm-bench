# Actual puzzle position follow-up (local models only)

## Goal and scope, fixed before new answers

The original benchmark displayed the setup position one move before each
Lichess puzzle. This exploratory follow-up tests first-move chess performance
on the actual solver position. It does not replace prior experiments, change
N1–N4/C1–C4, or establish an internal reasoning mechanism.

Purposively chosen contrasting local models: **qwen2.5:7b, qwen2.5:32b,
gemma4:31b**. Existing results motivated selection; these are not a random
sample of LLMs. Exactly **100 positions (25 per rating tier), seed42**, drawn
from the existing 500-position cohort before any new answers. Two prompts,
`fen_only` and `board_text`: **600 initial requests**. No legal-list rescue,
text-wide move recovery or retries of model answers.

This is descriptive/exploratory: fixed estimates and uncertainty below, no new
binary success verdict and no opportunistic selection based on observed scores.
Any broader/new experiment requires its own recorded design.

## Inputs and outputs

- Read-only experiment6 responses and data/puzzle_meta.json. Metadata maps setup
  FEN to puzzle_id,setup move,puzzle_fen,rating and full theme list.
- Stream the existing compressed Lichess CSV only for selected puzzle IDs;
  `Moves[0]` is setup and `Moves[1]` the first solver move. Do not rebuild data/.
- Isolated config/manifest/positions, references, resumable raw/evaluation ledger,
  descriptive reports under results/actual_puzzle_followup/. Never reuse an old
  queue, result path, checkpoint or campaign output.

## Data validation and reference generation

1. Deterministically sample25/tier from sorted parent position IDs (seed42),
   retain parent/setup/puzzle IDs, setup FEN, actual FEN and metadata. Derive a
   stable actual-FEN ID; assert no collisions,duplicate FENs or puzzle IDs.
2. Replay the legal setup move; its resulting FEN must equal puzzle_fen. Verify
   the selected CSV row and full solution continuation are legal, and the first
   solver move exists in actual-position legal moves. Fail closed on missing or
   conflicting targets/metadata. Store full themes and source hashes.
3. Recompute T1 on actual FENs with the existing ONNX network encoder and
   white-POV conversion (same reference method as base). Never inherit a setup
   position's eval. Recompute Lc0 searched T2 reference at800nodes separately.
4. Verify Ollama has no loaded model before ONNX/Lc0 work. Serialize all GPU
   tasks and release reference-engine resources before local inference. Keep
   the250W power cap; do not restart/stop services or load70B models.

## Inference and scoring

5. New isolated serial local runner, no API credentials/cloudcalls/training.
   Existing build_prompt/parse_response/score_all/build_result_record should be
   reused where compatible. Retain exact raw Ollama reply,done_reason,usage,
   options,elapsed time and errors; no automatic alternative-answer recovery.
6. Use the existing model-specific Ollama options, with fixed sampling seed42,
   num_thread4 for modest CPU contention and keep_alive30s. Preserve API-default
   sampling temperature by omitting overrides. Record all requested/returned
   settings; this follow-up is not a controlled rerun of historical settings.
7. Process one model at a time; wait read-only for natural expiry before next
   model/engine phase. No unload requests,signals,tmux manipulation of other
   jobs or service restarts. Do not run alongside memory-heavy base reanalysis.
8. Durable reservation before each local request and append-only saved reply;
   explicit errors/unknown outcomes stay visible rather than being resubmitted
   automatically. Lock execution, freeze manifest/options and resume completed
   IDs without duplication. Missing/partial state fails closed. Dry-run writes
   nothing and makes no network/model/engine request.
9. CPL enrichment happens on a finished answer file with the existing800node
   reference, after Ollama naturally expires. Preserve original response fields.
   Raw answers and failed/illegal attempts stay in all-request denominators.

## Fixed analysis and interpretation

Report per model/format, with1000position bootstrap resamples,seed42:
- first-answer legality, completion/truncation/parse failure and score coverage;
- first solver move match versus mean per-position1/legal-move-count baseline
  (not inverse mean move count); this is first-move accuracy, not full solving;
- legal/scored CPL/blunder rates and legal-nonblunder success per all requests;
- T1 correlation/error, zero baseline, white/black-to-move associations;
- side agreement plus majority and **side-to-move-is-better** baselines. Actual
  puzzle targets are often favourable to the solver, so direction can be easy;
- full themes and any tactical-theme keyword match after excluding non-tactical
  tags. This remains a keyword proxy, not evidence of explanation soundness;
- paired FEN-versus-board effects within this new cohort and descriptive
  matched first-answer setup controls from experiment6. Board position/turn
  changed, so setup-versus-puzzle differences are not a causal notation effect.

Report all chosen models/positions and missingness. Do not stop early based on
results, tune criteria after outcomes, pool setup/actual-FEN reference values,
claim “solved puzzles” from a first move, or infer absence/presence of reasoning.

## Tests before code and acceptance

Temporary fixtures,fake ONNX/engine/HTTP only. Cover seeded balanced sampling,
FEN/ID/solution validation,black-POV conversion,correct actual references, no
mutation of data/original outputs, two prompt formats,no rescue/noAPI endpoints,
serial requests/settings/cap capture, locking/resume/malformed ledger/unknown
request handling, safe response failures, no model load during engine scoring,
and toy-baseline/bootstrap computations. Independent review and full CPU tests
must pass before reference generation or any local model calls. Plan/manifest
is committed before new answers. Persist results and provenance after each phase.
