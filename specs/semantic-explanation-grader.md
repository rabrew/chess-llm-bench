# Two-stage grounded explanation judge — development harness

## Goal

Provide a concrete replayable semantic-judge pipeline with separate factual
 grounding and explanatory support outputs. It is an experimental scorer,
not a validated replacement for human assessment. The48already-viewed human
cases are development data, never a newly claimed held-out validation set.

## Inputs / Outputs

Prepare a frozen manifest from neutral JSONL: exactly case_id, fen, explanation,
availability. Do not send provider/model identity, old scores, theme, ratings or
notes to either model. Labels enter only a separate evaluation command after
grades are durably frozen. Store source/prompt/schema/code/settings hashes,
raw stage responses/usage/resolved judge model and separate dimension grades.
Missing/unavailable answers remain unavailable, never zero. New output paths
and isolated ledger must be external, symlink-free, and immutable on resume.

## Logic

Stage1 strict structured output extracts atomic claims and explicit root/branch
contexts, no quality grades. Every claim/move binds its exact source quote/span.
Contexts carry parent IDs, source-cited sequential SAN moves and explicit
interpretation assumptions. Unknown/invalid contexts abstain. No invented
move prefixes; a source token alone does not prove its inferred branch meaning.
Claims: occupancy, material counts, legality, check, checkmate, strategic. All
fields required and no unknown properties. Reject bad quotes or fields locally.

Python-chess verifies root inventory/side/check/legal SAN, and extracted claims
under explicit sequential contexts. SAN check/mate annotations are checked
separately; branches are not tested against the initial board. Extraction and
context assumptions stay distinct from verified facts. FEN alone does not prove
forced draw, repetition history, winning evaluation or explanatory sufficiency.

Stage2 receives only neutral FEN/snippet, extraction and deterministic evidence,
plus the same versioned rubric for all answers. Separate grounding/support
ratings0/1/2/NA, brief reasons and uncertainty flags; no combined quality sum.
Treat model snippets as untrusted quoted data, never operational instructions.
Both stages use strict JSON schema plus strict local validation. Refusal,
truncation, invalid JSON/schema/extraction or unknown billing yields unavailable,
never a fabricated zero. Keep raw responses and record failure status.

## Live cost and execution controls

Default prepare/replay/plan never reads API credentials or makes HTTP calls.
Live requires --execute and an explicit price configuration (model, positive
rates, official URL, verified date within48hours, caps, reasoning effort, margin
and output token limits). Parent-reviewed initial candidate is gpt-6.1-sol,
USD2input/USD10output per million, verified2026-11-09. No rates are hardcoded in
implementation. Responses payload uses text.format json_schema/strict,
default service tier, storefalse, no tools, retries or redirects.

One isolated durable stage ledger under an exclusive file lock reserves integer
nano-USD before every stage. Bound FULL serialized payload bytes including schema,
evidence and messages, plus8192overhead, output cap including reasoning and
minimum1.25margin. Charge measured usage only when model/tier/usage reconcile;
unknown/failed outcomes retain full reservation and block repeats on resume.
Stage2 depends on source-bound extraction; no automatic retries or regeneration.
Persist full response before parsing so crashes cannot rebuy a paid stage.
Preflight remaining case's two-stage conservative envelope before its first call.
No actual paid calls are authorized by implementation/testing alone.

## Evaluation and edge cases

Freeze grade outputs first. Evaluate a separate matching-version human JSONL by
case ID, rejecting duplicate/missing/foreign IDs and incompatible rubric versions.
Report dimension-specific coverage, exact agreement, MAE and confusion plus
per-position grouping, not a composite. NA/missing remain explicit. No v1/v2
pooling or fresh held-out claims. Same-case/different-model answers share a board
and are not independent positions.

Tests cover replay end-to-end, strict schemas/quotes/contexts, prompt injection
as data, separate dimensions, missing answers, refusal/truncation, stage ledger
reservation-before-transport, crash-safe/no-duplicate resume, budget exhaustion,
full-payload bounds, model/tier/usage mismatch, corrupt state, and label leakage.

## Dependencies

Existing requests/filelock/python-chess and standard library. Reuse audited
atomic money/usage/path helpers and fixed HTTPS transport without changing them.
No engines, GPU, Ollama or new training/benchmark workloads; fixtures only in tests.
