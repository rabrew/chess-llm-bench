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
unknown or unreliably billed outcomes retain the full reservation and block
repeats on resume. A completed, reliably metered response that fails semantic
parsing or refuses is unavailable with its measured charge retained; it is never
resent automatically.
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

## Frozen implementation clarifications

Candidate rubric identifier is exactly human_grounding_support_v2_candidate;
evaluate only matching-version24v2 references, never pool/relabel v1. All48viewed
cases remain development exposure. Generic prompts include central-error
precedence, critical omissions and justified slight advantages, without real
human examples. The selected judge shares a provider/family with two source
systems; author blinding does not eliminate self/style bias or make this an
independent gold standard. Preserve judge/human disagreement rather than tune it.

The API extraction schema allows BOTH citation offsets null. The deterministic
unique_verbatim_span_v1 binder fills them only for one exact occurrence, including
overlapping-match ambiguity checks. Explicit integers are never repaired; mixed
null/integer offsets fail. Raw extraction, normalized binding and strict helper
validation remain separate artifacts. A high judge grade with contradicted
checked evidence gets a derived review/tension flag without score replacement.

Budget-limited execution is case-by-case under a two-stage envelope. Freeze ALL
planned IDs even after a cap/unknown-response stop: not_run_budget, pending,
failed or unavailable carry null ratings. Unknown calls retain full reservation
(or larger recognized observed cost); reliable completed usage can be metered
even when the structured scientific answer is unavailable. No automated retries.
Run-directory and ledger locks prevent concurrent alternate-ledger duplicates;
pinned ledger path, manifest and replay hashes prevent lost-state budget reset.
Raw received responses persist before parsing, and every attempted stage keeps
its exact payload/schema. Replay is explicitly labelled and never incurs cost.

Safety constants name overhead/output/payload/price-age/margin/text bounds.
Evaluation reports human NA/null, judge NA/missing, full planned-case status
coverage, per-position grouping and separate components, never a composite.

## Infrastructure-only revision: semantic_v2

The first live pilot exposed character-offset failures despite exact unique source
quotes. This revision is based on infrastructure errors, not human quality labels.
Preserve all semantic_v1 run files and the exact extraction/judge prompts and API
schemas. Binder unique_exact_quote_primary_v2 may correct BOTH integer offsets
when the explicit slice is invalid/wrong only if the unmodified exact quote occurs
uniquely. Record original_start/end, bound span and explicit correction method.
Correct explicit spans still disambiguate repeated quotes. Unknown/ambiguous
quotes with incorrect spans, mixed null/integer or wrong-type offsets fail. No
fuzzy matching, Unicode normalization, quote rewriting or semantic relaxation.

Optional --reuse-ledger plus explicit --reuse-run reads immutable old live state.
Reuse only the same case ID/stage/full canonical payload fingerprint with reliable
received provider output, validated source-manifest link, unchanged neutral source,
prompts/schemas, rates and request settings. Reserved/blocked source requests are
never retried through cache misses. Pin source ledger and manifest byte hashes,
paths and response provenance in the new run; changes/loss fail closed. Read old
state without constructing a mutating ledger instance. Cache hits incur the same
actual conservative charge AGAIN in the new ledger. Require old conservative
spend plus new cap not exceed the old approved cap, so the approved USD3 total
cannot expand. Never alter/refund the source ledger or its grades.
