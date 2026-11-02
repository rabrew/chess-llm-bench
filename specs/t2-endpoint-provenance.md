# T2 endpoint evidence
## Goal
Retain same-engine before/after evaluations for exact derived T2 metrics.
## Inputs / Outputs
Lc0 enrichment results; output t2_reference_eval_before/after, t2_reference_engine, t2_reference_budget, t2_reference_units, t2_scoring_version.
## Steps / Logic
Write both endpoints only after a successful pair. For an engine-selected best move retain the legacy zero-loss convention explicitly, without inventing a measured endpoint; exact endpoint metrics stay unavailable. Keep raw stored CPL for compatibility. Writer copies explicitly provided provenance fields. Legacy records cannot be reconstructed from T1 truth and CPL.
## Edge cases
Failed before/after evaluation leaves endpoint pair unavailable. Engine runtime version is unknown unless supplied; class identity and configured nodes describe the invocation, not historical engine versions. No historical data is modified.
## Dependencies
Mocked engine, pytest; no live enrichment for validation.

Future endpoint-quality summaries include explicitly labelled, validated engine-best-move zero conventions in their denominator and report their count separately from measured endpoint pairs. This convention is not a measured after-move evaluation. Historical records without either form remain unavailable.
