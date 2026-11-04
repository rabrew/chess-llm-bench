# Terminal truncation recovery without repeat charges

## Goal

Continue the frozen main benchmark after a definitive max-output truncation,
counting that paid first attempt once as a failed response. Preserve USD20/
USD1 budgets, prompts, models, output cap, sample and original full reservation.

## Inputs / Outputs

Inputs: frozen manifest, original durable request and raw API body, existing
shared ledger. Output: distinct terminal_failure status, truncated outcome and
failure summary; no new request for the failed identity. Existing stopped entries
can be reconciled only by explicit main-only offline maintenance command.

## Logic

Require provider status incomplete, reason max_output_tokens, nonempty response
ID, matching requested/recognized snapshot model, default service tier, valid
internally consistent usage within input/output/financial bounds, and no error
or refusal. Preserve raw body and partial parse, mark scientific outcome failed.
Retain at least the full original reserved charge. Missing/ambiguous usage,
transport failures, model/tier mismatch and other incompletes remain blocking.
Pilot still requires successful completed parses before main promotion.

Migration accepts only the exact known original runner/budget fingerprints and
current unchanged prompt/evaluator/source/config/frozen requests. Preserve original
manifest and ledger snapshots, append old/new fingerprint and policy provenance,
use the same ledger lock, and change no payload, identity, cap or charge. Save
ledger first, then upgraded manifest; interrupted maintenance is idempotent or
fails closed. No API key or transport access is needed for maintenance.

## Edge cases / tests

Cover a stopped main truncation, future main truncation, main resume skip without
repayment, pilot terminal block, complete-looking partial text still failed,
wrong reason/model/tier/usage/refusal, altered original code or source/payload,
unchanged charges and budgets, and restart after partial migration writes.
All fixtures use temporary files and mocked transport; agent never mutates the
real experiment state or invokes paid requests.

## Dependencies

Existing requests/filelock/atomic JSON helpers; no new packages or services.
