# GPT-5.4 benchmark within a reported daily free allowance

## Goal and scope

Ryan requested GPT-5.4 after the existing Sol/Luna setup pilot, retained Luna,
and excluded mini/nano. Prepare a fixed, resumable **GPT-5.4-only** benchmark on
the existing 500-position cohort, using FEN, plain-board and plain-board+legal
prompts (1,500 first answers). The current paid pilot remains separate; this
does not authorize more paid Luna calls or another model sweep.

This is descriptive benchmarking, not a new registered superiority verdict.
Do not alter N1–N4/C1–C4 or pool first answers with assisted final responses.
Keep raw answers, illegal/missing answers, refusals and truncation visible.

## Inputs, outputs and eligibility

- Read-only source: experiment 6 responses, all 500 positions, 125 per tier.
  Require complete metadata, consistent FEN/reference and exactly the configured
  fixed cohort. Use existing prompt/parser/scoring helpers with engine=None.
- New config and runner; isolated `results/openai_gpt54/` plan, fsynced event
  ledger, raw responses, evaluations and summary. Never touch other research
  outputs, queues or model/runtime files. Reuse safe pilot helpers where useful.
- Request model **gpt-5.4**, no tools, store:false, no reasoning/sampling override,
  max_output_tokens:4000, no recovery/re-ask/automatic HTTP retries. Record
  returned model, usage, defaults and response status; parameter size unknown.
- The 250,000-token daily allowance is user-reported, shared across eligible
  larger models. Project enrolment/external usage/overage terms remain unverified.
  Default execution is prohibited. Paid pilot approval is not permission to
  assume this campaign is free.
- Real submission requires --execute, --confirm-free-project and
  --exclusive-quota-use. These are explicit operator confirmations, not an API
  entitlement check. If another app uses the shared pool, do not use this mode.
  Never report invoice cost as zero based solely on local token accounting.

## Ordering, limits and persistence

1. Commit spec before tests/code. Work only in new scripts/config/tests; do not
   change the still-running pilot or modules it imports.
2. Validate finite positive integers/limits, exact allowed model/formats/prices,
   nonempty paths and seed. Default offline invocation reads no secrets, makes
   no network calls, and creates/modifies no files.
3. Deterministically shuffle within each tier (seed42), interleave tiers, then
   emit all three formats per position. Freeze selected FEN/truth, prompts,
   settings, code/source hashes in a plan. No selection based on observed scores.
4. Enforce at most **200,000 total input+output tokens in any rolling 24 hours**,
   including reasoning within output exactly once. This conservative reserve
   below the reported 250k pool does not measure other applications' usage.
   Configured rolling limit may be smaller, never above 200k.
5. Before each call, reserve UTF-8 prompt bytes +4096 input-framing tokens + the
   entire output cap. Completed calls consume actual input+output usage; do not
   double-count reasoning. Timestamp reservations and completions in UTC; retain
   completed usage until 24h after completion. Unknown requests retain their
   full reservation and block further execution pending review. Reject unsafe
   future timestamps/clock rollback rather than reset allowance.
6. Stop before the next request when remaining allowance cannot cover its
   reservation. Save status `quota_exhausted` and earliest safe resume time.
   No persistent scheduler is installed. Re-running later continues the same
   plan without paying for completed requests again. An optional per-invocation
   --max-new-requests bound supports an initial eligibility/cost check.
7. Reuse the pilot's exclusive file locking, durable events and fail-closed
   pattern. Missing/empty established ledger, malformed/out-of-order events,
   duplicate IDs, lost completed transactions, ambiguous transport failure or
   missing/invalid token usage must halt before further calls or export loss.
   Bind state to the request plan; ordinary documentation commits may change
   informational git revision but must not silently change request identity.
8. Record uncached list-price estimates separately from billing entitlement:
   GPT-5.4 input $2.50, output $15.00 per million tokens at standard short-context
   rates. No billable tool use/caching writes. Plan and summary explicitly say
   invoice/free-credit application is not verified by the runner.
9. Export first-answer legality and missing/capped/refused counts, usage totals,
   estimated list price, per-format coverage and quota status. CPL remains
   pending separate engine scoring. Do not modify active campaign exports with
   an enrichment process; score a finished snapshot separately.

## Tests before code and acceptance

Temporary synthetic source data, fake time and injected transport only. Cover
balanced deterministic order/position identity, exact three prompts and no-tool
requests, config/NaN/bool/invalid model/price rejection, dry-run no writes/key/
network, explicit eligibility gates, first-answer/refusal/cap scoring, rolling
quota accounting at boundaries, no reasoning double-count, reservation safety,
one-call oversize denial, future/rollback clocks, no resend after successful
restart, missing/empty/valid-prefix ledger loss, duplicate/corrupt events,
unknown usage/timeout fail-closed, process locking and immutable source data.

Full CPU suite, affected regressions, required Ruff and independent final review
must pass. No real requests while implementation/testing; no dependency
installation or GPU activity. Document limitations honestly. No criteria are
changed after seeing answers. Initial execution waits for Ryan to confirm the
eligible project and absence of other shared-pool usage.

Official API references checked 1 November 2026:
- https://developers.openai.com/api/docs/models/gpt-5.4
- https://developers.openai.com/api/docs/guides/token-counting
