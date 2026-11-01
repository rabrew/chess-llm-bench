# OpenAI API setup and cost pilot

## Goal and scientific scope

Prepare a small, independently resumable API pilot before choosing a larger
commercial-model experiment. This is a feasibility/cost test, not a registered
comparison verdict. Do not change N1–N4 or C1–C4 or claim a 20-position pilot
establishes superiority or reasoning. Expansion requires a separate decision.

## Inputs and outputs

- Existing experiment 6 JSONL, read only; never load the large dataset tiers.
- Configured models `gpt-6.1-sol` and `gpt-6-luna`, 5 positions per tier (20
  positions each), `board_text`, seed 42, API default reasoning/sampling.
- A credential in `OPENAI_API_KEY` or the configured private file outside git.
  Dry runs never read the credential or contact any service.
- New output directory `results/openai_pilot/`: immutable plan/provenance,
  append-only request ledger and raw Responses API replies, evaluations JSONL,
  and descriptive pilot summary. Existing results and jobs remain untouched.

## Steps and execution policy

1. Validate configuration before writes or credential access. Models are limited
   to the two priced IDs; counts, timeouts, output limits and USD budget must be
   finite and positive. Default ceiling is $2 across this pilot, not per process.
2. Validate complete JSONL lines, required position metadata and consistent FEN
   and reference evaluation across repeated records. Select exactly 5 positions
   from each of easy/medium/hard/extreme, deterministically with seed 42, from
   the existing 500-position cohort. Never silently substitute another sample.
3. Reuse `build_prompt`, `parse_response`, `score_all(engine=None)` and existing
   result record fields. Explicitly mark model family OpenAI and size unknown.
   Retain every first reply, including refusals, empty and token-capped replies.
   No rescue, text-wide recovery, retry, tools, engine, GPU or Ollama requests.
   This first-answer pilot is distinct from assisted final-move comparisons.
4. Requests use HTTPS `https://api.openai.com/v1/responses`, sequentially, with
   `tools: []`, `store: false`, `max_output_tokens: 4000`. Do not set temperature
   or reasoning effort: record that API defaults were used. No automatic HTTP
   retries. A standard-library HTTPS transport avoids modifying dependencies.
5. Before any request, reserve a conservative input/output cost using configured
   standard rates (Sol $2/$10, Luna $0.10/$0.50 per million input/output tokens).
   Input allowance uses UTF-8 prompt bytes plus 4096 tokens of framing margin;
   output allowance uses the full configured token cap. Reserve an extra 10%
   margin and fail closed if remaining ceiling cannot cover the request.
   Actual recorded cost includes all output tokens (including reasoning), uses
   uncached input pricing conservatively, and never subtracts uncertain charges.
   This is a local submission guard, not a guarantee about provider billing;
   account hard limits remain an independent control.
6. Persist/fsync a reservation before sending, then append/fsync the successful
   raw response with usage, request identity, returned model and cost. A crash,
   timeout, missing/invalid usage or ambiguous HTTP failure leaves a reservation
   outstanding: stop and require review rather than silently resending. Resume
   must recognize cached responses and never duplicate paid requests.
7. A process file lock prevents concurrent writers. Bind the output to a plan
   hash covering selected positions/FEN/truth, prompt text, models, prices and
   settings. Refuse mismatched plans, malformed ledgers and duplicate identities.
   Raw credentials and authorization headers must never appear in logs, plans,
   exceptions or outputs. API failures print only safe class/status information.
8. Default invocation prints an offline plan and worst-case reservation total.
   Only an explicit `--execute` may load credentials or submit paid requests,
   and the lead must obtain Ryan's spending approval before using that flag.
   Dry run must make no output directories or files. Existing work is preserved.
9. Summary: requested/completed/capped/refused/missing parsed fields, first-answer
   legality, finite evaluation pairs, token totals and conservative recorded USD.
   CPL and best-move superiority remain pending; no engine is launched by this
   pilot. Engine scoring can happen separately once the GPU is available.

## Tests before implementation

Use temporary synthetic response fixtures and injected/fake transports. Cover
deterministic balanced sampling, matching prompts, no-tool request shape,
first-answer scoring, refusals and incomplete replies, invalid/nonfinite config,
missing tiers/inconsistent references, budgets/reservations across restarts,
unknown usage, duplicate/corrupt ledger, crash/ambiguous-request handling,
exclusive execution, unchanged source data, dry-run no secrets/network/writes,
secret-redacted errors, and plan mismatch. No paid calls in automated tests.

## Reproducibility and acceptance

Commit this specification before tests/code/results. Record config, source hash,
selected positions, request hashes, code revision, sampling seed, API-returned
model/status, usage/reasoning tokens, and truncation. Save pilot replies for
offline rescoring. Independent review, full CPU tests and required Ruff must
pass before an approved paid pilot is submitted. Existing runtime code and
registered experiment criteria are outside this implementation's scope.

Official references checked 1 November 2026:
- https://developers.openai.com/api/docs/models/gpt-6.1-sol
- https://developers.openai.com/api/docs/guides/reasoning
- https://developers.openai.com/api/docs/pricing
