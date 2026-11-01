# Fail-closed Claude batch submission

## Problem and goal

Post-integration review identified a pre-existing crash window: the Claude
runner submits a paid batch before recording its provider ID durably. An
ambiguous timeout or crash can therefore cause duplicate submission on restart.
The SDK also defaults to automatic retries. Fix execution safety before any
future paid Claude run, without changing collected answers or C1–C4.

## Inputs, outputs and implementation

- Existing batch ledger, requests, injected/fake batch client and CLI.
- Preserve legacy valid ledgers and already-open batches. Do not rewrite
  research responses, costs or metrics during development.
- Hold an exclusive process file lock across execution preflight, credential
  loading, collection and submission. A concurrent invocation must stop before
  credential/API access; atomic file replacement alone does not serialize two
  processes' reservation updates. Read-only status/dry-run remain nonmutating.
- Persist a submission reservation with round/request identities **before**
  calling `batches.create`. Flush/fsync before sending. Only after acceptance
  may the ledger atomically replace the reservation with the returned batch ID.
- A timeout, exception or crash leaves an explicit unresolved reservation.
  Collection/submission must detect it and halt without another create call.
  Never automatically clear it or assume that an error implies no charge.
  Even errors known to be uncharged can wait for an operator's review.
- Existing open-batch collection remains resumable. A reservation is not a
  provider batch ID and must never be polled as one. Reject malformed state.
- Fsync saved raw responses (and the directory when first created) before
  durably removing a completed batch ID. Losing unflushed replies after the
  ledger removal must not create a paid resubmission path.
- Disable SDK automatic retries explicitly at construction. Use a small factory
  if needed so this is meaningfully testable with a fake SDK and no credentials.
- Keep error messages safe: do not print API exceptions, headers or key values.
  Unknown submission state is reported as requiring manual review.
- No new paid requests, credentials, models, engine or GPU calls are authorized
  by this change. Existing success criteria, prompts, prices and results stay
  unchanged. Tests use temporary ledger fixtures only.

## Tests before code and acceptance

Reproduce the current absent-reservation failure before fixing it. Cover durable
reservation observed by fake create, successful acceptance/collection, failed
create, accepted-then-failed persistence, restart blocking duplicate creation,
reservation rejection before provider polling, legacy/open-ledger compatibility,
invalid ledger state, and explicit zero SDK retries with a fake SDK constructor.
Also cover concurrent lock rejection before credentials/transport and durable
response persistence ordering before batch removal.
Only one author edits the Claude runner/module/tests. Independently review the
diff; full CPU tests, required Ruff and renderer checks must pass before commit.

## Reproducibility

Record this as an execution-safety correction, not a scientific results change.
Preserve prior collected batch IDs and raw replies. Future operators must resolve
unknown reservations against provider records before considering a resubmission.
