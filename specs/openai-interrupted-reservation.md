# Explicit recovery of a shutdown-interrupted reservation

## Goal

Resume the frozen main benchmark after the host powers off during an uncertain
paid request. Keep its full reservation, retain its identity, and never resend it.
An unobserved interrupted request is not evidence that the model answered wrong.

## Inputs / Outputs

Existing frozen source/settings/requests and shared ledger. Explicit offline
`--phase main --abandon-interrupted` changes only non-pilot reserved entries into
interrupted_unknown, preserving charge and request. Raw response and observed
legality remain null. Main summary reports interrupted_unknown separately.

## Logic

Use the existing exclusive ledger lock. Refuse pilot reservations, existing
responses, changed payloads/settings/source, stopped errors and unknown runner
fingerprints. Reuse known-version migration snapshots and append policy/time/
previous-status/IDs and old/new hashes. Allow exact384825b runner fingerprints
in addition to prior known truncation versions. Keep budgets and input/output
limits unchanged. Main skips the abandoned paid identity; future ambiguous
requests still block until a separate explicit maintenance decision.

## Edge cases and tests

Successful8-request pilot then a reserved main request; recovery makes no calls,
retains full charge, and main issues only the remaining7 requests. Repeat
maintenance is idempotent. Pilot reservation is refused. Existing stopped error
is not abandoned. Interrupted result stores no response/legality. A held lock
prevents maintenance. Existing truncation migration crash tests continue to pass.

## Dependencies

Existing filelock and atomic state migration; no new packages or API access.

## Unobserved transport outcome extension

The same explicit offline abandon command may accept a non-pilot main entry
whose prior status is stopped, exact outcome stopped_ambiguous, raw_response
and usage are absent/null, parse_quality unavailable and no parsed/scored model
observations exist. Keep its entire existing charge and paid identity. Preserve
prior result metadata/timestamp and journal the actual previous status/outcome.
Use unobserved_request_v1 policy; no shutdown cause is assumed. Known provider
bodies, refusals, parsed observations, other stopped outcomes remain rejected.
Allow exact8bb5dbc module fingerprints through the existing migration protocol.
