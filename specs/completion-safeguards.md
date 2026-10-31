# Completion safeguards for the paused plain-board run

## Goal

Allow the existing pipeline to hand the GPU to Lc0 safely and distinguish a
finished collection with documented failed jobs from a genuinely partial run.
No prompt, sample, model setting, registered statistic or N1–N4 threshold changes.

## Inputs and outputs

Inputs are the selected configuration, Ollama's read-only loaded-model status,
existing response records and an optional pipeline completion log. Outputs are
an idle/failure exit status and explicit completion provenance in the comparison
summary. Do not stop/unload models, start services, resume collection, or spend
money as a side effect of checking readiness.

## Steps

1. Before pipeline enrichment, poll Ollama `/api/ps` until its model list is
   empty. Use a bounded configurable timeout, poll interval and request timeout.
   A transient unavailable/malformed status is unknown, never proof of idle.
   Fail clearly when the deadline expires. Do not query an inference endpoint.
2. The comparison keeps its existing strict expected-cohort check by default.
   The experiment specification explicitly permits failed/missing pairs, while
   the pipeline considers approximately 1,500 responses per model complete.
   An explicit `--completion-log` can establish finished collection: the last
   `EXIT` marker must be zero, every expected model and format must have records,
   and no unexpected model is accepted. Required measurements must still exist.
3. When that evidence exists, missing individual records do not force an
   otherwise defined registered verdict to stay pending forever. Preserve the
   strict cohort counts/status, expose the completion evidence and input-log
   hash, and disclose missing records/pairs. Never turn missing CPL into a zero
   loss, relax statistical thresholds, or treat a stopped/failed run as done.
4. Without valid completion evidence or with an absent model/format, retain the
   pending verdict. Undefined registered measures remain pending regardless of
   collection completion. Renderers report the stored outcome and limitations.

## Tests before implementation

Use fake clocks/HTTP status responses and temporary logs/records. Cover busy to
idle, unavailable/malformed state, timeout, and no inference or unload request;
check pipeline ordering without executing it. Cover failed and absent EXIT
markers, a later nonzero marker, complete-with-missing-pairs, absent models or
formats, missing scores and unchanged complete-data verdicts. Tests must not
touch real jobs, engines, services, datasets or research results.

## Reproducibility

The deadline constants and overrides are explicit. Completion evidence is stored
with the existing config/code/results provenance. This is execution/reporting
logic implementing existing missing-pair rules, not a post-results criterion.
