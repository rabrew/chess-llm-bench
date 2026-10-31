# Completion of experiments 6 and 7

## Goal

Complete collection, existing-response engine scoring, registered comparisons,
and source-driven reporting. The N1–N4 and C1–C4 criteria in the existing
experiment specifications are immutable. This document describes execution and
acceptance checks; it does not introduce a hypothesis or revise a threshold.

## Inputs and outputs

- Existing plain-board configuration, queue, append-only run log, and responses.
- Existing Claude configuration and collected responses in the separate
  `claude-models` worktree. Never access the API-key file for offline analysis.
- Existing Lc0 binary and weights at the configured 800-node search budget.
- Outputs: enriched response records, generated metric CSVs and provenance-bearing
  summaries, renderer-driven findings tables, and final write-up pages.

## Execution and dependencies

1. Review and commit the already-tested analysis/reliability corrections and
   conclusions snapshot before additional execution. Preserve untracked run logs.
2. Resume experiment 6 only after Ryan explicitly authorizes resumption. Use the
   handoff command and `>>`; exclude all 70B models. Freeze every file used by
   active workers until they finish. Never modify a running shell script.
3. Enrichment may run only when no Ollama model is loaded and no inference job
   is active. Check GPU state; retain a backup of the original response file.
   Claude's existing responses may be enriched while experiment 6 is paused.
   Use the corrected enrichment implementation without merging the branches.
4. Confirm enrichment preserves job IDs, original response text, model/prompt
   coverage, and record count. Every legal move must have finite nonnegative
   CPL and a stored search reference. Missing engine scores remain incomplete;
   do not manufacture scores or discard failures to satisfy completion.
5. Use the existing comparison criteria. Unrun models, missing required scores,
   or absent reference pairs remain pending. Disclose failed jobs and missing
   pairs separately from completion of a collection pass. Any repair of
   reporting completeness must not change statistical decision thresholds.
6. Implement Claude comparison through its own implementation specification,
   synthetic CPU tests, then code. Reuse position pairing and seeded bootstrap
   helpers; no inference or paid requests are part of comparison. C2/C3 are
   reported as `not run`, regardless of retained Sonnet pilot records.
7. Merge `claude-models` only after experiment 6 completes. Independently review
   changes, pass the full CPU suite, Ruff and renderer check, and commit/push
   completed steps. Fast-forward `master` only when it can be done without
   discarding other history. Remove the Claude worktree only after its work is
   merged and preserved.
8. Drive new findings tables from generated summaries. Final labels require
   actual completed results; keep pending/interim wording until then. State
   limited measured competence separately from interpretation about reasoning.

## Edge cases

Keep duplicate jobs, incomplete pairs, all-fields-missing replies, token-capped
responses, an absent local Gemma reference, unknown Claude parameter counts,
GPU-state query failures and engine failures explicit. A high absolute
plain-board correlation can trigger the registered major-barrier label even
when FEN scores higher; report this limitation without changing the rule.

## Reproducibility and acceptance

Existing seeds, sample, prompts and inference settings stay fixed. Summaries
record the configuration, code revision, input hashes and engine/weights
provenance. Validate record invariants before and after enrichment. Test new code
with temporary fixtures and mocked services before real-data execution. A paid
API submission, Sonnet expansion, 12B training run or resumption without Ryan's
instruction is outside the authorized execution scope.
