# Fine-tuning audit remediation
## Goal
Preserve saved training artifacts and make base versus fine-tuned comparisons represent matched observations and available measurements.
## Inputs / Outputs
Input: normal result records (t1_direction_correct), explicit T0 board-description records, recorded T2 reference endpoints, training output paths.
Output: matched cohort deltas with coverage counts and per-task sample sizes; missing values stay null; board precision, recall, F1 and exactness; persisted held-out responses.
## Steps / Logic
Join normalized model, position_id, prompt_format, attempt/trial, job_type; exclude nonstandard jobs and reject duplicate keys or conflicting FENs. Use only matched records. Apply prompt task masks. Compute CPL only with the shared recorded-endpoint helper, never cap loss at 1000 or infer endpoints from T1. Board legacy accuracy remains piece recall; held-out headline uses F1 and records exact correctness. Failure cleanup preserves every checkpoint/export; refuse accidental reuse for fresh runs.
## Edge cases
No common cohort, partial coverage, duplicate observations, missing/invalid schema, failed move responses, non-move prompts, missing T0, hallucinated pieces, zero measured score, failed resume/export, existing output paths.
## Dependencies
python-chess, pytest with all model/trainer calls mocked. No training, engines, or Ollama for checks.
## Reproducibility
Existing seed 42 retained. Matched keys sorted deterministically; no research inputs regenerated.

## Additional preservation guards
A requested benchmark-exclusion file must exist and contain valid FEN-bearing JSON records; corrupt or missing exclusions fail closed. GGUF import rejects ambiguous exports across either supported directory instead of selecting an arbitrary model.

CPL deltas require matching recorded reference engine, search budget, units, and scoring version on both members of each pair. Numerical endpoint aggregates alone do not establish comparable engine provenance. Per-task paired counts expose unavailable cohorts.

## Legality comparability
Paired legality deltas require compatible recorded assistance policy. Combined formats require explicit boolean rescue-enabled metadata on both records with equal values; successful or failed rescue routes under the same enabled policy remain comparable. Move-only requires the recorded legal-list-scan path on both records and is labelled legal-list assisted independently of the rescue setting. Unknown historical assistance or differing policies yield unavailable paired legality, zero eligible pairs where all are excluded, and explicit exclusion reasons. Eval/explanation-only remain inapplicable.
