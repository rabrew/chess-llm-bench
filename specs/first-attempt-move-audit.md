# First-attempt move audit
## Goal
Preserve first parsed moves and disclose rescue for future records, with an explicit switch for single-shot combined prompts.
## Inputs / Outputs
Input: evaluation.rescue_enabled (boolean, default true), model response, board.
Output: t2_move_attempt_1, t2_legal_attempt_1, t2_rescue_path, t2_rescue_enabled on new move-bearing records. Non-move formats use null first-attempt fields and not_applicable path.
## Steps / Logic
Capture the combined parser's move before any rescue. Label paths none, response_scan, reprompt, failed, disabled, legal_list_scan, or not_applicable. With rescue disabled retain the combined parser output and do not scan/re-prompt. Move-only uses its existing legal-list assisted parser and is explicitly labelled; this is not unassisted legality.
## Edge cases
Missing/illegal moves remain false first-attempt legality; failed rescue preserves the first attempt separately. Failed model calls retain existing failure handling. Invalid rescue config raises ValueError. Legacy records are never backfilled.
## Dependencies
python-chess; pytest mocks only, no engines or services.

## Denominator limitation
Results contain completed jobs only. Model-call errors and all-fields-missing responses that exhaust rescue still fail without a result row. Stored first-attempt legality is conditional on recorded completion and must not be presented as all-request legality; queue failure counts are needed for that denominator. move_only remains legal-list assisted even when rescue is disabled.
