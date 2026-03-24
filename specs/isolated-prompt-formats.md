# Isolated Prompt Formats

## Goal
Add three task-isolated prompt formats (`eval_only`, `move_only`, `explanation_only`) so
each benchmark task can be measured without interference from the other two.  The combined
prompts (`fen_only`, `pgn+fen`, `cot`) remain for direct comparison.

## Motivation
Current results show near-random T1 direction accuracy and <26% legal move rate even for
the best models.  Because all three tasks share a single prompt, it is impossible to tell
whether a model fails T2 because it cannot generate legal chess moves *or* because the
combined prompt confuses it.  Isolated prompts give a clean signal per capability.

## Inputs / Outputs

| Format | Input | Expected output |
|---|---|---|
| `eval_only` | FEN (+ optional PGN) | Single integer (centipawns from White's perspective) |
| `move_only` | FEN | Single SAN move token |
| `explanation_only` | FEN (+ optional PGN) | `Explanation: <White/Black/Equal> — <one sentence>` |

For `eval_only` and `explanation_only`, PGN history is included when available (same rule
as `pgn+fen`).  For `move_only` the existing `build_move_prompt` + `MOVE_SYSTEM_PROMPT`
is reused directly.

## Steps / Logic

### `llm_client.py`
- Add `EVAL_SYSTEM_PROMPT` — instructs model to output only an integer.
- Add `EXPLANATION_SYSTEM_PROMPT` — instructs model to output only the Explanation line.
- Add `build_eval_prompt(fen, pgn_moves)` — minimal eval-only user prompt.
- Add `build_explanation_prompt(fen, pgn_moves)` — minimal explanation-only user prompt.
- Add `parse_eval_response(text)` — extracts first integer from raw response.
- Add `parse_explanation_response(text)` — reuses existing explanation-parsing logic from
  `parse_response`, returns only `explanation` and `side_claimed` fields.

### `worker.py`
- Detect `prompt_format` before building the prompt.
- Isolated path: send the appropriate prompt with its system prompt, parse only the
  relevant field(s), leave the other task fields as `None`.  Skip the combined-prompt
  illegal-move retry for isolated formats.
- Combined path: unchanged existing logic.

### `config/config.yaml`
- Extend `prompt_formats` list with `eval_only`, `move_only`, `explanation_only`.
- Total jobs per model/tier: 6 formats × 1 000 positions = 6 000 (up from 3 000).

## Edge Cases
- `eval_only` response contains prose instead of a number → `t1_model_eval: null`, job
  still succeeds (not failed).
- `move_only` response is multi-word or illegal → stored as-is; `t2_legal: false`.
- `explanation_only` response missing the `Explanation:` prefix → fallback regex in
  `parse_explanation_response` still attempts extraction.
- Scoring functions already handle `None` inputs gracefully — no changes needed to
  `evaluator.py`.

## Dependencies
- `python-chess` — legal move checking (existing)
- No new libraries required
