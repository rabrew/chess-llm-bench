# Spec: T2 Move Source and First-Attempt Legality

**Builds on:** [t1-zero-baseline-and-ft-comparison.md](t1-zero-baseline-and-ft-comparison.md) §3–4, [retry-illegal-moves.md](retry-illegal-moves.md)
**Replaces:** checklist item T2-2 as first written (`exhibition-readiness.md`)
**Created:** 2026-10-24

---

## Goal

The published T2 legality is about 98% for most models. That figure is measured on the **final** move.
On the combined prompts (`fen_only`, `pgn+fen`, `cot`), the worker produces that final move like this:

1. Parse `Move:` from the reply.
2. If that move is missing or illegal, look for any legal move in the reply text (the text rescue).
3. If there is none, make a separate `move_only` call at temperature 0 (the isolated retry). That
   prompt **lists every legal move** and tells the model to pick one, so a retry answer measures
   copying from a list, not reading the board.

The base re-run (2026-10-24) saves raw replies, so step 1 can be scored. Only **7–13%** of first answers
on combined prompts were legal. The text rescue supplied about 2–5% more, and the isolated retry
supplied 83–91% of the final moves. The "98% legal" figure therefore measures legality after a re-ask.
It does not show that models can produce a legal move while answering the full question.

### The checklist premise was wrong

T2-2 assumed that a record with a stored illegal move predates the rescue code, and planned to split
records into `pre_rescue` and `post_rescue` by time. The data says otherwise:

- Every model in `results/evaluations.jsonl` stores illegal moves up to its last record. That includes
  records from 13–17 April, after the rescue was committed on 9 April.
- `move_only` has **zero** stored illegal moves, even on 25 March. The March-17 code kept the first word
  of the reply, so it would have stored some. The extract-a-legal-move code was therefore already
  running when the full benchmark started, before it was committed.
- The 9-April code kept the illegal move when both rescues failed. The 28-April code (`3e129c0`)
  blanks it. That is the only difference in move handling.
- The base re-run on the current code reproduces the original legality to within 0.3 points for every
  model and format, on 14–16k paired rows per model.

So there is no pre-rescue population. Every stored legality figure is after rescue, and the
cross-run comparisons (base vs fine-tuned, model vs model) compare like with like. Issue 18 is
corrected accordingly.

What the stored records cannot provide is first-attempt legality. That is only recoverable where
`raw_response` was saved: the base re-run and any later runs.

## Inputs / Outputs

**Input:** a `load_results_df()` DataFrame. The optional fields are `raw_response` and
`raw_move_retry_response` (see spec §3 of t1-zero-baseline). The required fields are `fen`,
`prompt_format` and `t2_legal`.

**New derived columns (in `load_results_df`):**

| Column | Values |
|---|---|
| `t2_move_source` | `first_answer`, `text_rescue`, `isolated_retry`, `unknown` on move-asking rows; NaN on `eval_only` / `explanation_only` |
| `t2_first_attempt_legal` | 1.0 / 0.0 where the source is known; NaN for `unknown` and non-move rows |

**New table:** `compute_move_source_breakdown(df)` returns one row per (model, prompt_format) for
move-asking formats with at least one known source:

| Column | Meaning |
|---|---|
| `model`, `prompt_format` | |
| `n` | rows with a known source |
| `n_unknown` | move-asking rows without a raw reply |
| `first_attempt_legal_rate` | mean `t2_first_attempt_legal` |
| `text_rescue_share`, `isolated_retry_share` | share of `n` whose final move came from that step |
| `final_legal_rate` | mean final `t2_legal` (missing = False) over the same `n` rows |

**Other outputs:**
- `save_metrics()` writes `t2_move_source.csv` through `_write_or_remove`.
- `summary.json["overall_metrics"]["t2_first_attempt_combined"]` holds the pooled
  `{n, first_attempt_legal_rate, final_legal_rate}` over combined prompts, or `None` when no combined
  row has a raw reply.
- `compare_finetuning_results.py` adds the metric `t2_first_attempt_legal_rate` (combined prompts). It
  is NaN for any side without raw replies, which currently means the fine-tuned run, so its delta is
  NaN rather than invented.

## Steps / Logic

For each move-asking row:

1. `move_only`: the reply is the only attempt, so the source is `first_answer` and first-attempt legal
   = `t2_legal`. Note that this prompt supplies the legal-move list, so `move_only` "first-attempt"
   legality measures picking from a list. Board-reading legality comes from the combined rows only
   (`t2_first_attempt_combined`). This holds with or without a raw reply. Note that the move is taken as any legal move
   found in the reply (`extract_move_from_text`), which is more lenient than `Move:` parsing.
2. Combined, no `raw_response` → `unknown`.
3. Combined, `parse_response(raw_response)["move"]` is legal on `fen` → `first_answer`.
4. Combined, otherwise, `raw_move_retry_response` present (an empty string counts) → `isolated_retry`.
5. Combined, otherwise → `text_rescue`. The worker skips the retry only when the text rescue found a
   legal move.

`t2_first_attempt_legal` is 1.0 only for `first_answer` rows with a legal final move. It is 0.0 for the
other known sources.

Legality is checked with python-chess (`parse_san` and membership in `legal_moves`), the same rule as
the worker. Invalid SAN or an invalid FEN counts as illegal.

## Edge cases

| Case | Behaviour |
|---|---|
| Old records with no raw replies | Combined rows are `unknown`; the table leaves the model out if no row is known |
| No `raw_response` column at all | Every combined row is `unknown`; no error |
| A retry call that failed (empty reply) | `raw_move_retry_response == ""` → `isolated_retry`, final illegal |
| A first answer in long-algebraic or junk text | Illegal under SAN parsing → falls through to rescue/retry, as in the worker |
| A null or malformed FEN | Treated as illegal, never an exception |
| Correction / control job rows | Labelled like any other row; the caller filters `job_type` |

## Dependencies

python-chess, pandas and numpy (already present). It reuses `parse_response` from `src/llm_client.py`
so the first answer is parsed exactly as the worker parses it.

## Reproducibility

The labels are a pure function of the stored record. No model calls are made.

## Verification (`tests/test_t2_move_source.py`)

- A legal first answer → `first_answer`, first-attempt legal = 1
- An illegal first answer with a retry reply → `isolated_retry`, first-attempt legal = 0, even when the
  final move is legal
- An illegal first answer, no retry, legal final → `text_rescue`
- An empty retry reply still counts as `isolated_retry`
- Combined without a raw reply → `unknown` and NaN
- `move_only` → `first_answer`, first-attempt legal follows `t2_legal`, with or without a raw reply
- `eval_only` / `explanation_only` → NaN for both columns
- A bad FEN does not raise
- The breakdown table has the right rates and shares, and leaves out a model with no known rows
- `save_metrics` writes `t2_move_source.csv`; `summary.json` has `t2_first_attempt_combined`
- The comparison reports `t2_first_attempt_legal_rate`, NaN on a side with no raw replies
