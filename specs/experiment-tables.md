# Experiment 6, Claude and GPT illustration tables

## Goal

Extend the existing findings renderer so experiment summaries, rather than
hand-entered numbers, supply the new write-up tables. This is reporting work;
the registered N1–N4 and C1–C4 rules remain unchanged. Missing or incomplete
results must remain visibly pending and must not acquire a final verdict.

## Inputs and outputs

Keep all existing `Sources` fields and table names compatible. Add optional JSON
summary inputs, each overridable by a CLI argument:

- `--plain-summary`: `results/plain_board/metrics/plain_board_summary.json`.
- `--claude-summary`: `results/claude/metrics/claude_summary.json`.
- `--gpt-summary`: `results/codex_illustration/summary.json`.

The renderer reads only these summaries and its existing metric inputs. It makes
no model, engine, API or network call. A missing optional summary produces the
existing unavailable-input notice. `--check` remains read-only and returns a
nonzero exit status when a selected document contains stale generated blocks.

New generated table markers are:

| Marker | Content |
|---|---|
| `notation` | N1–N4, paired counts, FEN/plain values and marginal intervals, registered thresholds, exact stored verdict and cohort status |
| `notation_secondary` | Plain versus plain-plus-legal first-answer legality and blunder rate, with counts and intervals |
| `claude` | Full variant and prompt-format descriptions: first-answer legality, T1 rho, side kappa, best move versus random and recorded variant cost |
| `claude_criteria` | C1 and C4 status/verdict per variant; C2/C3 explicitly not run; preserve stored outcome text |
| `gpt_illustration` | Small-sample counts only, prominently labelled illustrative; no population inference or comparison verdict |

## Steps

1. Read optional JSON only when present. Never infer missing results from the
   handoff narrative or substitute provisional values for absent summaries.
2. Render plain-board primary and secondary measures from the existing summary
   dictionaries. State N1–N4's fixed thresholds descriptively; do not recalculate
   the decision rule or alter its outcome. Unknown/nonfinite values render as
   an em dash, including missing confidence limits.
3. Render the Claude comparison summary produced by `claude_compare.py`. Include
   missingness/status, pilot exclusion and unknown model sizes as limitations,
   without pooling Sonnet pilots into full-variant results. Variant cost comes
   from the recorded-cost output, never a new price lookup or API request.
4. Render GPT counts only when a structured illustration summary exists. A
   12-position illustration is not a benchmark and cannot support rho, scaling
   or population claims. Keep its method/model label supplied by the summary.
5. Escape Markdown table cell separators, HTML-sensitive text and line breaks
   from supplied labels/verdicts so summary text cannot break document structure.
6. Extend the existing `render_all` mapping and CLI source construction, keeping
   all current blocks and defaults unchanged. Actual write-up editing and
   production rendering are the lead's responsibility.

## Edge cases and acceptance tests

Write synthetic tests before implementation. Cover complete and pending
plain-board inputs, missing CPL/intervals, missing optional summaries, exact
stored verdict text, Claude full variants and not-run criteria, null costs,
best-move random ratios, escaped labels, nonfinite values, and tiny GPT counts.
Exercise CLI overrides, stale detection and byte-for-byte no-write `--check`.
Existing renderer tests must remain green; no research artifacts are rewritten
by tests. Use temporary paths only.

## Dependencies and reproducibility

Use existing stdlib/pandas dependencies. The renderer reports analysis outputs
and their provenance; it does not change bootstrap seeds, resample counts,
cohort membership or registered success criteria. Summaries absent from the
current branch remain unavailable until the branch/data workflow supplies them.
