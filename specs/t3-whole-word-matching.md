# Spec: T3 Whole-Word Theme Matching

**Closes:** T3-1 in [exhibition-readiness.md](exhibition-readiness.md)
**Created:** 2026-10-20

## Goal

The T3 "theme mentioned" point (P2) currently checks `phrase in explanation.lower()`.
That finds the phrase anywhere, including inside other words:

| Theme | Phrase | Wrongly matched by |
|---|---|---|
| `mate` | "mate" | "estimate", "approximately", "ultimately", "material" |
| `pin` | "pin" | "opinion" |
| `equality` | "even" | "eventually" |

"Material" is everywhere in chess explanations ("material advantage"), so `mate`-themed
puzzles hand out free P2 marks. Matching must be on whole words, while still accepting
ordinary word endings so "forks", "mated" and "attacking" still count.

## Inputs / Outputs

- **Input:** a Lichess theme label (`str`) and an explanation (`str`). Either may be `None`.
- **Output:** `theme_mentioned(theme, explanation) -> bool`. Callers turn it into 0/1.
- `score_t3` (live scoring) and `rescore_t3_theme` (post-processing of stored answers)
  both call this one function. No matching logic is copied between them.

## Steps / Logic

1. Candidate phrases, in this order, with duplicates and empties dropped:
   - `THEME_SYNONYMS[theme]`
   - `theme.lower()`
   - `_camel_to_words(theme)`
   - `theme.lower().replace("_", " ")`
2. A phrase matches when it appears with a word boundary on both sides. One optional
   ending from `s`, `es`, `d`, `ed`, `ing` is allowed before the closing boundary:
   `\b<phrase>(?:s|es|d|ed|ing)?\b`, case-insensitive. The phrase is escaped, so
   characters like `-` and `'` are literal.
3. Return `True` on the first matching phrase.

## Edge cases

| Case | Behaviour |
|---|---|
| `explanation` or `theme` is `None` or not a `str` | `score_t3` keeps its current `None` handling; `rescore_t3_theme` returns `None` |
| Phrase followed by punctuation ("Checkmate!", "pin.") | Matches (punctuation is a word boundary) |
| Multi-word and hyphenated phrases ("back rank", "back-rank mate", "x-ray") | Match as whole phrases |
| Ending already in the synonym list ("pinned", "sacrificing") | Still matches directly |
| Doubled-consonant endings not in the list ("pinning" is listed; "forked" fits the rule) | Only the listed endings are allowed, so no stemming library is needed |
| Unknown theme label | Falls back to the label itself and its camelCase split, as now |

## Dependencies

Python's `re` only.

## Verification (tests in `tests/test_t3_whole_word.py`)

- "estimate", "approximately", "ultimately" and "material" do **not** match `mate`
- "opinion" does not match `pin`, and "eventually" does not match `equality`
- "forks", "mated", "pins", "attacking the king" and "Checkmate!" do match their themes
- "back-rank mate" matches `backRankMate`, and "advanced pawn" matches `advancedPawn` (camelCase fallback)
- Matching is case-insensitive
- `score_t3` and `rescore_t3_theme` agree on every test case
- The existing tests in `tests/test_artefact_fixes.py` still pass
