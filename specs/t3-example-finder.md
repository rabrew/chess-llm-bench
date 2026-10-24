# Spec: T3 Phrasing Example Finder (T3-2) and Evidence-Based Synonyms (T3-3)

**Checklist items:** T3-2 and T3-3 in [exhibition-readiness.md](exhibition-readiness.md)
**Builds on:** [t3-whole-word-matching.md](t3-whole-word-matching.md)
**Created:** 2026-10-24

---

## Goal

The theme mark (T3 part 2) credits an explanation only if it names the theme in chess vocabulary
("fork", "pin"). A model that writes "attacks the king and rook at once" gets nothing.

Ryan's decision (2026-10-23): **add only phrases the models actually used**. To find them,
`scripts/find_t3_examples.py` scans saved explanations. For each candidate plain-English cue it
measures two things:

- **Gain:** how many *unmarked* explanations of that theme contain the cue. These are answers that
  would now earn the mark.
- **Leak:** how often the cue appears in explanations of *other* tactical themes. This is a
  false-positive proxy: a cue that shows up everywhere ("at the same time") would hand out marks
  regardless of the idea.

It also prints a sample of unmarked explanations per theme, to read by hand.

T3-3 adds a cue to `THEME_SYNONYMS` only when **all** of these hold:
- gain ≥ `MIN_GAIN` (20) on the real results
- leak rate ≤ `MAX_LEAK_RATE` (0.02)
- hand-reading of the examples confirms that it describes the theme

Each added phrase gets a positive test (a real model sentence) and a false-positive test.

## Inputs / Outputs

**Input:** one or more results JSONL files (`--results`, repeatable). Records need `theme` and
`t3_explanation`; any other record is skipped.

**Output:** a Markdown report (`--out`, default `results/t3_examples.md`) with:
1. Totals: explanations scanned, tactical, unmarked.
2. A **cue table**, one row per (theme, cue): `gain`, `leak_hits`, `leak_rate` (= leak hits ÷
   explanations of other tactical themes), and whether it passes the T3-3 thresholds.
3. Up to 3 example sentences per cue that passes, preferring different models.
4. Up to 15 unmarked explanations per tactical theme (seeded sample).

The script also returns the cue table as a DataFrame, for tests.

## Steps / Logic

1. Matching reuses the evaluator's matcher: `theme_mentioned` decides "marked", and the new shared
   helper `phrase_mentioned(phrase, text)` (the same whole-word rule, exported from
   `src/evaluator.py`) decides cue hits. No regex is copied into the script.
2. **Tactical** = the theme is not in `NON_TACTICAL` (advantage, crushing, equality, phase and length
   labels).
3. Candidate cues come from `IDEA_CUES`, the hand-written list of plain-English descriptions carried
   over from the 2026-10-20 scratch script. They are **candidates only**; the data decides what is
   kept.
4. Sampling uses `random.Random(42)`.

## Edge cases

| Case | Behaviour |
|---|---|
| A record without `theme` / `t3_explanation`, or with non-string values | Skipped |
| A malformed JSON line | Skipped, with a warning naming the file and line |
| A theme with no unmarked explanations | No section-4 entry; its cues have gain 0 |
| No other tactical explanations (leak denominator 0) | `leak_rate` NaN; fails the threshold |
| A results file that doesn't exist | Exit 1 with the path |

## Dependencies

pandas, plus `src.evaluator`.

## Verification (`tests/test_find_t3_examples.py`)

- An unmarked "fork" explanation containing "attacks two pieces at once" counts toward the gain for
  (fork, "attacks two"); a marked one (it says "fork") does not
- A cue present in other themes' explanations raises the leak hits and rate
- A cue with high leak fails the threshold even when the gain is high
- `phrase_mentioned` is whole-word: "pins" matches "pin", "opinion" does not
- Non-tactical themes are not scanned
- Malformed lines are skipped; a missing file exits 1
- The report file contains the cue table and the sample section
