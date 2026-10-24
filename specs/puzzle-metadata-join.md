# Spec: Join Lichess Puzzle Metadata Back to the Benchmark Positions (T3-4, Issue 21)

**Checklist item:** T3-4 in [exhibition-readiness.md](exhibition-readiness.md); evidence for Issue 21
**Created:** 2026-10-24

---

## Goal

`dataset_builder.py` kept only `themes[0]` from each Lichess puzzle, and it stored the CSV `FEN`
unchanged. Two problems follow.

1. **Primary theme only (T3-4).** For about two thirds of records, `themes[0]` is `crushing` or
   `advantage`, so the "idea" mark only checks for words like "winning".
2. **The position shown is the setup position (Issue 21).** In the Lichess CSV, `FEN` is the position
   *before* the opponent's move. The puzzle starts after `Moves[0]`, and its themes and rating describe
   that later position, for the other side.
   - Checked on 2026-10-24: all 4,000 benchmark FENs appear in the CSV, and `Moves[0]` is legal in
     every one, so the side to move in the shown position is the one making the setup move.
   - T1, T2 and the T3 side mark are scored against the engine on the shown position, so they are
     valid.
   - The T3 theme mark and the difficulty tiers (puzzle rating) describe the next position.

This spec joins the CSV back on FEN, without re-running any model, so that the analysis can:
- report the theme mark for tactical themes only, using the full theme list (T3-4)
- measure how often a model's move **is the human setup move** (`Moves[0]`), a direct pattern-matching
  signal: the setup move is usually the human mistake that allowed the tactic
- carry `puzzle_fen` (the position after `Moves[0]`), so a corrected re-run can use the right
  positions later

## Inputs / Outputs

**Script `scripts/build_puzzle_meta.py`:**
- **Input:** `data/lichess_puzzles.csv.zst` (columns `PuzzleId, FEN, Moves, Rating, Themes, …`) and
  the benchmark FENs, taken from one or more results files (`--results`, repeatable).
- **Output:** `data/puzzle_meta.json`, a mapping from FEN to:

| Field | Meaning |
|---|---|
| `puzzle_id` | Lichess id |
| `themes_all` | the full theme list |
| `setup_move_uci`, `setup_move_san` | `Moves[0]`, the move the side shown to move plays in the real game |
| `puzzle_fen` | FEN after `Moves[0]` |
| `rating` | puzzle rating |

**`load_results_df` (when `<data_dir>/puzzle_meta.json` exists):**

| Column | Meaning |
|---|---|
| `themes_all` | list, or NaN |
| `setup_move_san` | |
| `t2_played_setup_move` | on rows with a legal move: 1.0 if `t2_move == setup_move_san`, else 0.0; NaN otherwise |
| `t3_tactical_theme_correct` | on rows with an explanation and at least one tactical theme in `themes_all`: 1.0 if the explanation names **any** tactical theme in the list (`theme_mentioned`), else 0.0; NaN otherwise |

**Metrics:**
- `compute_puzzle_meta_metrics(df)` gives per model: `n_moves`, `setup_move_rate`,
  `n_tactical_explanations` and `tactical_theme_rate`.
- `save_metrics` writes `puzzle_meta_metrics.csv`.
- The summary key `puzzle_meta` holds the pooled values, or `None` when there is no metadata.

## Steps / Logic

1. The script collects the unique FENs from the results, then streams the CSV once
   (`zstandard` + `csv.DictReader`), keeping rows whose `FEN` is in that set.
2. The SAN of `Moves[0]` comes from python-chess on the shown FEN.
3. The non-tactical labels (evaluation, phase and length labels such as `crushing`, `endgame`, `long`)
   live in one constant, `NON_TACTICAL_THEMES` in `src/evaluator.py`, which both this code and
   `scripts/find_t3_examples.py` import. Mate themes (`mate`, `mateIn2`, …) count as tactical.
4. The join is by exact FEN string.

## Edge cases

| Case | Behaviour |
|---|---|
| A benchmark FEN not in the CSV | Left out of the meta file; the script logs the count. **The run fails if coverage is below 99%** |
| A FEN that appears twice in the CSV | Keep the first; log the count |
| `Moves[0]` illegal on the FEN | Kept with `setup_move_san = None`; logged (expected 0) |
| No `puzzle_meta.json` | No new columns; metrics are empty and the summary key is `None` |
| A row with an illegal or missing move | `t2_played_setup_move` NaN (the model gave no move to compare) |

## Dependencies

`zstandard` (already in `requirements.txt`; it was missing from the PC venv and was installed
2026-10-24), python-chess, pandas.

## Reproducibility

The output is a pure function of the CSV and the results FENs, and the CSV path is recorded in the
file's `_source` entry.

## Verification (`tests/test_puzzle_meta.py`)

- `build_meta` on a small in-memory CSV: matched FENs only, the full theme list, correct setup SAN and
  puzzle FEN; duplicates keep the first
- Coverage below 99% raises; a missing CSV exits 1
- `load_results_df` with a meta file adds `t2_played_setup_move` (1 / 0 / NaN cases) and
  `t3_tactical_theme_correct` (credits a tactical theme anywhere in the list, ignores
  `crushing`/`advantage`)
- Without a meta file, the columns are absent and the summary key is `None`
- The per-model metrics table has the right rates
