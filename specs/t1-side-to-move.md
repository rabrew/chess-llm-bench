# Spec: T1 Split by Side to Move (Perspective Check, T1-3)

**Checklist item:** T1-3 in [exhibition-readiness.md](exhibition-readiness.md) (D2)
**Created:** 2026-10-24

---

## Goal

A model that reports the eval from the side to move's point of view, instead of White's, agrees with
the engine when White is to move and is inverted when Black is to move. Pooled over both sides, that
looks like noise. FINDINGS calls Llama-3.3-70B "below random" on direction, so the perspective
explanation has to be ruled in or out before that claim is used.

**Result on 2026-10-24 (526k base records):** no model flips. Every model's ρ has the same sign and a
similar size on both sides. It is about 0.19 for gemma4:26b and 31b, and |ρ| < 0.07 elsewhere,
including Llama-3.3-70B at −0.016 (Black) and −0.001 (White). Its poor direction score is noise, not
inversion.

**Limit:** the base runs saved no raw replies, so the 50-reply hand-check of the sign is possible only
for models in the base re-run, none of which are the 70B model. This limit is stated in the write-up.

## Inputs / Outputs

**Input:** a `load_results_df()` DataFrame with `fen` (or `position_id` resolved to a FEN),
`t1_model_eval` and `t1_stockfish_eval`.

**`compute_t1_by_side_to_move(df)`:** one row per model with:
- `n_white`, `n_black`
- `spearman_white`, `spearman_black`
- `kappa100_white`, `kappa100_black` (Cohen's κ at ±100 cp)
- `perspective_flip_suspect`: True when `spearman_white` and `spearman_black` have opposite signs and
  both have |ρ| ≥ `PERSPECTIVE_RHO_MIN` (0.1)

**Output:** `save_metrics()` writes `t1_by_side_to_move.csv`.

## Steps / Logic

1. Side to move comes from the FEN, using `_white_to_move_from_fen`.
2. Rows with an unparseable FEN are dropped.
3. ρ and κ come from the existing `compute_eval_rank_correlation` and `compute_direction_agreement`,
   applied to each side's rows, so there are no new formulas.
4. A model missing from one side gets NaN for that side and is not flagged.

## Edge cases

| Case | Behaviour |
|---|---|
| A model with rows on one side only | The other side is NaN; not flagged |
| Constant evals on one side | ρ is NaN; not flagged |
| No valid T1 rows | Empty table; any stale CSV is removed |

## Dependencies

pandas, python-chess.

## Verification (`tests/test_t1_side_to_move.py`)

- A model whose answers are the truth from White's view is not flagged, with ρ = 1 on both sides
- A model that negates the truth when Black is to move is flagged (ρ_black = −1)
- A model with only White-to-move rows has NaN Black columns and is not flagged
- `save_metrics` writes `t1_by_side_to_move.csv`
