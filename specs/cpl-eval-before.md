# Spec: Store the 800-Node "Before" Eval Used for CPL (T2-6)

**Builds on:** [cpl-enrichment.md](cpl-enrichment.md), [metric-defect-remediation.md](metric-defect-remediation.md) (D5)
**Checklist item:** T2-6 in [exhibition-readiness.md](exhibition-readiness.md)
**Created:** 2026-10-24

---

## Goal

`scripts/enrich_cpl.py` computes the raw centipawn loss as:

    t2_cpl = Lc0-800(before) − Lc0-800(after)   (from the mover's side)

It evaluates "before" at 800 nodes, but then **throws that value away**. `load_results_df` later
rebuilds "after" for the clamped CPL and win-probability loss as `t1_stockfish_eval ∓ t2_cpl`.
`t1_stockfish_eval` is the **no-search** network reading of the starting position, which is a
different number.

The puzzle positions are where this hurts. The 800-node search finds the winning tactic, so "before"
is often a mate score (about ±10,000). The no-search reading is close to level instead: its median
|value| on these rows is 43 cp. So the rebuilt "after" is pushed about 10,000 cp too far, and the
clamped CPL goes over its intended 1,000 cap. In the base re-run, the 90th-percentile clamped CPL is
1,101.

**Fix:**
1. Store the 800-node "before" eval on every enriched record.
2. Backfill it for existing results by re-evaluating only the unique starting positions. There are
   at most 4,000 per file; post-move positions are not re-searched.
3. Rebuild "after" from that stored value.

## Inputs / Outputs

**Record fields (new, optional for old readers):**

| Field | Meaning |
|---|---|
| `t2_eval_before` | Lc0 eval of the starting position (White's POV, cp), at `t2_eval_nodes` nodes |
| `t2_eval_nodes` | node budget used, e.g. 800 |

**`enrich_cpl.py`:**
- New enrichments write both fields alongside `t2_cpl` and `t2_best_move`.
- `--backfill-eval-before` fills both fields on records that have a `t2_cpl` but no
  `t2_eval_before`. It evaluates each unique FEN once and does not touch post-move positions or
  existing `t2_cpl` values.
- The write is atomic: a temp file, then a replace, as now.
- Before overwriting, it copies the original to `<results>.bak-<UTC timestamp>`, unless
  `--no-backup` is given.

**`load_results_df`:**
- `t2_cpl_clamped` and `t2_wp_loss` use `t2_eval_before` as "before" when present.
- Otherwise they fall back to `t1_stockfish_eval`, with **one** logged warning giving the number of
  rows that used the fallback.
- A new column `t2_eval_before_source` records which one was used: `lc0_search`, `no_search`, or
  NaN when there's no CPL.

## Steps / Logic

1. Refactor pass 1 of `enrich_records` into `evaluate_unique_fens(fens, engine) -> dict[fen, result | None]`
   so that both enrichment and backfill use it.
2. `enrich_records` sets `t2_eval_before = pre["eval"]` and `t2_eval_nodes = engine.nodes` on every
   record whose pre-move eval succeeded.
3. `backfill_eval_before(records, engine)`:
   - Select records with `t2_cpl` not None and `t2_eval_before` missing.
   - Evaluate their unique FENs.
   - Fill both fields, returning new dicts without mutating the input.
   - A FEN whose evaluation failed is left without the field, and the failure is counted and logged.
4. In `load_results_df`, "before" is `t2_eval_before` if it is not NaN, else `t1_stockfish_eval`.
   "After" is `before − cpl` with White to move, else `before + cpl`. These go through
   `compute_clamped_cpl` / `compute_wp_loss` unchanged.

**A known limit, stated in the write-up:** Lc0 on a GPU is not bit-for-bit deterministic. The
backfilled "before" can differ by a few cp from the value used when `t2_cpl` was computed. That is far
smaller than the no-search gap this fixes. New enrichments store the exact value, so they don't have
this limit.

## Edge cases

| Case | Behaviour |
|---|---|
| A record with `t2_cpl` but an illegal or missing move | Not possible (CPL is only set on legal moves); the backfill still only needs the FEN |
| `t2_cpl == 0` | The backfill still stores "before"; "after" = "before"; the clamped CPL is 0 |
| An engine failure on a FEN | The field is left missing; `load_results_df` falls back and counts it |
| The results path is a symlink (`results/evaluations.jsonl` on the PC points into the original checkout) | Resolve it first, so the target file is updated and the link is kept; the backup sits next to the target |
| The backfill is re-run | Records that already have `t2_eval_before` are skipped (idempotent) |
| Old results files never backfilled | They behave as today, plus one warning |
| A mate-score "before" (±10,000) | Clamped to ±1,000 by `compute_clamped_cpl`, so the clamped CPL is ≤ 2,000 by construction and ≤ 1,000 whenever "before" and "after" have the same sign |

## Dependencies

The Lc0 binary and weights from `config.lc0` (already used by `enrich_cpl.py`), plus the RTX 5080.
No LLM calls.

## Reproducibility

The node budget is recorded on every record (`t2_eval_nodes`). The weights file is the one in
`config.lc0.weights`; P-2 records its hash in `summary.json`.

## Verification

**`tests/test_enrich_cpl.py`:**
- Enrichment stores `t2_eval_before` and `t2_eval_nodes`
- The backfill fills only records with a CPL and no "before", evaluates each FEN once, leaves
  `t2_cpl` unchanged, and is idempotent
- An engine failure leaves the field missing and doesn't raise
- `run(..., backfill=True)` writes a `.bak-*` copy first

**`tests/test_cpl_eval_before.py` (`load_results_df`):**
- With `t2_eval_before` present, the clamped CPL is computed from it (a mate-score "before" plus a
  large raw CPL gives clamped ≤ 1,000 when the sign doesn't flip)
- Without it, the clamped CPL matches the old formula and `t2_eval_before_source == "no_search"`
- A row with no CPL has a NaN source
