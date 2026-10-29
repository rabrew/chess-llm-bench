# Spec: Engine-Strength Sensitivity Check (T2-5)

**Builds on:** [cpl-eval-before.md](cpl-eval-before.md), [t2-blunder-rate.md](t2-blunder-rate.md)
**Checklist item:** T2-5 (D7, optional)
**Created:** 2026-10-29, before any results

---

## Goal

Move quality (CPL, the blunder rate) is measured against Lc0 at 800 nodes. A stronger search could
judge some moves differently. This check re-scores a sample of the models' legal moves with Lc0 at
10× the nodes and asks: **does the model ranking by blunder rate stay the same?**

## Success criterion (fixed before running)

- **Ranking holds** when the Spearman ρ between the 22 base models' blunder rates at 800 nodes
  (as stored) and at 8,000 nodes is **≥ 0.9**.
- Also reported, not judged: each model's rank shift, and the overall blunder rate at
  each strength.
- **Noise floor:** the same subset is also re-scored at 800 nodes. Lc0 search is not fully
  deterministic, so the stored-vs-rerun-800 ρ shows how much ranking change is just engine noise.

## Inputs / Outputs

**Inputs**
- `results/evaluations.jsonl` through `load_results_df()` (the 22 base models). Rows used are those
  with a non-NaN `t2_cpl_clamped`.
- `config/config.yaml` for the Lc0 binary and weights.

**Outputs**
- `results/engine_strength/lc0_<nodes>.jsonl`: an evaluation cache, one line per position
  (`{"fen", "eval", "best_move"}`, White's point of view), so a stopped run resumes.
- `results/metrics/engine_strength.csv`: per model, `n`, then for each strength (`stored`, `800`,
  `8000`) the blunder rate, mean clamped CPL and rank (1 = fewest blunders).
- `results/metrics/engine_strength_summary.json`: positions sampled, moves scored, nodes, ρ for each
  comparison, largest rank shift, the criterion result, and the seed.

## Steps / Logic

1. **Sample** 250 positions per difficulty tier (1,000 in total) with `random.Random(42)`, taken
   from the sorted position ids of each tier.
2. **Collect work** from the sampled rows: every unique pre-move FEN, and every unique
   (FEN, move) pair.
3. **Evaluate** each unique FEN at the given node count (pre-move positions and the position after
   each move), skipping any FEN already in that strength's cache.
4. **Score** each row the way `enrich_cpl.py` does: CPL is 0 when the move is the engine's best
   move, otherwise `compute_clamped_cpl(eval_before, eval_after, white_to_move)` (both ends clamped
   to ±1,000).
5. **Compare** per-model blunder rates (clamped CPL ≥ 1,000) between strengths: Spearman ρ (Pearson
   correlation of average ranks, tie-aware, no scipy) and rank shifts.

## Edge cases

- A FEN the engine fails on is logged and left out; rows that need it are dropped from that
  strength's scores, and the count is reported.
- A model with no scored rows at one strength is left out of that comparison.
- Ties in blunder rate get the average rank (pandas `rank(method="average")`).
- A malformed line in a cache file raises an error naming the file; it is never silently skipped.

## Dependencies

- python-chess, pandas, Lc0 on the GPU. No scipy, as elsewhere in the repo.
- Runtime: about 16,000 evaluations; about 1.6 h at 8,000 nodes and 25 min at 800 on the PC.

## Reproducibility

- Seed 42 for the sample. The node counts, Lc0 binary and weights paths are written to the
  summary. Lc0 search on the GPU is not bit-for-bit deterministic, which the 800-node re-run measures.
