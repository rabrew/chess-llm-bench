# Spec: Relative-Error Metric for H1 Calibration

**Closes:** Issue 1 in `young-scientist-2027-readiness.md`
**Author:** Ryan Brew
**Created:** 2026-04-28

---

## Goal

Replace the unnormalised `t1_absolute_error` as the sole metric for hypothesis H1 ("evaluation error increases with difficulty"). The current data shows:

- easy:    1720 cp mean error
- medium:   740 cp
- hard:     436 cp
- extreme:  310 cp

This *looks* like the hypothesis is rejected, but it is almost certainly a measurement artefact: Lichess "easy" puzzles tend to have decisive Stockfish evaluations (e.g. ±2000 cp), so a model guessing near zero racks up huge absolute errors. On "extreme" puzzles the true eval is closer to zero, so the same model looks accurate in absolute terms.

We need a metric that is invariant to position-magnitude spread, so the hypothesis is tested on the model's actual ability rather than the puzzle distribution.

---

## Inputs

- `results/evaluations.jsonl` — already contains, per row:
  - `t1_model_eval` (int, centipawns) — model's estimate
  - `t1_stockfish_eval` (int, centipawns) — ground truth
  - `t1_absolute_error` (int, centipawns)
  - `t1_direction_correct` (bool)
  - `difficulty` ∈ {easy, medium, hard, extreme}

No new LLM calls required. No raw data to re-collect.

## Outputs

- A new column in the metrics dataframe: `t1_relative_error`
- Updated `summary.json["hypothesis_tests"]` with H1 reported under three metrics:
  - `t1_absolute_error` (existing — kept for transparency and to expose the artefact)
  - `t1_relative_error` (new — primary)
  - `t1_direction_correct` (already computed — secondary)
- A new section in the eventual report's methodology discussing the artefact and why the normalised view is preferred.

---

## Steps / Logic

### 1. Define the metric

```
relative_error(model, truth) = |model - truth| / max(|truth|, FLOOR)
```

with `FLOOR = 100` centipawns. Rationale:

- The clamp prevents division blow-up when the true eval is near zero.
- 100 cp ≈ one pawn — a meaningful unit of chess strength. An error of "100 cp on a 0 cp position" is genuinely a 1-pawn miss; treating that as 1.0 relative error is sensible.
- The metric is in `[0, ∞)`. A value of `1.0` means *"off by the magnitude of the truth."* A value of `0.1` is excellent; `>2.0` is bad.

### 2. Implement in `src/metrics.py`

Add a pure helper:

```python
def compute_relative_error(
    model_eval: float, stockfish_eval: float, floor: int = 100
) -> float:
    return abs(model_eval - stockfish_eval) / max(abs(stockfish_eval), floor)
```

Inside `load_evaluations_dataframe()` (the existing loader that builds the analysis DF), add a derived column:

```python
df["t1_relative_error"] = df.apply(
    lambda r: compute_relative_error(r["t1_model_eval"], r["t1_stockfish_eval"])
    if pd.notnull(r["t1_model_eval"]) and pd.notnull(r["t1_stockfish_eval"])
    else None,
    axis=1,
)
```

### 3. Extend `compute_hypothesis_tests()`

H1 is currently:

```python
h1_data = df.groupby("difficulty")["t1_absolute_error"].mean()
```

Replace with a multi-metric block:

```python
h1_metrics = {
    "absolute_error": df.groupby("difficulty")["t1_absolute_error"].mean(),
    "relative_error": df.groupby("difficulty")["t1_relative_error"].mean(),
    "direction_accuracy": df.groupby("difficulty")["t1_direction_correct"].mean(),
}
```

For each metric, compute `supported` against the predicted direction:

- absolute_error / relative_error: hypothesis predicts an *increase* with difficulty
- direction_accuracy: hypothesis predicts a *decrease* with difficulty

Order tiers as `[easy, medium, hard, extreme]` and check monotonic trend.

### 4. Update `summary.json` schema

Replace the existing single-result H1 block with:

```json
"H1": {
  "description": "T1 error increases with difficulty",
  "metrics": {
    "absolute_error": {"supported": false, "values": {...}, "rationale": "..."},
    "relative_error": {"supported": true, "values": {...}},
    "direction_accuracy": {"supported": true, "values": {...}}
  },
  "primary_metric": "relative_error",
  "primary_supported": true
}
```

The `primary_metric` field is what the report should headline.

### 5. Re-run metrics

Run the existing `scripts/generate_plots.py --save-metrics` (or whatever entry point produces summary.json from evaluations.jsonl) — no LLM calls, just postprocessing.

### 6. Update the dashboard

`dashboard/server.py` exposes the hypothesis-test JSON. Update its template to show all three metrics for H1 with the primary_metric highlighted.

---

## Edge Cases

- **`t1_stockfish_eval == 0`**: clamp denominator to `FLOOR = 100`, so `relative_error = |model| / 100`.
- **Mate-in-N positions**: already excluded upstream by the `--skip-mates` filter; `t1_stockfish_eval` will not be a mate score.
- **Missing `t1_model_eval`** (model failed to produce a parseable number): leave `t1_relative_error = None`. Pandas `mean()` ignores NaN by default.
- **Extreme outliers** (e.g. model says +9999 on a +50 truth → relative_error = 99): keep them in. Do not clip. The mean is what matters and the existing absolute-error code already keeps these. If the report wants robust statistics, also report median.
- **H2 (CPL trend)**: out of scope for this spec — CPL is already a magnitude-aware metric (centipawn loss is bounded by the model's actual move quality, not the position eval). Will be addressed separately if it shows the same artefact.

---

## Tests

`tests/test_relative_error.py`:

- `compute_relative_error(0, 0)` → 0.0
- `compute_relative_error(50, 50)` → 0.0
- `compute_relative_error(0, 100)` → 1.0
- `compute_relative_error(0, 1000)` → 1.0
- `compute_relative_error(0, 50)` → 0.5 (50 / max(50, 100) = 50/100)
- `compute_relative_error(-100, 100)` → 2.0
- `compute_relative_error(0, 0)` with default floor → 0.0 (no NaN)
- A small DataFrame fixture with mixed difficulty tiers → check that the H1 multi-metric block yields the expected per-tier means.
- A NaN-handling test: row with `t1_model_eval = None` produces `t1_relative_error = None` and is excluded from group means.

All existing metrics tests must still pass.

---

## Dependencies

- No new Python packages.
- `pandas`, `numpy` already imported in `src/metrics.py`.

---

## Definition of Done

- `tests/test_relative_error.py` passes alongside the rest of the suite.
- `summary.json` regenerated, `hypothesis_tests.H1` shows all three metrics with their support verdict.
- Discussion paragraph drafted (in the eventual report) explaining the artefact and the normalised view.
- Dashboard updated to show the new H1 view.
- Commit + push.
