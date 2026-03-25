# Spec: CPL Enrichment Post-Processing

## Goal

Populate `t2_cpl` (centipawn loss) and `t2_best_move` for all records in
`results/evaluations.jsonl` where those fields are currently null. This is a
post-processing pass that runs after the benchmark is fully complete, using Lc0
(GPU-accelerated) to evaluate positions.

CPL measures how many centipawns worse a model's move is compared to Lc0's best
move. A CPL of 0 means the model played the best move; higher is worse.

---

## Background / Why Fields Are Null

- `t2_cpl` requires evaluating the position *after* the model's move. This cannot
  be pre-computed into the dataset because it depends on what the model actually
  played. The worker sets `self.engine = None` to skip engine calls during
  inference for speed.
- `t2_best_move` is null in existing records for the same reason — it was intended
  to be populated from the dataset's `stockfish_best_move`, but that field was
  missing or not correctly passed through in earlier runs.

This script fills both fields in a single post-processing pass using Lc0.

---

## Why Lc0 (not Stockfish)

Lc0 is GPU-accelerated and already installed. Stockfish is CPU-only. For a batch
pass over ~50k records with ~25% legal moves (~12k evaluations), Lc0 is
substantially faster. The existing `Lc0Engine` wrapper in `src/engine_wrapper.py`
is reused as-is.

**Lc0 vs Stockfish eval scale:** CPL values from this script will not be
numerically identical to Stockfish-based CPL. Since both the pre- and post-move
evaluations come from Lc0, the CPL is internally consistent and usable for
ranking models against each other.

---

## Inputs

- `results/evaluations.jsonl` — existing benchmark results, one JSON object per line
- `data/{easy,medium,hard,extreme}.json` — position datasets, keyed by `id`,
  containing `fen` and `stockfish_eval` (used only as fallback; Lc0 re-evaluates)
- `config/config.yaml` — Lc0 binary path, weights path, and nodes setting

---

## Outputs

- `results/evaluations.jsonl` — updated **in place**: records with `t2_legal=True`
  and `t2_cpl=null` will have `t2_cpl` and `t2_best_move` filled in; all other
  records are written unchanged
- A summary printed to stdout on completion

---

## Steps / Logic

### 1. Identify records to enrich

Read `evaluations.jsonl` line by line. Collect all records where:
- `t2_legal == True`
- `t2_cpl is None`

Records that don't meet both criteria are passed through unchanged.

### 2. Deduplicate FEN evaluations

Because 6 prompt formats run on each position, the same FEN can appear up to 6
times (×N models that played a legal move). Evaluate each unique FEN only once.

Build two sets from the qualifying records:

- **Pre-move FENs** — unique `fen` values from the records. Evaluating each gives
  the baseline Lc0 score and best move for that position.
- **Post-move FENs** — unique `(fen, t2_move)` pairs. For each, apply the move to
  the board, get the resulting FEN, then evaluate that FEN.

### 3. Run Lc0 evaluations

Initialize a single `Lc0Engine` using config values:
- `path` from `config["lc0"]["binary"]`
- `weights` from `config["lc0"]["weights"]`
- `nodes` from `config["lc0"]["nodes"]` (default: 800 if not set — faster than
  the precompute scripts, sufficient for CPL ranking)
- `backend`: `"cuda-auto"`

**Pre-move pass** — for each unique pre-move FEN:
1. Call `engine.evaluate(fen)` → returns `{"eval": int, "best_move": str | None}`
2. Store `(lc0_eval, lc0_best_move)` keyed by FEN.

**Post-move pass** — for each unique `(fen, model_move)` pair:
1. Apply `model_move` to a `chess.Board(fen)` to get `new_fen`.
2. Call `engine.evaluate(new_fen)` → returns `{"eval": int}`.
3. Store `lc0_eval_after` keyed by `(fen, model_move)`.

**Best-move shortcut:** if `t2_move == lc0_best_move` for a record, CPL is 0 and
the post-move evaluation for that `(fen, model_move)` can be skipped.

### 4. Compute CPL for each record

Lc0's `evaluate()` returns eval from **White's perspective** (already adjusted for
side to move in the wrapper). Use:

```
lc0_eval_before  = pre-move result for record's fen
lc0_eval_after   = post-move result for (record's fen, record's t2_move)

if side to move is White:
    cpl = lc0_eval_before - lc0_eval_after   # White wants higher eval
else:
    cpl = lc0_eval_after - lc0_eval_before   # Black wants lower eval

cpl = max(0, cpl)
```

Set:
- `t2_cpl` = computed CPL (or `None` if engine call failed)
- `t2_best_move` = `lc0_best_move` from the pre-move evaluation

### 5. Write output atomically

Write all records (enriched and unchanged) to a temp file in the same directory,
then replace the original with an atomic rename. This prevents data loss if the
process is interrupted.

### 6. Print summary

```
CPL enrichment complete.
  Records processed : 50668
  Enriched          : 12445
  Unique FENs       : 6234  (pre-move evaluations run)
  Unique moves      : 9187  (post-move evaluations run)
  Skipped (best mv) : 311   (CPL=0, no eval needed)
  Already had CPL   : 0
  Skipped (illegal) : 38223
  Failed            : 12
  Output            : results/evaluations.jsonl
```

---

## Edge Cases

- **Move parse error** — if applying `t2_move` to the board raises an exception
  (shouldn't happen since `t2_legal=True`, but guard anyway), log a warning and
  leave `t2_cpl = None` for that record.
- **Lc0 engine error** — on failure, the `Lc0Engine` wrapper already attempts one
  restart. If it still fails, log the error and leave `t2_cpl = None`; continue
  processing remaining records.
- **Mate scores** — Lc0 may return large values (e.g., ±10000) for forced mates.
  These are valid — do not clamp beyond `max(0, cpl)`.
- **Already-enriched records** — idempotent: if `t2_cpl` is not null, the record
  is passed through unchanged. Safe to re-run.
- **Interrupted mid-run** — atomic write (temp file + rename) means a crash leaves
  the original intact. Re-running will re-process any records that weren't written.
- **Large file** — stream `evaluations.jsonl` line-by-line; do not load the full
  file into memory.

---

## Performance

- Lc0 at 800 nodes/position is fast on GPU — roughly 50–200ms per position
  depending on network size.
- Deduplication is the main speedup: 6 prompt formats per position means up to 6×
  fewer pre-move evaluations. With ~12k legal-move records across ~2k unique
  positions, expect ~2k pre-move + ~10k post-move = ~12k engine calls total
  (minus best-move shortcuts).
- At ~100ms/call: ~20 min. At ~50ms/call: ~10 min.
- Lc0 is a single GPU process — do not attempt multiple instances.

---

## Dependencies

- `src.engine_wrapper.Lc0Engine` — existing wrapper, reuse as-is
- `src.utils.load_config` — for reading `config/config.yaml`
- `chess` (python-chess) — already in `requirements.txt`

---

## Script Location

`scripts/enrich_cpl.py`

Run after the full benchmark pipeline completes:

```bash
python scripts/enrich_cpl.py
python scripts/enrich_cpl.py --dry-run    # print counts, no writes
python scripts/enrich_cpl.py --nodes 400  # override nodes for speed
```
