# Spec: Benchmark Artefact Fixes

**Created:** 2026-05-07
**Author:** Ryan Brew

---

## Goal

Fix the metric-computation artefacts that inflate the headline numbers in
`docs/FINDINGS.md`. None of these need new LLM calls or Stockfish re-runs —
all are postprocessing of `results/evaluations.jsonl`.

The qualitative findings of the benchmark survive (models are uniformly bad;
scaling doesn't help; Gemma 4 family is best; llama3.3:70b is bottom-tier).
What changes is the **magnitude** of those badness numbers and the
defensibility of the metric definitions.

---

## Artefacts being fixed

### A1 — CPL is dominated by mate-encoded evals

`engine_wrapper._evaluate_internal` encodes mate-in-N as `±10000 - mate_in*10`,
which is mathematically unbounded and produces values up to ±16,000 cp in the
dataset. `evaluator.score_t2` subtracts these without clamping, so:

- 4.3% of jobs have a mate-encoded *start* eval
- 20.0% of jobs have a mate-encoded *eval-after-move*
- 22.0% of jobs have at least one mate-encoded endpoint

This inflates mean CPL from a Lichess-convention ~700 cp to the reported
~4,800 cp. **Median is 2,431; the spread tail is what kills the average.**

**Fix:** clamp both `stockfish_eval` and `eval_after` to `±EVAL_CLAMP_CP`
before subtraction. Default `EVAL_CLAMP_CP = 1000` (Lichess convention; "down
a queen" already saturates the signal).

### A2 — T1 absolute_error inflated by mate-truth rows

The model's eval output is clamped to `±2000` ([src/evaluator.py:81]). The
truth can be `±16,000`. So mate-truth rows automatically contribute ~9,966 cp
of "error" that the model could never have reduced.

| Truth type | Mean abs error | Median |
|---|---|---|
| Mate truth (`|sf| ≥ 9000`) | 9,966 | 9,999 |
| Non-mate truth | 143 | 67 |

**Fix:** add a `t1_abs_error_excl_mate` aggregation column alongside the
existing absolute_error. Don't change the raw stored field — keep it for
transparency. The `relative_error` column already partially handles this
because the denominator scales with `|truth|`, but the headline-table
absolute number should also be reported in the corrected form.

### A3 — T3 theme matcher is broken for camelCase Lichess labels

[src/evaluator.py:14-36] has snake_case keys (`passed_pawn`, `back_rank`).
Lichess theme labels are camelCase (`advancedPawn`, `kingsideAttack`,
`backRankMate`, `bishopEndgame`, etc.). The fallback at line 208 produces
gibberish like `"advancedpawn"` that no human writes.

Resulting theme-component match rates:
- `advancedPawn`, `kingsideAttack`, `backRankMate`, `attraction`,
  `bishopEndgame`, `clearance`: **0.0%** match rate across thousands of rows
- `crushing`: 0.2%
- `endgame`: 2.1%

**Fix:** rewrite `THEME_SYNONYMS` keyed by exact Lichess label strings,
with chess-domain synonym lists for each. Provide a tokeniser that splits
camelCase into space-separated lowercase words as the final fallback (so
`advancedPawn` → `"advanced pawn"`, which models actually write).

This is a postprocessing rescore: the stored `t3_explanation` text is
unchanged, but `t3_p2_theme_correct` is recomputed from it.

---

## Inputs / Outputs

**Inputs:**
- `results/evaluations.jsonl` — 526,662 records, unchanged
- `data/{easy,medium,hard,extreme}.json` — for `fen` lookup (to get
  side-to-move for CPL reconstruction)

**Outputs:**
- `src/evaluator.py` — clamped CPL formula; rewritten THEME_SYNONYMS
- `src/metrics.py` — derived columns:
  - `t2_cpl_clamped_1000` (primary, headline)
  - `t2_wp_loss` (secondary, bounded [0, 1000] milli-WP)
  - `t1_abs_error_excl_mate` (mate-truth-aware T1 error)
- Aggregations include the new columns
- `results/metrics/by_model.csv` and `summary.json` regenerated
- Plots regenerated using clamped CPL as the headline metric
- `docs/FINDINGS.md` updated with corrected numbers and a methodology
  paragraph explaining the artefacts

---

## Steps / Logic

### 1. `src/evaluator.py`

Add a constant near the top:
```python
EVAL_CLAMP_CP = 1000  # Lichess convention; "down a queen" saturates signal
```

Modify `score_t2()` to clamp both endpoints before computing CPL:
```python
sf_clamped = max(-EVAL_CLAMP_CP, min(EVAL_CLAMP_CP, stockfish_eval))
ea_clamped = max(-EVAL_CLAMP_CP, min(EVAL_CLAMP_CP, eval_after))
if board.turn == chess.WHITE:
    cpl = sf_clamped - ea_clamped
else:
    cpl = ea_clamped - sf_clamped
cpl = max(0, cpl)
```

Rewrite `THEME_SYNONYMS` keyed by exact Lichess theme strings. Add helper:
```python
def _camel_to_words(s: str) -> str:
    """advancedPawn -> 'advanced pawn'; backRankMate -> 'back rank mate'."""
    out = []
    for ch in s:
        if ch.isupper() and out:
            out.append(' ')
        out.append(ch.lower())
    return ''.join(out)
```

`score_t3()` matching becomes:
```python
synonyms = THEME_SYNONYMS.get(theme, [_camel_to_words(theme)])
synonyms = synonyms + [theme.lower(), _camel_to_words(theme)]
for s in synonyms:
    if s in explanation_lower:
        p2 = 1
        break
```

### 2. `src/metrics.py`

Within `load_results_df()`, add derived columns by joining
`results/evaluations.jsonl` against `data/*.json` for FEN → side-to-move.
Reconstruct `eval_after` from `(t1_stockfish_eval, t2_cpl, fen)`:
```python
eval_after = sf - cpl if white_to_move else sf + cpl  # white-to-move sign
```

Then:
```python
sf_c = clamp(sf, -1000, 1000)
ea_c = clamp(ea, -1000, 1000)
df['t2_cpl_clamped_1000'] = max(0, sf_c - ea_c) if white_to_move else max(0, ea_c - sf_c)
df['t2_wp_loss'] = abs(sigmoid(sf_c/400) - sigmoid(ea_c/400)) * 1000
df['t1_abs_error_excl_mate'] = where(|sf|<9000, t1_absolute_error, NaN)
```

Add `t3_p2_theme_correct_v2` rescore using the new theme matcher applied
to `t3_explanation`. Keep the original column for transparency.

Aggregations include all new columns.

### 3. `scripts/generate_plots.py`

Change the headline CPL metric in the overview ranking + scaling charts
from `t2_cpl_mean` to `t2_cpl_clamped_1000_mean`. Add a small annotation
explaining "(clamped at ±1000 cp; raw uncapped value also shown)".

The summary heatmap and difficulty profile use the new column too.

### 4. Tests (`tests/test_artefact_fixes.py`)

| Test | Checks |
|---|---|
| `test_cpl_clamp_caps_mate_inflation` | mate-encoded endpoints → CPL ≤ 2000 |
| `test_cpl_unchanged_in_normal_range` | small evals (within ±1000) → identical to raw |
| `test_camel_to_words` | `advancedPawn` → `"advanced pawn"`; `backRankMate` → `"back rank mate"` |
| `test_theme_match_camelcase_label` | label=`advancedPawn`, explanation contains "advanced pawn" → match |
| `test_theme_match_existing_synonyms` | label=`fork`, explanation contains "double attack" → match (existing behaviour preserved) |
| `test_t1_abs_error_excl_mate_drops_mate_rows` | mate-truth row → NaN; non-mate row → original value |
| `test_wp_loss_bounded` | output always in [0, 1000] |

### 5. Re-run

```
python scripts/generate_plots.py --save-metrics
```

No LLM calls. ~1–2 minutes wall time.

### 6. Update `docs/FINDINGS.md`

- Replace the CPL table headline numbers with clamped values
- Add a "Methodology and metric artefacts" section near the bottom
- Note the T3 theme-matcher fix and the rescored T3 numbers
- Note the T1 absolute-error mate-truth caveat

---

## Edge Cases

| Scenario | Behaviour |
|---|---|
| `stockfish_eval` exactly at ±1000 | clamp is a no-op; CPL identical |
| `eval_after` is mate-encoded but `stockfish_eval` is normal | both clamped to ±1000; CPL ≤ 2000 |
| Both endpoints mate-encoded | CPL ≤ 2000 (was up to 20,000) |
| Theme label not in dict and not camelCase | use `theme.lower()` directly |
| Empty `t3_explanation` | both old and new matcher return 0 (unchanged) |
| Side-to-move ambiguity in CPL reconstruction | resolve by parsing the FEN (single source of truth) |

---

## Definition of Done

- [ ] `tests/test_artefact_fixes.py` passes alongside existing suite
- [ ] `results/metrics/by_model.csv` has new columns
- [ ] `results/plots/*.png` regenerated using clamped CPL
- [ ] `docs/FINDINGS.md` updated with corrected numbers and methodology section
- [ ] All existing tests still pass
- [ ] Commit + push
