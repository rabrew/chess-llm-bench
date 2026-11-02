# Spec: Metric Defect Remediation (post-audit review)

**Created:** 2026-08-13
**Author:** Ryan Brew
**Status:** Open — no fixes applied yet
**Supersedes nothing.** This is a follow-up to `specs/artefact-fixes.md` (the
2026-05-07 audit). That audit fixed A1–A5 in `by_model.csv`, `summary.json`
and `docs/FINDINGS.md` — but did **not** propagate the corrections to the
model-family aggregation or to hypothesis tests H3 and H4, and introduced no
chance-corrected baseline for direction accuracy.

**Amended 2026-08-13 (same day), after a full file-by-file read of the repo.**
D9–D11 were added. D9 supersedes part of D7: the T2 engine question is not
merely ambiguous, the *T1* ground-truth engine is also misstated, and the
Stockfish depth published everywhere is not the depth that ran. D10 upgrades
the A6/D6 rescue disclosure from a caveat to a P0, and identifies a route to
the single-shot legality number that D6 assumed was unrecoverable.

---

## Goal

Close eleven defects that would undermine the benchmark if published as-is.

**None of these require re-running LLM inference, Stockfish, or Lc0.** All are
postprocessing over the existing `results/evaluations.jsonl` (526,662 records),
documentation corrections, or file housekeeping — exactly like the May audit.
This is true of D9–D11 as well: D9 is prose plus a provenance block, D10 is a
read-only partition of records already on disk, D11 is deleting a stale file.
If D10 step 4 lands with adequate coverage, it *removes* the only item in this
register that would have needed GPU time (D6 step 3).

**What is not in question:** every headline number in `docs/FINDINGS.md`
reconciles against `results/metrics/by_model.csv` to within rounding. This was
re-verified on 2026-08-13 across all eight headline figures (legality 97.847
vs 97.85 claimed; clamped CPL 701.6 vs 702; direction t0/t50/t100/t200 =
44.18/46.19/61.75/79.06 vs 44.5/46.1/61.6/78.9; wp_loss 310.7 vs 311; T3 v2
0.519 vs 0.520). The retry experiment reconciles exactly (10,665 records,
7,212 legal = 67.6%). The defects below are about **metric construction and
stale propagation**, not arithmetic errors.

**What D9 and D10 add to that statement:** the arithmetic still holds, but
(a) the *provenance* attached to those numbers in the docs is wrong — they are
not Stockfish-at-depth-22 numbers (D9), and (b) the legality figure pools two
different measurement procedures, so it is arithmetically correct but
semantically undefined (D10). Reconciling a number against a CSV does not
establish that the number measures what the prose says it measures.

---

## Inputs / Outputs

**Inputs**
- `results/evaluations.jsonl` — 526,662 records (local only; not in git)
- `results/metrics/*.csv`, `results/metrics/summary.json` — current outputs
- `src/metrics.py`, `src/utils.py` — computation under repair

**Outputs**
- Corrected `results/metrics/by_model_family.csv` (new columns, correct groups)
- Corrected `summary.json` H3/H4 blocks
- New `results/metrics/direction_baselines.csv` — per-threshold class priors
- New `results/metrics/cpl_buckets.csv` — blunder-rate breakdown with CIs
- New `results/metrics/regime_split.csv` — pre/post-rescue legality split (D10)
- New `provenance` block in `summary.json` — engines, budgets, git SHA (D9)
- **Deleted** `results/metrics/learning_deltas.csv` — stale pilot artefact (D11)
- Updated `docs/FINDINGS.md` Findings 1, 3, 5, Summary Table, and the whole
  "Methodology and metric artefacts" section (engine/depth claims — D9)
- Corrected engine/depth claims in `README.md`, `docs/SIMPLE_SUMMARY.md`,
  `specs/benchmark.md`, `specs/benchmark-detailed.md` (D9)
- New tests in `tests/test_metric_defects.py`

---

## Defect register

Severity: **P0** blocks publication · **P1** materially misleads · **P2** cosmetic

---

### D1 — Direction accuracy is compared against an invalid baseline (P0)

**Status:** Root cause verified in code. Magnitude needs raw data to confirm.

`docs/FINDINGS.md` Finding 3 reports direction accuracy against a stated
**"Random baseline (3-class) = 33%"** at every threshold. That baseline is only
valid when the three classes are balanced. They are not.

`_compute_direction_correct` (`src/metrics.py:258-271`) applies the *same*
threshold to both the model eval and the Stockfish eval:

```python
me_dir = np.where(model_eval > threshold, "W",
                   np.where(model_eval < -threshold, "B", "E"))
sf_dir = np.where(stockfish_eval > threshold, "W",
                   np.where(stockfish_eval < -threshold, "B", "E"))
correct = (me_dir == sf_dir)
```

As `threshold → ∞`, both sides collapse to `"E"` for every record and accuracy
converges to **100% by construction**, independent of model skill. So the
reported climb 44.5% → 46.1% → 61.6% → 78.9% is partly a mechanical artefact of
raising the threshold, not evidence of improving model judgment.

This matters concretely because the ground truth is heavily imbalanced. Our own
methodology section states **92% of Stockfish ground-truth evals lie within
±300 cp**. If the ±200 "equal" class holds ~80% of positions, then a degenerate
model answering `"E"` unconditionally scores ~80% — **above** the 78.9%
all-model average currently presented as a positive result.

**Why it matters:** Finding 3's strongest claim ("at a 2-pawn threshold, models
agree with Stockfish 79% of the time") could invert to "models fail to beat the
majority-class baseline." This is the first thing a reviewer will test.

**Fix**
1. Compute the ground-truth class prior at each threshold in
   `DIRECTION_THRESHOLDS_CP` and emit `direction_baselines.csv` with columns
   `threshold, pct_W, pct_B, pct_E, majority_class, majority_baseline_acc`.
2. Report **Cohen's κ** and **balanced accuracy** per (model, threshold)
   alongside raw accuracy. κ corrects for chance agreement given the actual
   marginals; raw accuracy does not.
3. Replace the "Random baseline 33%" column in Finding 3 with the
   majority-class baseline per threshold.
4. If κ ≤ 0 at ±200, say so explicitly and rewrite Finding 3's conclusion.

**Verification**
- `test_direction_baseline_matches_priors` — synthetic df, known class split,
  assert `majority_baseline_acc` equals the majority prior.
- `test_degenerate_predictor_scores_baseline` — a model emitting constant 0 cp
  must score ≈ the majority baseline and **κ ≈ 0** at every threshold.
- `test_kappa_zero_for_random_predictor` — shuffled predictions give κ ≈ 0.

---

### D2 — Direction accuracy is non-monotonic and reverses model ranking (P0)

**Status:** Verified from `by_model.csv`.

Same root cause as D1, different symptom. Per-model accuracy across thresholds:

| model | t0 | t50 | t100 | t200 |
|---|---|---|---|---|
| gemma3:4b | **10.4** | 52.2 | 75.8 | 91.0 |
| llama3.3:70b | 58.2 | 34.3 | **21.9** | 42.2 |
| gemma4:31b | 52.2 | 38.8 | **31.8** | 61.7 |
| qwen2.5:14b | 58.2 | 47.6 | 66.5 | 80.0 |
| deepseek-r1:14b | 41.8 | 53.0 | 76.7 | 91.8 |

gemma3:4b swings 7× (10.4% → 75.8%) on identical underlying predictions.
llama3.3:70b and gemma4:31b both dip below the 33% figure and then *recover* at
±200 — non-monotonic, which a well-posed accuracy metric should not be. The
metric is measuring the interaction between each model's **output scale** and
the threshold, not its ability to identify the stronger side.

Consequence: Finding 3's headline claims — "llama3.3:70b is systematically
wrong", "the largest model in the study is anti-correlated with truth", and the
whole "play vs reason decoupling" narrative — are **threshold-contingent**. At
t0, llama3.3:70b (58.2%) *outranks* deepseek-r1:14b (41.8%), the exact reverse
of the reported ranking.

**Competing explanation that must be ruled out first:** a sign or
perspective bug. If a model reports evaluations from the side-to-move's
perspective rather than White's, its sign inverts on every Black-to-move
position (~50% of records) and produces exactly this signature. The two models
that dip below random (llama3.3:70b, gemma4:31b) are the prime suspects.
"70B models are anti-correlated with chess truth" is an extraordinary claim; a
parser bug is the ordinary one, and it must be eliminated before publication.

**Fix**
1. Split direction accuracy by side-to-move (`_white_to_move_from_fen` already
   exists at `src/metrics.py:61`). If llama3.3:70b's accuracy is high on
   white-to-move and inverted on black-to-move, it is a perspective bug — fix
   the parser in `src/evaluator.py` and recompute. This is the decisive test.
2. Sample 50 raw llama3.3:70b and gemma4:31b eval responses and hand-check the
   parsed sign against the model's own words.
3. Report a scale-free rank measure (Spearman ρ between model eval and
   Stockfish eval per model) that does not depend on any threshold. Lead with
   this; keep the threshold table as a secondary robustness view.

**Verification**
- `test_direction_split_by_side_to_move` — synthetic perspective-inverted
  model scores high on white-to-move, near-zero on black-to-move.
- `test_spearman_rho_threshold_free` — ρ is unchanged when all evals are
  scaled by a positive constant (thresholded accuracy is not).

---

### D3 — Model family/size parsing merges distinct models (P0, live bug)

**Status:** Verified exactly. Arithmetic below is exact, not approximate.

`parse_model_info` (`src/utils.py:140-186`) has two independent faults.

**Fault A — incomplete family map.** Only five prefixes are recognised:

```python
family_map = {"qwen": "qwen", "llama": "llama", "mistral": "mistral",
              "phi": "phi", "gemma": "gemma"}
```

Missing: `deepseek`, `mixtral`, `solar`, `codellama`, `yi`, `command-r`,
`wizardlm`. **8 of 22 models fall into family `"unknown"`.**

**Fault B — size parser fails on non-integer tags.**

```python
size_clean = size_str.lower().replace("b", "")   # "e2b" -> "e2", "8x7b" -> "8x7"
try: size_b = int(size_clean)
except ValueError:
    try: size_b = float(size_clean)
    except ValueError: pass                       # size_b stays 0
```

`e2b`, `e4b`, and `8x7b` all silently parse to **0.0 billion parameters**.

`aggregate_by_model_family` (`src/metrics.py:454`) then groups by
`["model_family", "model_size_b"]`, so any two models colliding on that pair
are **averaged into one row**. Since no single model exceeds 24,000 records,
any row above that is a merge. Three rows are:

| row in `by_model_family.csv` | n | actually contains |
|---|---|---|
| `gemma`, size `0.0` | 47,962 | gemma4:e2b (23,963) + gemma4:e4b (23,999) |
| `unknown`, size `7.0` | 46,935 | deepseek-r1:7b (22,970) + wizardlm2:7b (23,965) |
| `unknown`, size `34.0` | 47,982 | codellama:34b (24,000) + yi:34b (23,982) |

All three sums are exact. `mixtral:8x7b` also lands at `unknown/0.0`.

**Why it matters:** CodeLlama and Yi averaged together is not a family, and
**H4 ("larger models perform better within family") is computed over these
mislabelled groups** — see `src/metrics.py:677-682`. Any scaling claim derived
from this table is unsound.

**Fix**
1. Extend `family_map` to cover all 22 tags. Note ordering: check `codellama`
   before `llama` if prefix matching is kept, or switch to explicit exact-tag
   lookup (preferred — it fails loudly on an unknown tag).
2. Parse sizes with a regex that handles `e2b`/`e4b` (Gemma "effective"
   variants) and `8x7b` (MoE). Record MoE models with both total and active
   parameter counts; they are not comparable to dense models on a single axis
   and should be excluded from scaling regressions.
3. **Raise on `family == "unknown"` or `size_b == 0`** rather than silently
   bucketing. A silent 0.0 is what caused this.
4. Add an assertion in `aggregate_by_model_family` that each output row's
   `job_id` count is ≤ the max per-model count, and that group count equals the
   distinct model count.

**Verification**
- `test_parse_model_info_all_configured_tags` — every tag in `config.yaml`
  yields a non-`unknown` family and non-zero size.
- `test_parse_model_info_effective_and_moe_tags` — `gemma4:e2b`→2,
  `gemma4:e4b`→4, `mixtral:8x7b`→(56 total / 12.9 active) or explicit MoE flag.
- `test_codellama_not_matched_as_llama` — regression guard on prefix ordering.
- `test_family_rows_are_one_model_each` — no output row exceeds max per-model
  record count.
- `test_unknown_family_raises` — unrecognised tag raises, not defaults.

---

### D4 — H3, H4 and the family table still use pre-audit metrics (P1)

**Status:** Verified in code and in `summary.json`.

The May audit added corrected columns (`t2_cpl_clamped`, `t2_wp_loss`,
`t3_score_v2`, `t1_abs_error_excl_mate`, `t2_legal_attempted`,
`t1_direction_correct_t{0,50,100,200}`) and wired them into H1 and H2 via the
`primary_metric` / `metrics` pattern. **H3, H4 and
`aggregate_by_model_family` were missed.**

| site | code | uses | should use |
|---|---|---|---|
| H3 | `metrics.py:661` | `t3_score` | `t3_score_v2` |
| H4 | `metrics.py:682` | `t1_absolute_error`, `t2_cpl`, `t3_score` | `t1_abs_error_excl_mate`, `t2_cpl_clamped`, `t3_score_v2` |
| family agg | `metrics.py:454` | `t1_absolute_error`, `t2_cpl`, `t2_legal`, `t3_score` | corrected equivalents |

Confirmation from the shipped artefacts: H3's values in `summary.json`
(`easy 0.4176, medium 0.4342, hard 0.4029, extreme 0.3809`) match the **v1**
`t3_score` column in `by_difficulty.csv` — the matcher documented in A3 as
missing 70% of theme classes. And `by_model_family.csv`'s `t2_legal` column
sits at ~0.66, i.e. it is still the **A4 buggy legality denominator** that the
audit replaced with 97.85%.

So the repo simultaneously publishes 97.85% legality in `by_model.csv` and
64.6%-flavoured legality in `by_model_family.csv`. That internal contradiction
is trivially discoverable.

**Fix:** apply the H1/H2 `primary_metric`/`metrics` pattern to H3 and H4;
add corrected columns to the family aggregation; keep the legacy columns
alongside for transparency, suffixed `_legacy`.

**Verification**
- `test_h3_uses_v2_theme_score` — H3 reports `primary_metric == "t3_score_v2"`.
- `test_h4_uses_corrected_metrics` — H4 reports clamped CPL, not `t2_cpl`.
- `test_family_table_has_corrected_columns` — asserts presence of all five
  corrected columns.
- `test_no_buggy_legality_in_published_csvs` — no shipped CSV carries a
  legality column whose all-model mean is < 0.90.

---

### D5 — Clamped CPL is saturated; model ranking rests on ~2% of scale (P1)

**Status:** Verified from `by_model.csv`.

Per-model **median** clamped CPL runs **901–980** against a clamp of 1000.
More than half of all legal moves sit within 5–10% of the ceiling. Cross-model
standard deviation is **22.5 on a 0–1000 scale**.

The headline "gemma4:31b best at 634, gemma4:e2b worst at 727" is a 93-point
gap on a metric that is pinned near its cap, with 39.1% of moves exactly at it.
Mechanically, `mean_clamped_cpl ≈ 1000 × P(blunder) + (small residual)` — the
metric is a **blunder-rate proxy in centipawn clothing**, and presenting it as
a continuous quality score overstates its resolution.

**Fix**
1. Report **blunder rate** (`P(CPL ≥ 1000)`) as the primary T2 quality metric —
   it is what the number actually encodes, and it is directly interpretable.
2. Attach bootstrap 95% CIs to every per-model T2 figure. With ~24k records per
   model the CIs will be tight, but the point is to show the reader that a
   634-vs-727 gap on a saturated metric is a weak ordering.
3. Emit `cpl_buckets.csv` with the five buckets already in Finding 2, per model,
   with counts — so readers can see the bimodality directly rather than
   inferring it from a mean.
4. Keep median alongside mean everywhere; a mean of 702 against a median of
   ~950 is itself informative and currently unreported per-model.

**Verification**
- `test_blunder_rate_matches_bucket_counts` — blunder rate equals the
  `CPL ≥ 1000` bucket fraction.
- `test_cpl_ci_overlap_flagged` — models with overlapping 95% CIs are marked
  `rank_tied = True`.

---

### D6 — Legality headline measures the harness, not the model (P1)

**Status:** Known (A6), under-weighted in the writeup.

Finding 1 leads with **97.85% legality**. Per A6 this is *post-rescue*: on
`cot`/`fen_only`/`pgn+fen`, `src/worker.py:208-222` silently (a) scans the
response for any other legal SAN/UCI token, then (b) re-prompts with the full
legal-move list. And `move_only` supplies the legal-move list up front. The
single-shot rate is unrecoverable from stored data.

Publishing "models produce a legal move 98% of the time" as the lead finding
invites exactly one question — *what was the first-attempt rate?* — which
currently has no answer. The supporting evidence points the other way: on the
10,665 records where the rescue still failed, an explicit re-prompt with the
answer set fixed only **67.6%**.

**Fix**
1. Relabel throughout as **"harness-assisted legality (≤2 attempts)"**. Never
   present the bare 98% without that qualifier.
2. Instrument `src/worker.py` to record `t2_move_attempt_1`,
   `t2_legal_attempt_1`, and `t2_rescue_path` for future runs.
3. Re-run a **single-shot legality probe** on a stratified subsample
   (suggest 200 positions × 22 models × 3 combined formats ≈ 13,200 calls,
   a few GPU-hours) with rescue disabled. That yields a defensible
   single-shot number without re-running the full 526k benchmark.
4. Demote legality from Finding 1. Lead with the bimodal CPL result — it is
   the genuine contribution and the finding least affected by D1–D6.

**Verification**
- `test_worker_records_first_attempt` — rescue path populates attempt-1 fields.
- `test_rescue_disabled_flag` — `rescue_enabled=False` returns the raw first
  parse unmodified.

---

### D7 — T2 ground-truth engine is ambiguous and weak (P2, document)

`docs/FINDINGS.md` states T2 best-move/CPL truth is **Lc0 @ 800 nodes**, but
`results/evaluations_retried.jsonl` names its field `stockfish_best_move`, and
`retried_cpl` is null in all 10,665 records. The artefacts disagree with the
documentation about which engine produced T2 truth.

800 nodes is also a weak reference — and we already report that the two engines
disagree on the best move on **~67%** of the overlap.

**Fix:** rename the field to its actual source; state node/depth settings and
engine versions in `summary.json` provenance (see D8); soften "ground truth" to
"reference engine" for T2; add a sensitivity check re-scoring a 1,000-position
subsample at higher Lc0 node counts to show the ranking is stable.

---

### D8 — Doc/data drift and missing provenance (P2)

| item | doc | data |
|---|---|---|
| raw uncapped mean CPL | 4,786 (`FINDINGS.md`) | 4,735.0 (`summary.json`) |
| all-model legality | 97.85% | 97.88% |
| direction t50 | — | `summary.json` carries **two** values: `t1_direction_accuracy_t50 = 0.46262` and `by_threshold["t50"] = 0.46135` |

Also: `summary.json` has no `generated_at`, git SHA, config snapshot, or engine
versions. The only committed run logs are March **pilot** runs (50 and 5
positions per tier, 2 models, one at depth 15 not 22) — there is no run log for
the actual 526k run, so its date is only recoverable from commit archaeology.

**Fix:** single source of truth for each number (docs generated from
`summary.json`, never hand-typed); reconcile the duplicate t50; stamp
`generated_at`, git SHA, engine versions and config snapshot into
`summary.json` in `scripts/generate_plots.py`.

> **See D9.** The depth-15 pilot log noted above is not an outlier — depth 15
> is the only depth ever committed. D9 supersedes this bullet's framing.

---

### D9 — Ground-truth engine and search depth are misstated in every document (P0)

**Status:** Confirmed by the author, 2026-08-13. Lc0 was chosen **deliberately**
— this machine's GPU is far stronger than its CPU, so a GPU-resident
neural-net engine buys much more search per wall-clock second than CPU
Stockfish. That decision is sound and is not in question. The defect is that
**no document says so**, and several state the opposite.

Every shipped document claims T1 ground truth is **"Stockfish 17 @ depth 22"**.
Both halves of that claim appear to be wrong.

**Evidence — depth.** `config/config.yaml:5` reads
`depth: 15  # Reduced from 22 for ~4x speedup`, and
`git log -S'depth: 15' -- config/config.yaml` places that line in the **initial
commit** (`73c74e0`, 2026-03-17). No committed config has ever specified depth
22. The only artefact showing depth 22 is `results/logs/run_20260316_132053.json`
— a 2-model, 50-position-per-tier pilot that predates the repo's first commit.
The second pilot log (2026-03-17) already shows depth 15.

**Evidence — engine.** Four independent signals point to Lc0, not Stockfish,
as the source of `stockfish_eval`:

1. `scripts/precompute_lc0_batch_v2.py:192-193` writes Lc0 ONNX WDL-derived
   centipawns into the field literally named `stockfish_eval`, and sets
   `stockfish_best_move = None`.
2. All **10,665** rows of `results/evaluations_retried.jsonl` carry
   `stockfish_best_move: null`. That field is read by
   `retry_illegal_moves.py` from the source record's `t2_best_move`, which
   `worker.py` populates from the dataset's `stockfish_best_move`. A Stockfish
   precompute writes a SAN string; an explicit `null` is the signature of
   `precompute_lc0_batch_v2.py`.
3. The A2 mate-truth error table (mean 9,966 / median 9,999) is consistent with
   truth pinned at **exactly ±10,000** — the saturation constant in
   `wdl_to_centipawns()` — and not with Stockfish's variable
   `10000 − mate_in × 10`, which would spread those values out.
4. The **"±16,000 cp"** figure quoted in A1 and A2 is not producible by either
   encoding. Both cap at ±10,000. It appears to be simply an error, and it has
   been repeated as fact through two audit passes.

**Why it matters.** `docs/FINDINGS.md` builds a whole "Two engines" bullet on
the T1-Stockfish / T2-Lc0 distinction, including *"the two engines disagree on
the best move on ~67% of positions."* If T1 truth is also Lc0, that bullet
describes a comparison that never took place, and the headline framing of the
benchmark misidentifies its own instrument. This is the first thing a judge
with engine knowledge will probe.

**Fix — documentation and provenance only. No code change, no data rewrite.**

1. **Verify first, before editing a single document.** On the Linux PC:
   ```bash
   python3 -c "import json; d=json.load(open('data/easy.json')); \
     print({k: d[0].get(k) for k in ('stockfish_eval','stockfish_best_move')})"
   grep -o '"stockfish_best_move": null' data/*.json | wc -l
   python3 -c "import json; \
     m=max(abs(p['stockfish_eval']) for t in ('easy','medium','hard','extreme') \
     for p in json.load(open(f'data/{t}.json')) if p.get('stockfish_eval') is not None); \
     print('max |truth| =', m)"
   ```
   A null `stockfish_best_move` confirms the Lc0 path. The third command
   settles the ±16,000 question outright.
2. **Do not rename the stored fields.** `stockfish_eval`, `t1_stockfish_eval`
   and `stockfish_best_move` are read by `evaluator.py`, `worker.py`,
   `metrics.py`, `result_writer.py`, `dashboard/server.py`,
   `retry_illegal_moves.py` and `validate_responses.py`. Renaming them means
   rewriting the 526k-record primary data file and touching seven modules for
   zero analytic gain, on the eve of a submission. Record them as **legacy
   names with documented contents** instead. Revisit only if there is time to
   spare after everything else is green.
3. Add a `provenance` block to `summary.json` (in `metrics.save_metrics`, which
   already writes that file). This also closes the D8 provenance gap:
   ```json
   "provenance": {
     "generated_at": "<iso8601>",
     "git_sha": "<rev-parse HEAD>",
     "t1_truth_engine": "lc0-onnx (WDL->cp, saturates ±10000)",
     "t2_truth_engine": "lc0 @ 800 nodes (enrich_cpl.py)",
     "stockfish_depth_config": 15,
     "stockfish_used_for": "none in the shipped 526k run — see D9",
     "legacy_field_note": "fields named stockfish_* hold Lc0 values",
     "config_snapshot": { }
   }
   ```
4. Correct the engine and depth claims in **all five** documents: `README.md`,
   `docs/FINDINGS.md`, `docs/SIMPLE_SUMMARY.md`, `specs/benchmark.md`,
   `specs/benchmark-detailed.md`. State the real engine, the real budget, and
   **the reason** — GPU throughput — which is a defensible methodological
   choice and reads far better than a silent inconsistency.
5. Rewrite the "Two engines" bullet in FINDINGS' methodology section. Either
   drop the 67%-disagreement claim or re-derive it honestly, depending on
   step 1.
6. Correct the "±16,000 cp" figures in A1/A2 to the measured maximum from
   step 1.

**Verification**
- `test_summary_has_provenance` — `summary.json` carries a `provenance` block
  with non-null engine fields and a git SHA.
- `test_no_stale_engine_claims_in_docs` — no shipped `.md` contains the strings
  `"depth 22"` or `"Stockfish 17"`.

---

### D10 — The 526k records span two worker versions; legality is not uniformly defined (P0)

**Status:** Verified from git history and from the shipped retry file.

The silent illegal-move rescue at `src/worker.py:208-223` **was not present for
the whole collection run.**
`git log -S'extract_move_from_text' -- src/worker.py` returns exactly one
commit: `1e799bb`, 2026-04-09, *"chore: catch up two weeks of code changes"*.
Collection ran from roughly 2026-03-18 to the 2026-04-28 metrics refresh. So an
unknown fraction of the 526,662 records was collected with **no rescue at all**.

A6 and D6 both describe legality as "post-rescue" for every combined-prompt
record. That is only true after the cutover. Before it, `t2_legal` is a raw
single-attempt result.

**Why this is P0 rather than a footnote.** The reported 97.85% is an average
over two different measurement procedures. It is therefore neither a
single-shot rate nor a harness-assisted rate, and answers no well-posed
question. Because the commit is a *catch-up* commit covering two weeks of work,
the git date is only an upper bound on when the code went live — so the
boundary cannot be taken from git alone.

**The boundary is recoverable from the data itself.** Under the post-rescue
code path, `parsed["move"]` is only ever assigned from
`extract_move_from_text()`, which returns a legal SAN move or `None`. A stored
`t2_move` that is **non-null and illegal is therefore impossible after the
cutover.** Records matching `(t2_move.notna() & t2_legal == False)` are
necessarily pre-rescue.

At least **7,366** such records already exist in a committed file:
`results/evaluations_retried.jsonl` has 10,665 rows, of which 3,299 have
`original_move: null`, leaving 7,366 with a stored illegal move.

**Fix — all post-processing over data already on disk. No inference.**

1. Classify each record. Mark `regime = "pre_rescue"` where
   `t2_move.notna() & (t2_legal == False)`. Take the **maximum `timestamp`**
   over those confirmed records as the empirical cutover, then classify every
   record by timestamp against it. Cross-check the derived boundary against
   2026-04-09; report both.
2. Emit `results/metrics/regime_split.csv`:
   `regime, model, prompt_format, n, legality` — so the split is inspectable
   per model rather than asserted in prose.
3. Report legality **separately per regime**. Do not present a pooled 97.85%
   again in any document.
4. **Only if** the pre-rescue cohort is large enough and covers enough of the
   22 models, report its legality as the genuine **single-shot** rate. Gate
   this explicitly: publish per-model `n` alongside every figure, and report
   no single-shot number for any model whose pre-rescue cohort falls below a
   stated minimum (suggest **1,000** records). Where coverage fails, say so —
   do not extrapolate across models.
5. If step 4 succeeds with adequate coverage, **D6 step 3 is superseded** —
   the 13,200-call single-shot probe becomes unnecessary and should be struck.
   If coverage is partial, D6 step 3 survives but is rescoped to only the
   models lacking a usable pre-rescue cohort, which will be far cheaper than
   13,200 calls.
6. Document the second consequence of the cutover: `t2_move is None` means
   *"the parser found no move"* pre-rescue and *"both rescue attempts failed"*
   post-rescue. Two different events sharing one encoding, and
   `t2_legal_attempted` silently drops both.
7. Resolve the `move_only` inconsistency this exposes. Under current code
   `move_only` stores `extract_move_from_text()` output, so its legality is
   **100% by construction** post-cutover — yet FINDINGS Finding 1 caveat 2
   quotes 96.5%. That figure can only come from pre-cutover records or from the
   buggy pooled `t2_legal` column. Determine which and correct it.

**Verification**
- `test_pre_rescue_detection` — synthetic df: a record with a non-null illegal
  move classifies as `pre_rescue`; a legal-or-`None` record is not
  force-classified either way.
- `test_regime_split_partitions_all_records` — regime counts sum to `len(df)`;
  no record silently uncounted.
- `test_no_pooled_legality_in_summary` — `summary.json` exposes legality keyed
  by regime, not as a single pooled scalar.

---

### D11 — `learning_deltas.csv` contradicts the "no correction data" claim (P1)

**Status:** Verified. The file is committed and dates to the initial commit.

`results/metrics/learning_deltas.csv` is in the repo with **24 rows**, all
`qwen2.5:7b`, carrying populated `cpl_correction` and `cpl_control` values and
job IDs in the pre-hash sequential format
(`job_00002_qwen2_5_7b_pgn+fen_1`). It is a March pilot artefact.

Meanwhile `src/feedback_loop.py`'s module docstring, `docs/FINDINGS.md` and
`specs/benchmark.md` all state that **zero** records have
`job_type='correction'` and that no correction-loop data exists.

**Root cause of the survival.** `metrics.save_metrics()` writes that CSV only
`if not learning_df.empty` (`src/metrics.py:863-865`). With the real data the
frame *is* empty, so the write is skipped and the March file is left untouched
on disk — through every regeneration since.

A reviewer who opens `results/metrics/` finds correction-loop data sitting
directly beside the sentence saying none exists.

**Fix**

1. Delete `results/metrics/learning_deltas.csv`.
2. Make the skip explicit rather than silent: in `save_metrics()`, when
   `learning_df` is empty, **remove** any existing `learning_deltas.csv`
   instead of leaving a stale one. Audit `save_metrics()` for any other
   conditionally-written artefact with the same hazard.
3. Resolve the tracking inconsistency this exposes: `results/metrics/` and
   `results/evaluations_retried.jsonl` are listed in `.gitignore` but are
   **tracked in git** (ignore rules do not apply to already-added files). Decide
   deliberately whether these are committed artefacts or generated output, then
   make `.gitignore` and the index agree.

**Verification**
- `test_empty_learning_deltas_removes_stale_file` — write a stale CSV, call
  `save_metrics()` with a correction-free df, assert the file is gone.

---

## Steps / Logic

Ordered. D1–D3 and D9–D10 gate publication; do them first.

Everything below is postprocessing, documentation, or housekeeping. **No step
requires re-running the benchmark.** Do not re-collect data for any of it.

0. **D9 verification first — 5 minutes, and it changes what every other step
   says.** Run the three commands in D9 step 1. Until you know which engine
   produced `stockfish_eval` and what the true max |truth| is, you cannot write
   a correct methodology section, and D7's "two engines" claim is unresolved.
   Do this before touching any document.
1. **D3** — live code bug, cheapest fix. Correct `parse_model_info`, add the
   guard assertions, regenerate `by_model_family.csv`. ~1 hour.
   ⚠ Budget extra: `tests/test_utils.py` currently **asserts the buggy
   behaviour** — `test_unknown_family` expects `wizardlm2:7b → "unknown"` and
   `test_float_size` expects `solar:10.7b → family "unknown"`. Both must be
   rewritten as part of the fix, or the suite will fail on a correct
   implementation.
2. **D10 regime split** — classify records, emit `regime_split.csv`, and find
   out whether a usable single-shot cohort exists. Do this early: the answer
   determines whether D6 keeps its GPU probe and whether Finding 1 can be
   rewritten with a real first-attempt number. ~2 hours, no inference.
3. **D2 decisive test** — split direction accuracy by side-to-move. If a
   perspective bug exists, this reveals it immediately and changes what D1
   needs to say. Do not proceed to D1 write-up before this resolves. ~1 hour.
4. **D1** — compute class priors and κ, emit `direction_baselines.csv`, then
   rewrite Finding 3 against whatever the numbers actually show. ~2 hours.
5. **D4** — propagate corrected metrics into H3/H4 and the family table. ~1 hour.
6. **D5** — blunder rate, bootstrap CIs, `cpl_buckets.csv`. ~2 hours.
7. **D6** — relabel legality as harness-assisted; fold in the D10 regime split.
   Strike step 3's probe if D10 step 4 delivered adequate coverage. ~1 hour.
8. **D9 documentation pass** — provenance block, then correct the engine and
   depth claims across all five documents. ~1–2 hours.
9. **D11** — delete the stale `learning_deltas.csv`, make the empty-frame path
   remove it, reconcile `.gitignore` against the git index. ~30 minutes.
10. **D7/D8** — remaining provenance and doc-drift cleanup not already absorbed
    by D9. ~30 minutes.
11. Rewrite `docs/FINDINGS.md` Findings 1/3/5 and the Summary Table.
    Restructure to lead with the bimodal CPL result.
12. Full `pytest` run; commit; push.

---

## Edge cases

- **Positions where Stockfish eval is mate-encoded** (7% of records) — must be
  excluded from Spearman ρ, or the ±16,000 values dominate the rank
  correlation. Reuse the `MATE_SCORE_THRESHOLD_CP = 9000` constant.
- **Models with missing records** — deepseek-r1:7b is short ~1,030 of 24,000
  (~4.3%). Bootstrap CIs must resample within model, not across.
- **κ undefined** when a model emits a single class for every position;
  return `NaN` explicitly and surface it rather than coercing to 0.
- **MoE parameter counts** — `mixtral:8x7b` has 56B total / ~12.9B active.
  Excluded from scaling regressions; flagged, not silently assigned one number.
- **gemma4:e2b / e4b** are "effective parameter" variants; their real memory
  footprint differs from dense models of the same nominal size. Document in the
  scaling section rather than plotting them on the same axis.
- **Ties in `rank_tied`** — CIs overlapping pairwise is not transitive; report
  as a set of statistically indistinguishable models, not a chain.
- **Rerunning the family aggregation** must not silently drop a model — assert
  output group count equals distinct model count (22).
- **`qwen2.5:72b`** is commented out in `config.yaml` (OOM at 32GB RAM/16GB
  VRAM) but listed as pulled in `docs/CHANGES.md`. Reconcile the docs; it is
  not in the 22.
- **Records with no `timestamp`** (D10) — classify as `unknown` regime and
  report the count; never silently assign them to a side of the cutover.
- **Records straddling the cutover within one model/tier** (D10) — the pipeline
  processed one model at a time, so most models sit wholly on one side. Expect
  a small number of models split mid-run; report their split rather than
  assigning the model to a regime.
- **A pre-rescue cohort that covers only some models** (D10 step 4) — report
  single-shot legality only for covered models, and state explicitly which
  models have no single-shot figure. Do not average across an unbalanced set
  and present it as an all-model rate.
- **`enrich_cpl.py` ran after the cutover** (D10) — CPL enrichment is
  regime-independent (it re-evaluates stored legal moves post hoc), so the
  clamped-CPL and WP-loss results are unaffected by the split. Only legality
  and the meaning of a null `t2_move` are regime-dependent. State this, so a
  reader does not assume D10 contaminates the bimodal result.
- **Deleting `learning_deltas.csv`** (D11) — it is tracked in git despite the
  `.gitignore` entry, so removal needs `git rm`, not just an `rm`.

---

## Dependencies

- Python 3.11+, `pandas`, `numpy` (already in `requirements.txt`)
- **`scipy`** — new. Needed for `scipy.stats.spearmanr` and bootstrap CIs.
  Add to `requirements.txt`.
- `sklearn.metrics.cohen_kappa_score` and `balanced_accuracy_score`, **or** a
  hand-rolled κ (~15 lines) to avoid a large new dependency. Prefer
  hand-rolled; it is trivial and keeps the dependency surface small.
- `results/evaluations.jsonl` — **local to the Linux PC only, not in git.**
  All of this work must happen there.
- D6 step 3 (single-shot probe) additionally needs Ollama + all 22 models
  resident, and ~13,200 inference calls.

---

## Acceptance criteria

1. `pytest` green, including all new tests in `tests/test_metric_defects.py`.
2. No shipped CSV contains a pre-audit metric column without a `_legacy` suffix.
3. `by_model_family.csv` has exactly 22 rows, one per model, none `unknown`,
   none size 0.
4. `summary.json` carries `generated_at`, git SHA, engine versions, config
   snapshot.
5. `direction_baselines.csv` exists; Finding 3 quotes the majority-class
   baseline and κ, not the 33% figure.
6. Every number in `docs/FINDINGS.md` is traceable to a CSV or `summary.json`
   key — none hand-typed.
7. The D2 side-to-move split is resolved and its outcome documented, whichever
   way it lands.
8. No shipped `.md` claims "Stockfish 17" or "depth 22"; every document names
   the engine that actually produced the numbers, with its search budget and
   the GPU-throughput rationale.
9. `summary.json` carries a `provenance` block naming the T1 and T2 engines,
   and flagging `stockfish_*` as legacy field names holding Lc0 values.
10. `regime_split.csv` exists; no document reports a pooled legality figure
    across the pre/post-rescue boundary.
11. The D10 single-shot question is resolved either way — a gated single-shot
    number with per-model `n`, or an explicit statement that the pre-rescue
    cohort is too thin, with D6's probe rescoped accordingly.
12. `results/metrics/learning_deltas.csv` no longer exists, and a correction-free
    regeneration removes it rather than leaving it stale.
13. `.gitignore` and the git index agree on whether `results/metrics/` is
    committed output.

---

## Note on scope

D1–D11 are metric-construction, provenance, and propagation defects. **No fix
in this register requires re-running the benchmark.** Every item is
postprocessing over `results/evaluations.jsonl`, a documentation correction, or
file housekeeping. Do not re-collect data for any of them.

The data collection itself — 526,662 records at 99.75% of planned coverage, no
duplicate `(model, position_id, prompt_format)` keys, seeded and stratified
sampling — is sound. The one qualification D10 adds is that it was not
collected under a *single code version*: the worker's rescue logic landed
mid-run. That is a defect in how the results are described, not in the results
themselves, and it is fully characterisable from the stored `timestamp` field.
It is also an opportunity — the pre-rescue cohort is genuine single-shot data,
which is the number D6 assumed had to be bought with GPU time.

The **bimodal move-quality result** (17.8% near-perfect, 39.1% catastrophic,
flat across 2B→70B) is the study's real contribution and is the finding least
disturbed by these fixes — including D9 and D10, since CPL enrichment is
regime-independent and its engine (Lc0 @ 800 nodes) was always correctly
documented. The restructured writeup should lead with it.

---

## Appendix — smaller items found in the same 2026-08-13 read

Not triaged into the register. Recorded so they are not rediscovered later.

| # | Item | Where |
|---|---|---|
| A | Plots still use the legacy `t1_direction_correct_mean` (±50 cp) **and draw the invalid "Chance (33%)" reference line** — D1 is baked into the committed charts, not just the CSVs. D4 lists H3/H4/family-agg but omits the plotting layer. | `scripts/generate_plots.py:153, 229` |
| B | `by_source.csv` is single-valued — every row is `lichess_puzzles`. The three-source design (Lichess / real-game / generated), `aggregate_by_source`, the "Source Comparison" plot and detailed-spec acceptance criterion are all vestigial. `dataset_builder.py` still carries the unused PGN and generated code paths. | `results/metrics/by_source.csv`, `src/dataset_builder.py` |
| C | `phase` is a pure material heuristic, not real game phase: `_validate_puzzle_row` calls `determine_phase(board, 20)` with a hardcoded move count, since puzzle rows have no PGN. Distribution per model is ~5.7% opening / 25% endgame / 69% middlegame — so **H5, the one supported hypothesis, rests on 1,368 rows per model** labelled by piece count alone. | `src/dataset_builder.py:62`, `results/metrics/by_phase.csv` |
| D | `flask` is missing from `requirements.txt` despite `dashboard/server.py` requiring it. `scipy` will also be needed for D1/D5. | `requirements.txt` |
| E | `docs/SCRIPTS.md` documents four Lc0 scripts (`precompute_lc0.py`, `_batch.py`, `_fast.py`) of which only `precompute_lc0_batch_v2.py` exists, and mislabels T3 as "move quality". | `docs/SCRIPTS.md` |
| F | `docs/RESEARCH.md` is badly stale: "5.8M positions across 19 models" (actual: 4,000 × 22), and its central anti-contamination argument to judges is *"the correction loop is your best defense"* — infrastructure that produced no data. This is the document framing the YSTE submission. | `docs/RESEARCH.md` |
| G | All `bin/*.sh` hardcode `/home/rabrew/Desktop/chess-llm-bench`; `specs/dashboard.md` references a different absolute path (`/mnt/shared/...`). Violates the "no hardcoded paths" rule in `CLAUDE.md`. | `bin/*.sh` |
| H | `test_parse_response_formats.py` defines `test_gemma3_12b` twice in one class — the second silently shadows the first. Its "All 19 models" section also predates the 22-model set and omits gemma4, codellama, yi, command-r, mixtral. | `tests/test_parse_response_formats.py:1231, 1293` |

## Implementation clarification (2026-11-02)

Goal: repair aggregation and expose auditable diagnostics without altering raw
research artifacts. Inputs are existing records and optional explicitly supplied
collection provenance. Outputs may be written to a caller-selected temporary
directory; tests use synthetic frames and temporary paths only.

Logic: validate model metadata; retain model identity in family tables; compute
class priors, Cohen kappa, balanced accuracy, side-to-move splits and Spearman
correlation; propagate corrected metrics into hypotheses; report per-model CPL
buckets and seeded bootstrap intervals. Confidence intervals resample positions
as clusters within each model, because repeated prompt responses are dependent.
Old metric columns remain available in memory for consumers but CSV exports
mark superseded fields with `_legacy`. Empty correction outputs remove stale
files only inside the requested output directory.

Scientific edge cases supersede contradictory prescriptions above: constant
predictor kappa is zero unless expected agreement is one (then undefined).
Threshold accuracy need not be monotonic. Rank correlation does not depend on
outlier magnitudes. The last stored illegal move is evidence of a pre-rescue
record, not a verified deployment boundary. Historical legal/null records stay
unknown without explicit collection metadata; timestamp bins are labelled
inferred and never presented as genuine single-shot cohorts. Provenance must
separate current analysis config/git SHA from unknown historical engine versions.
No invented provenance or automatic raw-data rewrite is permitted.

Dependencies: existing numpy/pandas plus scipy for Spearman; bootstrap seed 42,
500 replicates by default, with replicates/seed recorded in outputs. Tests cover
missing values, invalid model tags, same-size family collisions, degenerate
classes, uncertain regimes, reproducibility and stale-output cleanup.

### Endpoint evidence and safe analysis entry point

Historical `t2_cpl` and separately sourced `t1_stockfish_eval` do not uniquely
identify the T2 engine endpoints. No inverse reconstruction is valid without
recorded before/after values. Exact corrected quality metrics require
`t2_reference_eval_before`, `t2_reference_eval_after`, and either
`t2_reference_white_to_move` or valid FEN. Missing endpoints produce unavailable
values; raw CPL remains explicitly historical/legacy. Prompt applicability
excludes unsolicited T1/T3 fragments and includes null failed answers in the
move-request denominator. An explicit-output CPU script generates audit tables
and strict JSON with unknown historical provenance; source/config fingerprints
identify the analysis inputs without claiming they identify collection settings.

### Missing hypothesis evidence (review correction)

A missing corrected metric is unavailable evidence, not a failed hypothesis.
H2 reports null support and insufficient_data unless all four difficulty means
are observed for its primary metric. H4 records per-family, per-metric counts
at the compared smallest/largest eligible sizes; missing required measurements
produce null comparisons and null family/overall support with insufficient_data.
A testable negative comparison remains false. A family without two eligible
sizes supplies no comparison and cannot establish support by an empty all().

### Explicit best-move-zero scoring convention

Version `lc0_endpoints_v1_best_move_zero` assigns zero quality loss to a legal
stored move matching the reference best move without inventing an after-score.
Accept this only with a true convention flag, true legality, matching nonempty
moves, Lc0Engine reference identity, positive node budget, and White-centipawn
units. Its status is `best_move_zero_convention`, distinct from measured endpoint
pairs. Include these rows in quality denominators and publish measured-endpoint,
convention-zero and unclassified quality counts separately. Historical rows
without this explicit provenance remain unavailable.
