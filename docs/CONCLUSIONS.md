# Project conclusions after the analysis fixes

**Project:** Ryan Brew, PR09426 — *Do LLMs Reason? Benchmarking Logical Performance via Chess*

**Snapshot:** 1 November 2026, including scoring of existing Claude responses

**Status:** Base and fine-tuning results analysed; plain-board incomplete; Claude scored, local-reference comparison pending.

## Main conclusion

The tested local language models possess some chess knowledge, but their
evaluation and move selection are unreliable under this benchmark. Board-focused
fine-tuning improved board-description behaviour without reliably improving
overall chess performance, and often reduced it. The available Claude results
show stronger competence on evaluation and first-answer legality, and measurable
but limited move-selection competence. The matched local comparison is pending.

These behavioural results are consistent with fragile pattern-based performance.
They do not identify the models' internal reasoning mechanism, prove that LLMs
cannot reason, or establish that chess fine-tuning generally cannot work.

The fixes did not reverse the base-benchmark point estimates. Their main effects
are more realistic uncertainty, corrected reporting denominators, better control
of selection effects, and explicit pending states for unfinished experiments.

## 1. What changed after the fixes

The base reanalysis covers the same 526,662 responses, 22 models and 4,000
positions as the saved benchmark. No new model answers or engine evaluations
were generated.

| Base measure | Previously saved | Post-fix reanalysis |
|---|---:|---:|
| Blunder rate among final legal/scored moves | 41.86% | 41.86% |
| Blunder-rate 95% confidence interval | 41.69–42.02% | **40.74–42.97%** |
| Pooled evaluation Spearman correlation | 0.110 | 0.110 |
| Non-mate evaluation error / constant-zero baseline error | 3.04× | 3.04× |
| Engine-best-move rate / uniform-random legal-move rate | 1.16× | 1.16× |
| Incorrectly reported failure rate over all prompt formats | 35.36% | Not an applicable move-failure measure |
| Failed move requests / prompts that request a move | — | **3.04%** |

The former all-format failure rate included eval-only and explanation-only
prompts, which requested no move. Removing those prompts corrects the
denominator; it does not improve the models. Final assisted legality remains
96.96%, or approximately 97%.

The corrected pooled confidence interval is approximately **6.74 times wider**.
It resamples whole positions, retaining responses associated with each sampled
position, rather than treating repeated responses as independent observations.
Per-model intervals are approximately **1.71–1.94 times wider**. The bootstrap
uses 10,000 resamples and seed 42; the pooled interval covers 4,000 positions.

All per-model blunder point estimates reproduce to floating-point precision.
Exact adjacent rankings need paired inference. Overlapping individual confidence
intervals establish neither a statistical tie nor equivalence.

Other fixes correct heatmap colour alignment and dashboard scoring. The dashboard
now distinguishes corrected measurements from legacy records lacking enough
evidence; those display corrections do not change the stored model answers.

## 2. What the completed base experiment supports

### Weak evaluation, with some genuine association

Pooled evaluation correlation is **0.110**. Non-mate mean absolute error is
**142.74 centipawns**, compared with **46.96 centipawns** for always answering
zero. Every tested model is worse than that baseline on this error measure.

This supports weak and poorly calibrated evaluation, rather than no chess
knowledge. Individual model correlations differ, and a pooled correlation can
be affected by differences in their output scales. It must not be interpreted as
every model having exactly the same ability.

### Modest move-selection advantage and substantial unreliability

Models select the reference engine's best move **10.22%** of the time, against
**8.82%** for a uniform random legal move on the same evaluated positions. That
is an advantage of approximately **1.16×**. Some knowledge is present, but this
does not amount to dependable play.

Among final legal moves with engine scores, **41.86%** lose at least 1,000
clamped centipawns. This is conditional on a legal, scored output; illegal or
unanswered attempts are not included in that blunder denominator. Final moves
can also include parser recovery and legal-list-assisted retries.

Gemma4:31b and Gemma4:26b retain the lowest blunder point estimates, approximately
**35.8%** and **37.3%**. Their corrected intervals are **34.51–37.11%** and
**35.97–38.55%**, respectively. These are point-estimate rankings, not a new
paired significance result.

The reference-strength check supports broad robustness: increasing Lc0 search
from 800 to 8,000 nodes on 1,000 positions produced a model-ranking correlation
of approximately **0.97**. It does not establish every adjacent rank.

### Legality and explanation need their denominators stated

The approximately **9.3% unaided first-answer legality** figure applies to the
**five rerun models on combined prompts**, not all 22 models. Approximately 97%
final legality is a different, assisted measurement.

Side-identification accuracy is **30.34%**, compared with a **54.21%** majority
baseline; Cohen's κ is **0.067**. Verbal side identification is therefore weak.
Theme-keyword scoring is a limited proxy for explanatory quality, not an
independent demonstration of sound reasoning.

There is no consistent monotonic size benefit across the tested model families.
Some Gemma comparisons improve with size. Differences between architectures,
training histories and model configurations prevent a general causal claim that
parameter count has no effect.

## 3. What fine-tuning achieved—and what it worsened

**Fine-tuning did not damage the benchmark run, overwrite the baseline model
answers, or invalidate the existing research data.** It changed the separately
fine-tuned models, whose downstream chess performance often became worse.

The training taught board-description behaviour. High piece recall is evidence
of that learned task, but it is not the same as a perfectly described board or
general spatial understanding. Exactness and performance depend on the prompt;
extra hallucinated pieces must be assessed with precision and exact correctness,
not recall alone.

The following pooled descriptive comparisons use matched position/prompt records
for five models. A legal non-blunder success means a requested move was legal,
received an engine score, and had clamped CPL below 1,000. Here no legal outcomes
in these comparisons were left unscored.

| Intervention | Final legality, base → fine-tuned | Legal non-blunder successes / all matched move requests | Blunders on identical legal/scored pairs |
|---|---:|---:|---:|
| Pieces-only | 95.88% → 76.89% | **56.21% → 43.96%** | 42.36% → 42.77% |
| Pieces + legal moves | 95.88% → 66.90% | **56.21% → 39.23%** | 41.04% → 41.29% |

There are 79,981 matched move requests in the pieces-only comparison and 79,972
in the legal-move comparison. Their shared legal/scored subsets contain 59,203
and 51,587 pairs, respectively. The base matched-subset rates differ between
interventions because each selects a different shared subset.

Neither intervention demonstrates improved reliable overall play. Some
individual conditional measures improve slightly, so it would be inaccurate to
say every model worsened on every measure. These supplementary comparisons are
descriptive; no new significance claim or retrospective success criterion is
being introduced.

### Why Gemma3's apparent improvement is misleading without coverage

For Gemma3:4b's pieces-plus-legal-moves fine-tune:

| Measure | Base | Fine-tuned |
|---|---:|---:|
| Blunders on each model's own legal/scored subset | 42.95% | 31.81% |
| Blunders on the same 4,671 legal/scored pairs | **35.28%** | **31.56%** |
| Final legality | **98.74%** | **29.45%** |
| Legal non-blunder successes / all matched move requests | **56.33%** | **20.08%** |

The apparent blunder reduction is **11.14 percentage points** on separately
selected subsets, but only **3.73 points** on identical legal/scored pairs.
Much of the apparent gain therefore reflects selection. The remaining
conditional gain does not establish better overall play, because successful
legal non-blunder responses become much less frequent.

The legal-move training intervention also failed its registered training-quality
criterion. Legal-list recall was approximately **65–81%**, below the required
90%. It cannot establish whether accurately learning the rules would transfer
to better play.

The defensible fine-tuning conclusion is:

> These fine-tuning methods improved board-description performance but failed
> to improve reliable overall chess performance, often making it worse. This
> is a useful negative result about these interventions, not evidence that
> chess fine-tuning in general cannot work.

## 4. Plain-board formatting: an incomplete experiment

The run is paused after codellama:34b. There are **20,991 saved responses** across
14 models and 500 positions. Thirteen models have 1,500 responses each;
WizardLM has 1,491. Seven configured models remain unrun, and all CPL values are
missing.

| Paired measure, current 14-model cohort | FEN | Plain board | Usable pairs |
|---|---:|---:|---:|
| First-answer legality | 16.48% | **23.39%** | 6,991 |
| Evaluation Spearman correlation | 0.0963 | **0.1359** | 6,774 |
| Side-identification κ | 0.0736 | **0.0786** | 6,740 |

Plain-board first-answer legality has a position-bootstrap 95% interval of
**22.17–24.67%**. The current N2, N3 and N4 point estimates are below their
registered thresholds. **N1 cannot be evaluated, and the overall verdict remains
pending.** The 1,000-resample, seed-42 intervals and registered criteria are
unchanged.

Providing the legal-move list raises paired first-answer legality from
**23.38% to 88.83%**. That demonstrates answer-list assistance, not high-quality
move selection; CPL is still unavailable.

Model responses are heterogeneous. Qwen2.5:32b's paired evaluation correlation
rises from **0.0836 to 0.3300**, while Gemma3:12b falls from **0.3535 to 0.0844**.
The pooled result cannot stand in for every model.

### The new snapshot is not a scoring reversal

Recalculating the handoff's original 13-model cohort reproduces its rounded
numbers: first-answer legality **16.85% → 24.57%**, correlation
**0.0970 → 0.1381**, and side κ **0.0785 → 0.0858**. Adding CodeLlama, whose
plain-board first-answer legality is 8%, largely explains the lower 14-model
pooled plain-board legality. Do not attribute that cohort change to bug fixes.

## 5. Claude: stronger descriptive competence, incomplete move-quality evidence

Haiku and Opus-low each have **1,500 saved responses**, covering the same 500
positions in all three formats. Sonnet's 48 pilot responses are excluded from
these full-variant comparisons.

| Variant | First-answer legality, FEN → plain | Paired evaluation correlation, FEN → plain | Paired side κ, FEN → plain |
|---|---:|---:|---:|
| Haiku | 88.8% → 86.6% | **0.4823 → 0.4752** | 0.2580 → 0.2430 |
| Opus-low | 93.8% → 92.6% | **0.5769 → 0.5296** | 0.3625 → 0.2921 |

Haiku's evaluation comparison has 467 usable pairs; Opus has 500. With a legal
list, first-answer legality reaches **99.2%** and **100%**, respectively. Haiku
has 41 token-capped initial replies; Opus has none. Complete record coverage
does not mean no responses were truncated.

These results show stronger descriptive evaluation association and unaided
legality than the current local-model results. The paired evaluation, side
agreement and first-answer legality measures above do not improve with
plain-board formatting. This is not a completed move-quality superiority
result or proof of an internal reasoning mechanism.

### Completed Claude scoring

All **3,048 legal moves**, including Sonnet pilots, now have finite nonnegative
CPL from the configured 800-node Lc0 search. Record identities and original
responses were preserved. No new API requests were made.

Full-variant FEN blunder rates are **29.0% (Haiku)** and **19.2% (Opus-low)**;
plain-board rates are **24.4%** and **21.2%**. FEN best-move rates are **17.8%**
and **26.8%**, against an **8.04%** uniform random-legal baseline (2.21× and
3.33×). These are assisted final-move measures; first-answer legality is
reported separately above.

**C1 remains pending:** the matched Gemma4:31b reference is unfinished.
**C2/C3 are “not run”:** pilots do not replace the full Sonnet experiments.
**C4 is complete for both variants: “notation was a major barrier”.** This
is exactly the registered rule's result, triggered by its absolute plain-board
correlation threshold of 0.30. It does **not** establish improvement over FEN:
both paired correlations decrease, and neither variant satisfies the large
blunder-improvement branch. Keep this limitation alongside the verdict.

The existing ledger records **$12.2555896** spent, including pilots and retries:
Haiku $0.8918706, Opus-low $11.058668, Sonnet pilots $0.305051 combined. Offline
scoring added no API cost. Claude ran at API defaults; temperature could not be
set. Parameter counts are unknown and public-puzzle contamination is possible.

The committed [summary](https://github.com/rabrew/chess-llm-bench/blob/1da3d14/results/claude/metrics/claude_summary.json)
and [CSV](https://github.com/rabrew/chess-llm-bench/blob/1da3d14/results/claude/metrics/claude_comparison.csv)
include provenance. They remain on the separate branch until experiment 6
finishes; a final matched local-versus-frontier conclusion awaits C1.

## 6. What can—and cannot—be claimed

Supported by the available results:

- The tested local models show limited, unreliable chess competence under this
  harness, with weak evaluation and substantial assisted-move blunders.
- The tested board-focused fine-tuning interventions did not reliably improve
  downstream play, despite teaching board-description behaviour.
- Formatting helps local first-answer legality modestly in the current partial
  cohort, with heterogeneous evaluation effects.
- Claude shows stronger competence on the currently measurable tasks.
- The analysis fixes chiefly change uncertainty and interpretation, rather than
  reversing the completed base-benchmark point estimates.

Not established:

- That LLMs cannot reason, or that a particular internal mechanism caused failure.
- That fine-tuning generally cannot improve chess performance.
- A final local notation verdict, matched Claude move-quality advantage, or thinking benefit.
- Statistical equivalence between adjacent model ranks.
- That public-puzzle contamination, harness assistance, output parsing, reference
  limitations, prompt dependence or selective missingness have been eliminated.

The 12-position GPT illustration remains illustrative only, not a comparable
benchmark or evidence for a population-level conclusion.

## 7. Evidence and reproducibility

This document is a manual post-fix snapshot. It does not replace the registered
specifications or claim that the older generated tables have been overwritten.

Authoritative existing inputs and methods:

- [Base findings](FINDINGS.md) and local `results/evaluations.jsonl`.
- Local fine-tuning inputs under `results/finetuning_experiment/`:
  `evaluations_base_rerun.jsonl`, `evaluations_ft.jsonl`, and
  `evaluations_legal_ft.jsonl`.
- [Fine-tuning comparison code](../fine-tuning/scripts/compare_finetuning_results.py)
  and [board-description evaluation](../fine-tuning/scripts/board_reading_test.py).
- [Current metric definitions](../src/metrics.py),
  [plain-board comparison](../scripts/plain_board_compare.py), and
  [post-results correction specification](../specs/analysis-review-corrections.md).
- [Registered plain-board criteria](../specs/plain-board-prompts.md).
  Claude criteria and responses remain on `claude-models` in the separate
  `~/clb-claude` checkout; the branches have not been merged.

The recalculation artifacts are temporarily stored in
`/tmp/chess-postfix-analysis.MgxGMJ/`: `base_summary.json`,
`base_before_after.csv`, `ft_report.json`, `ft_pooled_descriptive.json`,
`partial_plain_summary.json`, `partial_plain_same13_points.json`, and
`partial_claude_summary.json`, with their analysis scripts and supporting CSVs.
They include source fingerprints. Temporary files may be removed; the conclusions
and essential numerical comparisons are preserved here.

Reanalysis used CPU-only current analysis functions. It skipped large dataset-tier
lookups because every input record already contained its FEN; supplementary puzzle
metadata was not required for the comparisons reported here. Claude model metadata
was adapted only inside that analysis process, without changing the registry.

Independent numerical review checked the results and denominator interpretations.
The underlying corrected implementation passed **1,132 tests**, Ruff and the
findings-table consistency check. Those checks establish implementation coverage,
not the scientific validity of every interpretation.

The initial CPU-only reanalysis did not modify research inputs. Subsequent
authorized Claude enrichment added engine scores while preserving original
responses, with a committed validation record on `claude-models`. No paid
calls were made. Plain-board remains paused until Ryan requests resumption.
