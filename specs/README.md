# Specs

Every feature and experiment has a spec, written **before** its code and before its results. Each
one gives the goal, inputs and outputs, the logic, edge cases and dependencies. Where an experiment
has success criteria, they are fixed in the spec before it runs. Fine-tuning specs are in
[`../fine-tuning/specs/`](../fine-tuning/specs/).

## 1. The benchmark itself

| Spec | What it covers |
|---|---|
| [benchmark.md](benchmark.md) | Overview: models, positions, tasks, hypotheses |
| [benchmark-detailed.md](benchmark-detailed.md) | Full benchmark design, written as a paper-style method |
| [isolated-prompt-formats.md](isolated-prompt-formats.md) | The single-task prompts (`eval_only`, `move_only`, `explanation_only`) |
| [retry-illegal-moves.md](retry-illegal-moves.md) | Re-asking for a move when the first answer is illegal |
| [model-registry.md](model-registry.md) | Explicit family and size for every model |
| [puzzle-metadata-join.md](puzzle-metadata-join.md) | Joining the full Lichess theme list back to each position |

## 2. Experiments

| Spec | Experiment | Status |
|---|---|---|
| [finetuned-benchmark-run.md](finetuned-benchmark-run.md) | Benchmarking the fine-tuned models with base-identical settings | Done |
| [engine-strength-check.md](engine-strength-check.md) | Re-scoring moves with a 10× stronger engine (T2-5) | Done |
| [plain-board-prompts.md](plain-board-prompts.md) | The notation check: plain-English board instead of FEN | Running |
| [commercial-models.md](commercial-models.md) | Commercial API models | Planned, not built |

## 3. Scoring: evaluation (T1)

| Spec | What it changes |
|---|---|
| [t1-headline-rho.md](t1-headline-rho.md) | Spearman ρ as the headline T1 number |
| [t1-zero-baseline-and-ft-comparison.md](t1-zero-baseline-and-ft-comparison.md) | Comparing errors with always answering 0 |
| [direction-baseline-kappa.md](direction-baseline-kappa.md) | Chance-corrected "who is winning" agreement (κ) |
| [t1-side-to-move.md](t1-side-to-move.md) | Checking models don't report from the wrong side's view |
| [relative-error-metric.md](relative-error-metric.md) | Relative error for the difficulty hypothesis (H1) |

## 4. Scoring: moves (T2)

| Spec | What it changes |
|---|---|
| [cpl-enrichment.md](cpl-enrichment.md) | Computing centipawn loss for every move |
| [cpl-eval-before.md](cpl-eval-before.md) | Measuring both sides of the CPL from the same engine search |
| [t2-blunder-rate.md](t2-blunder-rate.md) | Blunder rate and CPL buckets as the main move-quality numbers |
| [t2-random-baseline.md](t2-random-baseline.md) | Best-move rate against a correct random baseline |
| [t2-move-source.md](t2-move-source.md) | Legality of the model's own first answer, before any re-asking |

## 5. Scoring: explanations (T3)

| Spec | What it changes |
|---|---|
| [t3-whole-word-matching.md](t3-whole-word-matching.md) | Whole-word theme matching ("mate" no longer matches "material") |
| [t3-side-mark-chance.md](t3-side-mark-chance.md) | Side mark against a majority baseline (κ) |
| [t3-example-finder.md](t3-example-finder.md) | Finding real model phrasings for theme synonyms |

## 6. The audit and the write-up

| Spec | What it covers |
|---|---|
| [artefact-fixes.md](artefact-fixes.md) | The first round of fixes to inflated metrics |
| [metric-defect-remediation.md](metric-defect-remediation.md) | Eleven defects found in the post-audit review |
| [corrected-hypotheses-family.md](corrected-hypotheses-family.md) | Hypotheses and the family table on corrected columns |
| [findings-tables.md](findings-tables.md) | FINDINGS tables generated from the metrics, with a staleness check |
| [summary-provenance.md](summary-provenance.md) | Git commit, data hash and engine settings in every `summary.json` |
| [exhibition-readiness.md](exhibition-readiness.md) | The checklist for making every claim defensible |

## 7. Infrastructure

| Spec | What it covers |
|---|---|
| [run-all-tier-tracking.md](run-all-tier-tracking.md) | How `run_all.sh` tracks finished tiers and resumes |
| [repo-relative-scripts.md](repo-relative-scripts.md) | Shell scripts find the repo themselves (no hardcoded paths) |
| [dashboard.md](dashboard.md) | The live progress dashboard |
