# Chess LLM Benchmark — audited findings
Updated 2026-11-02. This replaces the earlier publication draft, whose direction baseline, engine attribution, legality interpretation, and reconstructed move-quality claims were not supported.

## Scope and evidence
The read-only audit counted 526,662 records, 4,000 unique positions and 22 models.
The selected benchmark positions are stratified Lichess puzzles. The source archive is much larger; archive size is not evaluation sample size. Only this puzzle source was evaluated. Phase labels are material/move-count heuristics, not independently observed game phases.

Figures below are generated from the audit exports. [audit-evidence.json](audit-evidence.json) records the summary and input provenance; [audit-direction_baselines.csv](audit-direction_baselines.csv) and [audit-direction_diagnostics.csv](audit-direction_diagnostics.csv) retain exact denominators and per-model results. Reproduce with:
```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/generate_audit_metrics.py --help
```
Use an explicit new output directory, such as /tmp/chess-audit-corrected. Existing results/ outputs and charts are historical snapshots and must not be cited as the corrected report.

## Direction agreement must beat the class-prior baseline
| Threshold (cp) | Raw accuracy | Position-majority baseline |
|---|---:|---:|
| ±0 | 44.48% | 47.975% |
| ±50 | 46.14% | 54.225% |
| ±100 | 61.60% | 78.550% |
| ±200 | 78.95% | 93.275% |

Raw accuracy pools available eval-request records; position-majority priors count unique positions. Use each model's matched baseline in the diagnostic CSV when comparing incomplete cohorts.

Every model falls below its own matched majority baseline at ±200 cp. The high raw accuracy at that threshold mostly reflects an overwhelmingly equal reference class. Per-model Cohen's kappa and balanced accuracy, plus threshold-free Spearman correlation, replace the former universal chance line. A change in model ranking across thresholds does not itself prove a bug or anti-correlation.

The side-to-move diagnostics do not establish a simple sign-inversion bug for Llama3.3-70B or Gemma4-31B. Their pooled correlations differ from within-side correlations, so the split must accompany interpretation. Raw model response text is absent from the primary file; the requested response-by-response sign audit cannot be completed from those records.

## Exact historical move-quality rankings are unavailable
No primary records retain the actual matched T2 evaluation endpoints. Combining the separately sourced T1 evaluation with stored T2 loss cannot recover those endpoints. Corrected endpoint-clamped CPL, win-probability loss, their bootstrap intervals and blunder buckets therefore remain unavailable.

Earlier claims about a clamped mean near 700, bimodal proportions, exact model ranking, or scaling based on those reconstructions are withdrawn pending recoverable endpoint evidence. The stored raw CPL is retained as legacy data, not presented as the corrected metric. Future enrichment persists measured endpoints and reference settings. An engine-selected best move may retain a separately labelled zero-loss convention without fabricating a measured post-move score.

## Historical legality has no verified single-shot cohort
The audit classifies 175,584 non-move records as not applicable, 7,361 illegal non-null moves as limited per-record evidence, and 343,717 records as unknown.
The last illegal move is not an exact rescue rollout timestamp. The illegal-only evidence subset is selected on failure and cannot estimate a single-shot rate. No pooled historical legality number is published here.

For future records, first parsed move, first-attempt legality, rescue setting and rescue path are stored separately. move_only remains legal-list assisted even with rescue disabled. A missing requested move counts as a failure; eval-only and explanation-only prompts are excluded. Completed-result denominators still exclude model-call and complete-parse failures; all-request rates require those failed jobs too.

## Reference provenance remains unresolved
Selected primary reference values span ±10,000 cp. The much larger easy archive contains values beyond this range, which does not describe the selected benchmark cohort. The primary file also contains non-null best moves, contradicting the older spec's blanket-null premise.

Current preprocessing supports Stockfish search and direct Lc0 ONNX value evaluation; enrichment uses Lc0 search with a configured node budget. GPU throughput motivates the ONNX path, but a script's existence or today's configuration does not prove which path produced every historical score. Exact engine versions, model weights and budgets for collection remain unverified. Fields named stockfish_* are legacy schema names. No claim of a uniform historical engine or depth is justified.

## Explanations and scaling
The rule-based T3 v2 mean is 0.519520 on its defined 0–2 scale (summary overall_metrics.t3_mean_score_v2).
It measures side/theme matching, not independently validated reasoning. Model-family exports now preserve one row per model. MoE total/active sizes and effective-parameter variants are flagged and excluded from dense scaling regressions. Missing exact quality evidence cannot support either a positive or negative quality-scaling claim.

The correction-loop infrastructure produced no primary correction cohort. Historical pilot learning-delta output is stale and is not evidence of learning. A fresh correction-free export omits it; the original protected results snapshot has not been deleted.

## Publication limits
No inference, engine sensitivity experiment, training or dataset rebuild ran during repair. These changes correct future behavior and postprocessing; they do not recover missing historical responses, endpoints or deployment logs. [EXPERIMENT_REPAIRS.md](EXPERIMENT_REPAIRS.md) distinguishes implemented fixes from outstanding evidence and protected publication outputs.

Future endpoint-quality summaries include explicitly labelled, validated engine-best-move zero conventions in their denominator and report their count separately from measured endpoint pairs. This convention is not a measured after-move evaluation. Historical records without either form remain unavailable.
