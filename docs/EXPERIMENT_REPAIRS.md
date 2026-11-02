# Experiment repair status
Updated 2026-11-02. Code changes follow specs → regression tests → implementation, with independent review and CPU validation. No research outputs, model weights, training jobs, services or live experiments were changed.

## Main defect register
| Register | Implemented | Historical/publication limit |
|---|---|---|
| D1–D2 | Position priors, matched baselines, kappa, balanced accuracy, Spearman and side splits; corrected narrative | Raw response sign audit unavailable because primary responses were not stored |
| D3 | Strict model metadata; distinct family rows; MoE/effective-size flags | Dense scaling excludes incomparable size definitions |
| D4 | Corrected family/H3/H4 metrics, explicit unavailable hypothesis results | Historical exact quality unavailable |
| D5 | Position-cluster bootstrap, buckets, medians, overlap sets for measured endpoints | No historical endpoint records; old quality rankings withdrawn |
| D6/D10 | First-attempt/rescue metadata, disable switch, request-based denominator and uncertain regimes | No verified rollout boundary or unbiased historical single-shot cohort; failed-job denominator remains separate |
| D7–D9 | Input hash, analysis revision/time/config, engine uncertainty, future source metadata and endpoint retention | Historical engines/versions/budgets unresolved; sensitivity experiment not run |
| D11 | Fresh exports remove stale learning-delta output | Existing protected results snapshot, including tracked pilot CSV, left untouched |

## October audit
| Finding | Repair |
|---|---|
| 01 | Failed training/export preserves checkpoints and exports; fresh-run collisions rejected |
| 02 | Matched T2 endpoints retained; historical reconstruction removed |
| 03 | Dry workers and CLI use memory queue snapshots; dry-run log writes suppressed |
| 04 | Engine EOF, bounded output, protocol deadline and startup cleanup |
| 05 | Actual T1 schema and missing applicability used in comparisons |
| 06 | Explicit preprocessing reference provenance and valid completion checks |
| 07 | Heatmap colors and labels use the same ordering; plots use audited diagnostics |
| 08–09 | Stable parallel validation order, explicit puzzle-only design, full tier quotas |
| 10 | King square a1 handled with explicit None check |
| 11 | Board precision/recall/F1 and exactness; legacy accuracy labelled recall |
| 12–14 | Matched cohorts and compatible references, no loss-cap shortcut, missing T0 preserved; held-out descriptions persisted separately |
| 15–16 | Shared dashboard metrics with visible limits; comparisons exclude non-requested tasks and include missing requested answers |

Other fixes: duplicate parser test restored; portable shell root resolution; model tag passed as argv rather than injected into Python; missing/corrupt benchmark exclusion fails closed; ambiguous GGUF import rejected. Single-source and heuristic-phase limitations are disclosed. Obsolete preprocessing script references and unsupported contamination arguments were removed.

## Evidence and outputs
Fresh full-data metrics are in `/tmp/chess-audit-corrected` on the Linux PC. Small source tables and summary evidence are retained alongside FINDINGS.md. Corrected charts are generated separately, with a manifest; pre-existing chart names are retired and must not be quoted as current.

The repository has historically tracked generated results despite ignore rules. These are retained as legacy snapshots rather than silently deleting research artifacts. Fresh output belongs in a separate explicit directory. Publishing new metrics in results/ and deleting/replacing historical snapshots remains a deliberate follow-up; code repair does not imply those files were refreshed.

Dedicated held-out T0 callback files are descriptive evidence with their own test cohort. They are not automatically joined to base/FT benchmark rows. No unmeasured T0 or quality delta is reported as zero.

## Reproducibility
Run `make check` for CPU tests and fatal Ruff checks. See DEVELOPMENT.md. Audit bootstrap uses seed 42 and resamples positions within model; repeated prompt rows are not treated as independent samples. Input SHA256 and analysis metadata are in generated summary provenance. Runtime engine versions and training-stack versions were not exercised during this repair.

Future endpoint-quality summaries include explicitly labelled, validated engine-best-move zero conventions in their denominator and report their count separately from measured endpoint pairs. This convention is not a measured after-move evaluation. Historical records without either form remain unavailable.

## Final validation
665 CPU tests passed; fatal Ruff and git diff --check passed. Independent review found no actionable introduced issues. The only warning is the existing python-chess deprecated event-loop API. The chart CLI produced three nonempty PNGs and a manifest in /tmp/chess-audit-charts. No live training or engine validation was performed.
