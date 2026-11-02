# Audited reporting
## Goal
Use corrected metrics across plots and dashboard, preserving unknown historical evidence.
## Inputs / Outputs
Metric CSVs and dashboard JSONL input; charts and JSON responses with explicit definitions.
## Steps / Logic
Plot kappa and Spearman, never fixed chance baseline. Quality needs actual endpoints. Sort heatmap colors and labels together. Share T3 and endpoint derivation; exclude mate truth from MAE.
## Edge cases
Missing quality renders unavailable; old files require regeneration; empty and reordered rows retain identity.
## Dependencies
pandas, matplotlib, Flask; temporary fixtures and mocked files only.

Future endpoint-quality summaries include explicitly labelled, validated engine-best-move zero conventions in their denominator and report their count separately from measured endpoint pairs. This convention is not a measured after-move evaluation. Historical records without either form remain unavailable.
