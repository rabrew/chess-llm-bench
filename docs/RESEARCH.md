# Research framing — Stripe Young Scientist 2027
## Question
How do local language models evaluate chess positions, choose moves, and explain positional themes across model families and prompt formats?

## Design
The primary file contains 526,662 records from 22 models and 4,000 stratified Lichess puzzle positions. The multi-million-position source archive is not the benchmark sample. This is a single-source tactical-puzzle study; it does not establish general game-playing strength.

## Interpretation
Use class-prior baselines, chance-corrected agreement and side-to-move diagnostics for direction. Reference engines provide estimates under finite budgets, not infallible truth. Historical provenance remains unverified. Exact endpoint-based historical move-quality claims and uniform single-shot legality are unavailable.

Model comparisons require matched position/prompt/trial cohorts, task-specific denominators and compatible reference metadata. Model size is confounded with architecture and training. MoE/effective sizes require separate treatment.

## Contamination and reasoning
Public Lichess puzzles may overlap model training data. Neither a large source archive, loss magnitude, nor a plausible explanation rules out memorization. No primary correction-loop cohort was collected. Do not present that inactive infrastructure as evidence of reasoning. Future held-out/newly generated studies could investigate this, but no new experiment was launched during repair.

## Fine-tuning
Dedicated held-out T0 descriptions now persist precision/recall/F1 and exact-board correctness. These descriptive callback outputs are separate from downstream benchmark records and are not automatically a matched base-versus-fine-tuned cohort. Transfer claims require matched downstream records and compatible metric evidence.

See [FINDINGS.md](FINDINGS.md) and [EXPERIMENT_REPAIRS.md](EXPERIMENT_REPAIRS.md).
