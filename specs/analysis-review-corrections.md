# Analysis and reliability corrections after the handoff review

## Goal and scope

Correct implementation defects and disclose statistical limitations identified
after reviewing the processed results. This is a post-results methods amendment,
not a new experiment or a replacement for the registered N1–N4/C1–C4 criteria.
The paused plain-board run must remain paused. Preserve its prompts, sampled
positions, model settings, responses, checkpoints, and all existing result files.

## Inputs and outputs

Inputs are existing result records, configuration, and temporary synthetic test
fixtures. Outputs are corrected analysis code, regression tests, and explanatory
documentation. Validation must use CPU-only mocks and temporary files. Do not
regenerate research artifacts, call model services, enrich CPL, or spend money.
Old processed artifacts retain their original methods until explicitly rebuilt;
documentation must not present their row-based intervals as newly corrected.

## Steps and acceptance criteria

1. Establish the full pytest/Ruff baseline. Write regression tests before code.
2. Plain-board summaries must handle empty, partial, or unenriched results:
   unavailable measures and the overall verdict are explicitly pending, never
   silently mixed or successful. Complete inputs keep the registered rule.
3. Use the same validity criteria for pairing, measures, and coverage. Report
   observed, usable, paired, missing/unmatched and duplicate records per measure
   and model/format. Retain existing summary fields where practical. Report
   per-model correlations and paired differences as descriptive supplements;
   do not substitute them for the registered pooled decision statistic.
4. Mark the absolute-correlation notation rule as a limitation: a high board-text
   correlation alone does not show improvement over FEN. Keep its thresholds and
   complete-data outcome unchanged. Claude conclusions remain pending until
   supported, and token-capped responses require explicit disclosure.
5. Blunder intervals must resample positions, keeping all responses for each
   sampled position together. Preserve row-weighted point estimates and seeded
   reproducibility. Report method, number of clusters, and any explicitly labelled
   IID fallback for legacy inputs without position identifiers. Use bounded
   batches to avoid excessive bootstrap memory. Interval overlap is descriptive,
   not a statistical tie or equivalence test.
6. Hallucination rates must count only move-requesting formats, including missing
   or illegal requested answers as failures. Correct heatmap row alignment.
   Dashboard metrics must share corrected CPL/theme/error definitions with the
   analysis layer, disclose legacy fallback provenance, and show T3 on a labelled
   0–2 scale or divide by two when displaying a percentage.
7. Fine-tuning comparisons must disclose legal/CPL coverage and provide a matched
   legal-pair sensitivity analysis alongside existing conditional estimates.
   Illegal/missing answers must not masquerade as improved overall play.
8. Fix remaining data-loader/FEN cache invalidation, square-zero king validation,
   and nondeterministic parallel validation selection without touching datasets.
   Configured Lichess sources default to puzzles only and use the full CSV tier
   quota. `dataset.include_generated` and `dataset.include_pgn` explicitly opt in
   to mixed medium-tier positions; an empty dataset config retains generated-only
   behaviour. API sampling requires a positive finite position limit.
   Stockfish precomputation must treat null reference values and missing/empty
   best moves as incomplete, rather than skipping rows merely because keys exist.
9. Reliability fixes must detect engine EOF and bound protocol waits; refuse GPU
   enrichment while Ollama reports a loaded model; and make worker dry-run leave
   the real queue and correction state untouched. Normal inference prompts and
   rescue behaviour remain unchanged. Test service/process interactions with
   fakes only; never signal or start live processes during development.
   Training failure cleanup must also preserve directories that existed before
   a resumed run, including checkpoints and prior GGUF exports.
10. Review the final diff independently, resolve introduced defects, and run the
    affected and full CPU checks plus read-only findings-renderer validation.

## Edge cases and reproducibility

Cover no CPL, missing columns, invalid side labels, constant predictions,
duplicate position/model/format rows, absent models/formats, unequal legal
coverage, repeated responses at one position, unequal cluster sizes, zero/all
blunders, non-default DataFrame indexes, two dataset directories, EOF, blank UCI
lines, stalled engine output, and service-state query failures. Seeds and method
identifiers must be explicit. Tests use temporary fixtures; no training stack or
network access is necessary.

## Dependencies and scientific limits

Use installed CPU test dependencies, stdlib, NumPy and pandas; add no training or
GPU dependencies. Statistical amendments cannot establish internal reasoning
mechanisms, recover missing engine evaluations, complete unrun models, or remove
public-puzzle contamination. These remain explicit limitations rather than code
claims. No registered success criterion is changed after inspecting results.
