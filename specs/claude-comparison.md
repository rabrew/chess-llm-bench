# Offline comparison of the completed Claude variants

## Goal

Implement completion item B5 using the already registered C1 and C4 criteria in
[commercial-models.md](commercial-models.md). This document specifies reporting
and implementation only; it changes no experimental success criterion. C2 and
C3 remain **not run**, because the Sonnet configurations have only pilot data.

## Inputs and outputs

Inputs are the scored Claude JSONL, raw response ledger, local plain-board JSONL,
and `config/config_claude.yaml`. Every path is configurable. The default full
variants are Haiku and Opus-low; retained Sonnet pilot records are counted and
disclosed separately, never pooled into the full-variant comparison.

The CLI writes `claude_summary.json` and `claude_comparison.csv` into the selected
metrics directory. Outputs use strict JSON: missing/undefined values are null,
not NaN. The script is entirely offline, imports no API runner/SDK, reads no key,
starts no engine, and queries no model service.

## Steps

1. Read scored records and derive metrics using the existing analysis layer.
   Preserve original records. Count malformed lines, duplicate jobs, missing
   references and incomplete measurements. Use first standard record per
   variant/position/format and report excluded duplicates. Do not silently treat
   pilot coverage or missing observations as complete.
2. Reuse `plain_board_compare.pair_rows` and `bootstrap_ci`. All uncertainty uses
   the registered 1,000 whole-position resamples and seed 42. Pair by position and
   prompt after selecting the variant or local comparator; never require the
   different model tags to be equal. Validate finite references/predictions,
   recognised side labels, and legal/scored CPL before pairing.
3. C1 uses FEN-only Claude versus Gemma4:31b on common legal/scored positions.
   Compute both marginal blunder rates and position-bootstrap confidence
   intervals on that same subset. Report **frontier model plays clearly better**
   exactly when the local-minus-Claude rate is at least 0.05 and the intervals
   do not overlap; otherwise report **no clear difference**. Missing local model,
   missing CPL, incomplete cohort or invalid intervals yields **pending**.
   Coverage of legal/scored pairs is always disclosed.
4. C4 compares FEN-only with board-text for each full Claude variant. Compute all
   four registered N1–N4 measures using the same pairing/bootstrap helpers. Apply
   `plain_board_compare.verdict` unchanged when the cohort and measures are
   complete; otherwise report **pending**. Preserve its absolute board-text rho
   threshold. Explicitly explain that a high absolute rho does not itself show
   improvement over FEN; report paired differences separately.
5. Also report by variant and format: observed/usable coverage, first-answer and
   final legality, evaluation rho, side kappa, legal/scored move quality,
   engine-best-move rate and its empirical uniform-random legal-move baseline.
   These supplementary descriptions introduce no new success criteria.
6. Report recorded cost per variant from unique raw response transactions,
   including first responses and retries. Deduplicate repeated ledger entries by
   batch ID and custom ID; the same custom ID billed in a different batch is a
   separate transaction. Never add scored-record cost on top of raw-ledger cost.
   If cost is absent, reconstruct it only from stored token counts and configured
   prices/discount, and label the source. No ledger means cost is unavailable,
   not zero. Count first/retry truncations and disclose pilot expenditure.
7. Add provenance for both scored datasets and raw responses: hashes, record
   counts, code/config state and engine metadata through the existing provenance
   helpers. Do not call Ollama or add API-key contents to provenance. Retain the
   documented hosted-model/default-temperature and unknown-size limitations.

## Completion and missingness

Expected positions per format come from the configured finite sample: four tiers
times the minimum positive sample/max-position cap. Unknown expected sample,
missing model/format or mismatched position sets is disclosed and prevents a
final claim. The registered comparisons remain complete-case comparisons;
report observations that are dropped for each measure. A final verdict must not
turn missing CPL into zero loss or a mixed result.

## Tests before code

Use temporary synthetic JSONL and fake provenance only. Test C1's 5-point edge,
interval overlap/non-overlap, identical position pairing across different model
tags, unchanged C4 absolute-rho branch, pending local/CPL/cohort/constant-rho
cases, valid finite pairs, unknown side labels, missing columns, duplicate jobs,
strict null JSON, pilot exclusion, missing ledger and cost transaction
deduplication. Exercise the CLI with explicit temporary paths and prove it never
imports the API runner or calls a model service.

## Dependencies and reproducibility

Use the existing stdlib, pandas, NumPy and python-chess dependencies. No package
installation, network or GPU is required. Tests use reduced bootstrap resamples
only through monkeypatching the imported helper; production constants and
registered criteria remain unchanged. Analysis outputs are generated only by the
lead after tests/review, without new paid inference.
