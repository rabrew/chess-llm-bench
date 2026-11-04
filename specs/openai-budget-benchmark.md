# Budget-limited OpenAI chess benchmark

## Goal

Collect matched first-attempt OpenAI responses within a conservative USD20
session ceiling, leaving reserve against the user's EUR40 credits. The USD1
pilot limit is a cumulative sublimit inside USD20, never an additional budget.
No exchange-rate equivalence or account access is assumed.

## Inputs / Outputs

Read historical primary evaluations JSONL as a stream, recover unique selected
positions with FEN/reference/metadata, require consistent reference values, then
freeze a seeded stratified sample. Default50positions per tier, fen_only combined
prompt, Luna and Sol; pilot1position per tier. Proprietary sizes are unknown.
Require explicit external run directory and shared external --budget-ledger.
Plan-only prints JSON and never writes files or calls APIs. Execute freezes
manifest, raw responses, usage, first parse, scores and billing in private files.

## Steps / Logic

1. Validate configuration, pricing provenance, positive limits and known models.
   Prices verified2026-11-04: LunaUSD0.10/0.50, Sol2/10, optional Astra10/50 per
   million input/output tokens, standard tier. See official model/pricing URLs.
2. Freeze full main cohort, prompt content and settings. Pilot uses a deterministic
   subset of that manifest; main requires a completed successful pilot.
3. Acquire one exclusive session ledger lock. Atomic JSON writes fsync files and
   directories. Before each HTTP call persist worst-case reservation. Monetary
   accounting uses integer nano-USD and Decimal configuration, never floats.
4. Bound text input conservatively by UTF8 bytes plus8192token overhead and10%
   charge margin, output2048tokens including reasoning. This is a conservative
   planning bound, not a claim of exact tokenizer correspondence. Preflight the
   entire phase's remaining matched cohort against both session/pilot caps.
5. Send one synchronous HTTPS Responses request, low reasoning/default service
   tier, no tools, no redirects, no retries. Read OPENAI_API_KEY only from env;
   never persist or print credentials or HTTP exception contents.
6. Persist full outcome. Release reservation remainder only for completed output
   with well-formed measured usage within reserved bounds. Ambiguous timeout,
   HTTP failure, refusal, truncated output or missing usage keeps full reserve
   and stops. Observed excess usage is recorded as actual conservative charge
   and execution stops. Resume never reissues any reserved/failed request.
7. Reuse repository prompt builders, first parser and score_all(engine=None).
   No rescue, Ollama, jobs DB, live engine or protected research writes.

## Edge cases

Fail closed on corrupt ledger/manifest, changed caps/rates/settings, path overlap
with repo, inconsistent position references, insufficient strata, ledger loss
with existing output, incomplete pilot, duplicate request identities and concurrent
lock contention. A shared ledger tracks paid IDs across different run directories.
Atomic ledger completion includes result so output write failure cannot buy a
second response. Historical PGN was not stored: default fen_only is reproducible;
pgn+fen is rejected when history is absent. Other formats remain explicitly
labelled, and legal-list move_only is assisted by design.

## Dependencies / Reproducibility

Existing requests,filelock,python-chess; no SDK installation. Seed42. Frozen
source SHA256, prompts/config, parser/evaluator source fingerprints, pricing URLs
and requested/resolved model IDs identify provenance. API calls occur only under
explicit --execute; all implementation tests mock transport and use temp files.
Official docs: https://developers.openai.com/api/docs/guides/reasoning and
https://developers.openai.com/api/docs/pricing and model pages.

## Usage and outcome policy clarification

Default main sample is **200 positions (50 per tier), fen_only**, shared by both
models. Pilot8 requests cost at most approximatelyUSD0.18 under current bounds;
full400 request upper bound approximatelyUSD8.87 (plan prints exact nano-USD).

Metered completed responses with parse failures remain in the main cohort and
are charged measured usage. Pilot parse failure blocks promotion without
reissuing that response. Incomplete/refused output, malformed usage, model/tier
mismatch or transport ambiguity stops and retains full reservation. Optional
usage totals, reasoning and cached-token counts must reconcile.

Plan with no key, writes or network requests:

```bash
venv/bin/python scripts/run_openai_benchmark.py \
  --output-dir ~/chess-openai-20261104/run-200 \
  --budget-ledger ~/chess-openai-20261104/budget.json
```

`--phase main` prints the full plan. Explicit `--execute` enables pilot requests
only when OPENAI_API_KEY is present in the process environment. Main execution
also requires the successful frozen pilot. Reuse the same ledger and run directory
for pilot/main/resume. Never replace the ledger to work around a safe stop.

Private state uses fsynced unique temporary writes. Symlink components and
manifest/records/lock paths are rejected. The ledger must be outside each run
output directory so output writes cannot overwrite accounting state. Config
allows only documented non-secret fields; credentials cannot be placed in config.
Completed ledger entries retain immutable requests and raw responses so a later
output-file write failure recovers without another API charge.
