# Local report refresh and scored OpenAI snapshots

## Goal and authorization

Ryan authorized the remaining local work, excluding further API work and heavy
GPU training. Rebuild older metrics using corrected uncertainty, and score a
finished copy of already-collected GPT answers. Never submit API requests,
resume the API campaign, train, reboot, stop services or load 70B models.
Existing N1–N4/C1–C4 criteria are unchanged.

## Inputs and outputs

- Base responses/config and saved legacy base metrics; read-only inputs.
- A completed/quota-stopped OpenAI campaign directory with its lock, plan,
  ledger, responses, evaluations and summary. Source directory stays untouched.
- New staging report directory with metrics, provenance and invariant validation.
- New snapshot directory containing original source files and a separate enriched
  evaluations JSONL, scoring report, hashes and configured engine provenance.
- Promotion into canonical reports occurs only after independent validation;
  preserve the old reports in a verified archive and git history first.

## Implementation and steps

1. Follow spec → tests → code. Start CPU baseline early; use only temporary
   fixtures/mocked engine calls in tests. No training/model imports in validation.
2. Extend exact API model metadata to collected OpenAI IDs with family OpenAI
   and parameter size None, never a numeric guess. Preserve Claude/local parsing
   and unknown-model rejection. Remove tests' assumption that Claude exhausts
   every possible API model; test actual registrations explicitly instead.
3. Avoid loading massive dataset tiers when result rows already carry complete
   FENs. Preserve fallback lookup for rows needing FEN recovery and the separate
   small puzzle-metadata join. Regression-test both paths and missing FENs.
4. CPU report helper uses existing load_results_df/save_metrics and seeded
   position-cluster intervals, without modifying raw answers or search scores.
   Write to an isolated configured staging directory; refuse unsafe overlapping
   paths and unexplained pre-existing output. Record input/config/code hashes.
5. Validate base coverage (526,662 records,22 models,4,000 positions) and unchanged
   primary point estimates against the saved report, using tight numeric
   tolerances. Blunder CIs/method metadata should change to position-cluster
   uncertainty. Denominator corrections may change explicitly named legacy
   failure measures; never silently require those incorrect values to persist.
6. Snapshot creation obtains a nonblocking shared lock on the existing campaign
   lock file, compatible with the writer's exclusive lock. Refuse active writers,
   partial/malformed/outstanding ledger state or disagreeing exports. Copy only
   a consistent validated checkpoint and record hashes; no credential access.
7. Scoring is a distinct explicit action. Verify no Ollama model is loaded and
   no unknown GPU state before Lc0 construction. Use configured existing Lc0
   binary/weights,800 nodes,250W cap unchanged. Reuse enrich_records logic, not
   a broad dataset loader. Do not modify live campaign exports.
8. Validate identical IDs, FENs, original replies, metadata, usage and count;
   only declared T2 enrichment fields may change. Every legal saved move must
   have finite nonnegative CPL and a stored search reference. Failed/illegal
   answers remain in all-request denominators. Unfinished engine work is pending.
9. Save descriptive scored-snapshot metrics with model/format/position coverage,
   first-answer legality, legal/scored blunder rates, successful legal nonblunder
   replies per request, and CPL provenance. No superiority/overall reasoning
   verdict, no mixing partial GPT outputs with completed Claude comparisons.

## Tests and acceptance

Test exact API metadata and unknown rejection; self-contained FEN skips huge
lookup while missing FEN falls back; source/report immutability; invalid config,
path overlap and nonfinite metrics; partial/live/unknown snapshot rejection;
ledger/export identity agreement; fake-engine enrichment and payload invariants;
zero legal moves/missing scores; raw/inference failures stay in denominators.
Full CPU tests, required Ruff and independent review precede real scoring.
Existing source/settings files used by any active process must not be edited.

## Reproducibility

Use existing metric/bootstrap definitions and seeds (base cluster bootstrap
10,000 resamples,seed42). Record old/new uncertainty and unchanged point values.
Immutable snapshots separate collection from scoring; retain source lineage and
explicit partial coverage. All actual model answers remain exactly as collected.
