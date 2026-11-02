# Pipeline audit remediation
## Goal
Resolve audit findings 04, 06, 08, 09, 10 without touching historical research data.
## Inputs / Outputs
Inputs: mocked UCI text streams, dataset configuration/local CSV, preprocessing records.
Outputs: bounded protocol reads or explicit EOF/timeout errors; reproducible tier cohorts;
explicit engine-specific reference metadata; valid a1 king acceptance.
## Steps / Logic
- Use a bounded reader queue to avoid blocking buffered readline under a protocol deadline.
  Timeout is configurable (positive finite seconds), default 60. EOF differs from blank lines.
  Retain at most 100000 protocol lines. Startup failures close their own subprocess.
- Consume parallel validation in sampled input order before deterministic truncation.
- Default dataset.design is lichess_puzzles; request full quota each tier and reject
  incomplete finite tiers before any file write. Mixed PGN/generated construction requires
  explicit design=mixed. require_full_tiers=false permits documented exploratory partial builds.
- New preprocessing records retain compatibility fields but also engine-specific values and
  reference provenance (engine, budget, units, perspective). Stockfish completion requires
  finite numeric score, legal best move and matching engine/depth metadata. Lc0 completion
  checks its own metadata and value. Existing unprovenanced values are not silently certified.
- Compare king square with None; square zero is a valid occupied a1.
## Edge cases
EOF, blank lines, a live silent engine, excessive output, invalid timeout; invalid puzzle
rows, parallel scheduling differences, insufficient unique puzzles, unknown design;
null/NaN scores, invalid move, wrong engine/depth, missing provenance, both colors on a1.
No validation starts a chess engine, model call or dataset-generation CLI.
## Dependencies
Python queue/threading/time/math, python-chess, pandas; pytest with temporary fixtures
and mocked subprocess/executor only.
## Reproducibility
Use configured random seed (default42), stable sampled order and existing FEN-derived IDs.
Historical outputs are preserved; rebuilt cohorts need a separate approved research run.

## Reader lifecycle amendment
A protocol flood may fill the bounded output queue while the consumer times out or
hits the line limit. Queue insertion must poll a per-reader stop event; close sets
that event and joins the tracked daemon with a finite timeout after subprocess
shutdown. Restart must not leave old readers blocked on queue insertion.
