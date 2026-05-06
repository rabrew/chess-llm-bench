# Chess LLM Benchmark

**Research question:** How accurately do locally-run open-source LLMs evaluate chess positions and select legal, high-quality moves — and does performance scale with model size?

Benchmarks 19+ local models via Ollama against Stockfish 17 as ground truth across ~5.8M Lichess puzzle positions at four difficulty tiers.

---

## Quick Start

```bash
# Run the full benchmark pipeline
bash bin/run_pipeline.sh
```

For step-by-step instructions see [docs/RUNBOOK.md](docs/RUNBOOK.md).  
For a description of every script see [docs/SCRIPTS.md](docs/SCRIPTS.md).

---

## Project Layout

```
chess-llm-bench/
├── bin/               Shell scripts (pipeline, monitoring, utilities)
├── config/            config.yaml — models, paths, thresholds
├── dashboard/         Live web dashboard (Flask + Chart.js)
├── data/              Puzzle datasets by difficulty tier (gitignored)
├── docs/              Runbook, scripts reference, change log, research notes
├── jobs/              Runtime job queue — SQLite DB + skipped-tier list (gitignored)
├── results/           Evaluations, plots, metrics, logs (gitignored)
│   ├── evaluations.jsonl
│   ├── evaluations_retried.jsonl
│   ├── plots/
│   ├── metrics/
│   ├── logs/
│   └── archive/
├── scripts/           Python scripts (dataset build, workers, analysis)
├── specs/             Feature specifications
├── src/               Library modules
└── tests/             pytest test suite
```

---

## Scoring

Each position is evaluated on three tasks:

| Task | What is measured | Metric |
|------|-----------------|--------|
| **T1 — Eval** | Model estimates centipawn evaluation | Absolute error vs Stockfish |
| **T2 — Move** | Model selects a legal move | Legality rate + centipawn loss |
| **T3 — Explanation** | Model explains who stands better and why | Binary: correct side + justification |

---

## Models

19 Ollama models across five size tiers (3B → 72B). See `config/config.yaml` for the full list.

---

## Hardware

- GPU: RTX 5080 (Lc0 ONNX batch inference)
- Models split across main SSD and shared NTFS partition via symlinks

---

## Research Context

Stripe Young Scientist Award 2027. Full research framing: [docs/RESEARCH.md](docs/RESEARCH.md).  
Detailed benchmark specification: [specs/benchmark-full.md](specs/benchmark-full.md).
