# Chess LLM Benchmark

**Research question:** How accurately do locally-run open-source LLMs evaluate chess positions and select legal, high-quality moves — and does performance scale with model size?

Benchmarks 22 local models via Ollama on **4,000 stratified Lichess puzzle positions** (1,000 per difficulty tier × 4 tiers). Each position is probed with 6 prompt formats per model, giving **~526k total evaluations**. Ground truth: Stockfish 17 @ depth 22 for evaluation truth, and Lc0 (GPU-accelerated) for move-quality / centipawn-loss truth.

For the headline findings see [docs/FINDINGS.md](docs/FINDINGS.md). For a non-technical summary see [docs/SIMPLE_SUMMARY.md](docs/SIMPLE_SUMMARY.md).

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
| **T1 — Eval** | Model estimates centipawn evaluation | Direction accuracy (multi-threshold), absolute / relative error vs Stockfish |
| **T2 — Move** | Model selects a legal move | Legality rate + clamped centipawn loss + win-probability loss |
| **T3 — Explanation** | Model explains who stands better and why | Binary: correct side + theme keyword match |

---

## Models

22 Ollama models from 2B to 70B parameters, spanning 9 architecture families (LLaMA 3, Gemma 3, Gemma 4, Qwen 2.5, DeepSeek-R1, Mistral, Solar, Phi-4, WizardLM, CodeLlama, Yi, Command-R). See [config/config.yaml](config/config.yaml) for the full list.

---

## Hardware

- GPU: RTX 5080 (Lc0 ONNX batch inference for CPL ground truth)
- CPU: Stockfish-17 @ depth 22 for T1 ground truth (precomputed once into the dataset)
- Models split across main SSD and shared NTFS partition via symlinks

---

## Reproducibility

- Random seed: `random_seed: 42` in `config/config.yaml` is propagated through:
  - `src/dataset_builder.py` for puzzle sampling (1,000 per tier)
  - `src/feedback_loop.py:select_follow_up_position` for the (currently inactive) correction-loop
- Position IDs are derived from a SHA-256 hash of each FEN, so dataset rebuilds with different `max_positions_per_tier` values produce stable IDs.
- Ground-truth Stockfish evaluations are precomputed once and stored alongside the position files; LLM scoring is therefore deterministic given the same model + prompt + Stockfish-eval triple.
- Engine versions and depths are pinned in `config/config.yaml` (Stockfish depth, Lc0 nodes).
- Python dependencies are pinned in `requirements.txt`.

---

## Research Context

Stripe Young Scientist Award 2027. Full research framing: [docs/RESEARCH.md](docs/RESEARCH.md).
Detailed benchmark specification: [specs/benchmark-detailed.md](specs/benchmark-detailed.md).
Full audit findings + methodology: [docs/FINDINGS.md](docs/FINDINGS.md), [specs/artefact-fixes.md](specs/artefact-fixes.md).
