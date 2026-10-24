# Chess LLM Benchmark

**Research question:** Do locally-run open-source LLMs reason about chess positions, or pattern-match? How well do they evaluate positions, choose legal and good moves, and explain them; does that scale with model size; and does fine-tuning them to read the board change it?

Benchmarks 22 local models via Ollama on **4,000 stratified Lichess puzzle positions** (1,000 per difficulty tier × 4 tiers). Each position is probed with 6 prompt formats per model, giving **~526k total evaluations**. Ground truth: for evaluation, a single forward pass of the Lc0 neural network (no search), with its win/draw/loss output converted to centipawns and capped at ±10,000; for move quality / centipawn loss, Lc0 @ 800 nodes. Lc0 was chosen because it runs on the GPU, which on this machine is far faster than CPU Stockfish. (Fields named `stockfish_*` are legacy names and hold Lc0 values.)

For the findings see [docs/FINDINGS.md](docs/FINDINGS.md). For a non-technical summary see [docs/SIMPLE_SUMMARY.md](docs/SIMPLE_SUMMARY.md). The tables in both are generated from `results/metrics/` by `scripts/render_findings.py` (`--check` fails if they are stale). Known measurement issues and their fixes are logged in [docs/issues-log.md](docs/issues-log.md).

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
| **T1 — Eval** | Model estimates centipawn evaluation | **Spearman ρ** vs Lc0 (headline); error relative to always answering 0; chance-corrected direction agreement (κ) at four thresholds |
| **T2 — Move** | Model selects a legal, good move | **Blunder rate** (clamped CPL ≥ 1000, bootstrap CI); CPL buckets; best-move rate vs a random legal move; legality, both final and first-answer |
| **T3 — Explanation** | Model explains who stands better and why | Side mark with majority baseline and κ; whole-word theme mark (tactical themes from the full Lichess list) |

---

## Fine-tuning experiment

`fine-tuning/` trains LoRA adapters on five of the models and re-runs the same benchmark on them:
- **Pieces only:** done. Spec: `fine-tuning/specs/chess-board-finetuning.md`.
- **Pieces + legal moves:** running; the success criteria were committed before any results.
  Spec: `fine-tuning/specs/legal-moves-finetuning.md`.

`fine-tuning/scripts/compare_finetuning_results.py` pairs base and fine-tuned answers on the same
position and prompt. Each training run is recorded in `runs/`.

---

## Models

22 Ollama models from 2B to 70B parameters, spanning 9 architecture families (LLaMA 3, Gemma 3, Gemma 4, Qwen 2.5, DeepSeek-R1, Mistral, Solar, Phi-4, WizardLM, CodeLlama, Yi, Command-R). See [config/config.yaml](config/config.yaml) for the full list.

---

## Hardware

- GPU: RTX 5080 (Lc0 ONNX batch inference for CPL ground truth)
- GPU: Lc0 network (single forward pass, `scripts/precompute_lc0_batch_v2.py`) for T1 ground truth, precomputed once into the dataset
- Models split across main SSD and shared NTFS partition via symlinks

---

## Reproducibility

**Reproducible:** the dataset, the engine ground truth and the scoring.
- **Positions:** sampled with `random_seed: 42` (`config/config.yaml`). Position IDs are a hash of the
  FEN, so they are stable across rebuilds.
- **Engine truth:** Lc0, precomputed and stored. T1 uses the no-search network; CPL uses 800 nodes. The
  node count is stored on every record (`t2_eval_nodes`).
- **Provenance:** every `summary.json` records the git commit, the results-file hash, the Lc0 weights
  hash and the config (`provenance`).
- **Dependencies:** pinned in `requirements.txt`. Dev tools (ruff) are in `requirements-dev.txt`.

**Not reproducible:** the model outputs.
- Five of the six prompt formats ran at Ollama's default temperature.
- Asked the identical prompt twice, a model gave the same move 68.7% of the time (Issue 23).

## Development

```bash
venv/bin/python -m pytest -q tests fine-tuning/tests   # full test suite
venv/bin/ruff check src scripts fine-tuning tests      # lint (config: ruff.toml)
venv/bin/python scripts/render_findings.py --check     # docs match the metrics
```

Every change follows `CLAUDE.md`: spec, then tests, then code, then commit.

---

## Research Context

Stripe Young Scientist Award 2027. Full research framing: [docs/RESEARCH.md](docs/RESEARCH.md).
Detailed benchmark specification: [specs/benchmark-detailed.md](specs/benchmark-detailed.md).
Full audit findings + methodology: [docs/FINDINGS.md](docs/FINDINGS.md), [specs/artefact-fixes.md](specs/artefact-fixes.md).
