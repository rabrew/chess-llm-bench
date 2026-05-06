# Chess LLM Benchmark - Runbook

## Quick Reference Commands

```bash
cd /home/rabrew/Desktop/chess-llm-bench
```

---

## 1. Setup

### Install dependencies
```bash
pip install -r requirements.txt
```

### Install Stockfish (if not installed)
```bash
sudo apt install stockfish
```

### Start Ollama (if not running)
```bash
ollama serve
```

---

## 2. Build Dataset

```bash
# Build dataset from all sources
python scripts/build_dataset.py

# With verbose output
python scripts/build_dataset.py -v
```

**Note:** For Lichess puzzles in local mode, download the puzzle database:
```bash
# Download Lichess puzzle database (optional, ~300MB)
wget https://database.lichess.org/lichess_db_puzzle.csv.zst -O data/lichess_puzzles.csv.zst
zstd -d data/lichess_puzzles.csv.zst
```

---

## 3. Pre-compute Stockfish Evaluations

```bash
# Compute evals for all positions (required before running benchmark)
python scripts/precompute_stockfish.py

# With custom depth
python scripts/precompute_stockfish.py --depth 25
```

---

## 4. Pull Ollama Models

```bash
# Pull all configured models
python scripts/pull_models.py

# Pull a specific model
python scripts/pull_models.py --model qwen2.5:7b

# List available models
python scripts/pull_models.py --list
```

---

## 5. Generate Jobs

```bash
# Estimate job count (no changes)
python scripts/generate_jobs.py --estimate

# Generate and insert jobs
python scripts/generate_jobs.py
```

---

## 6. Run Benchmark

```bash
# Dry run (test without saving results)
python scripts/run_workers.py --dry-run --max-jobs 10

# Run with 4 workers (default)
python scripts/run_workers.py --workers 4

# Run with custom worker count
python scripts/run_workers.py --workers 8

# Check queue status
python scripts/run_workers.py --status
```

---

## 7. Generate Plots & Metrics

```bash
# Generate all plots
python scripts/generate_plots.py

# Generate plots and save metric CSVs
python scripts/generate_plots.py --save-metrics
```

---

## 8. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_evaluator.py -v
```

---

## Quick Start (Minimal Test)

```bash
cd /home/rabrew/Desktop/chess-llm-bench
pip install -r requirements.txt
python scripts/build_dataset.py
python scripts/precompute_stockfish.py
python scripts/generate_jobs.py
python scripts/run_workers.py --dry-run --max-jobs 5
```

---

## Full Benchmark Run

```bash
cd /home/rabrew/Desktop/chess-llm-bench

# Setup
pip install -r requirements.txt

# Build and prepare data
python scripts/build_dataset.py
python scripts/precompute_stockfish.py

# Prepare models and jobs
python scripts/pull_models.py
python scripts/generate_jobs.py

# Run benchmark
python scripts/run_workers.py --workers 4

# Generate results
python scripts/generate_plots.py --save-metrics
```

---

## Output Locations

| Output | Location |
|--------|----------|
| Dataset files | `data/*.json` |
| Job database | `jobs/db/jobs.db` |
| Results | `results/evaluations.jsonl` |
| Plots | `results/plots/*.png` |
| Metrics | `results/metrics/*.csv` |
| Run logs | `results/logs/run_*.json` |

---

## Configuration

Edit `config/config.yaml` to change:
- Stockfish path and depth
- Ollama URL and timeout
- Models to test
- Worker count
- CPL threshold for correction loop

Environment variables override config:
```bash
export CHESS_STOCKFISH_DEPTH=25
export CHESS_WORKERS_COUNT=8
export CHESS_OLLAMA_BASE_URL=http://remote-server:11434
```
