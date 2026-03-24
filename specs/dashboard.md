# Spec: Chess LLM Benchmark Dashboard

## Goal
A local web dashboard that displays live benchmark metrics for the chess LLM benchmark run. Must be accessible from the machine itself and via Tailscale remote SSH (i.e. binds to 0.0.0.0).

## Inputs / Outputs
- **Inputs:**
  - `/mnt/shared/chess-llm-bench/jobs/jobs.db` — SQLite job queue (status, per-model progress)
  - `results/evaluations.jsonl` — completed evaluation results
  - `results/metrics/by_model.csv` — precomputed per-model aggregates
  - `results/metrics/by_difficulty.csv` — per-model × difficulty aggregates
  - `results/metrics/hallucination_rate.csv` — hallucination rates
- **Outputs:** Single-page HTML dashboard served over HTTP

## Panels / Metrics

### 1. Run Progress
- Overall: jobs done / total, % bar
- Per-model table: model name, size, progress bar, done/total, currently running indicator
- Auto-refreshes every 5 seconds

### 2. Legal Move % (T2)
- Per-model bar chart
- Grouped by difficulty (easy / medium / hard / extreme) where data exists

### 3. Hallucination Rate (T2)
- Per-model × difficulty heatmap or table
- Hallucination = move attempted but illegal

### 4. Centipawn Loss / CPL (T2)
- Per-model bar chart (mean CPL, only for legal moves)
- Grouped by difficulty where data exists

### 5. Eval Direction Accuracy (T1)
- Per-model bar: % of times model correctly identified which side is winning

### 6. Explanation Score (T3)
- Per-model bar: mean score (0–1 scale)
- Grouped by difficulty where data exists

## Tech Stack
- **Server:** Python + Flask (minimal, no build step)
- **Frontend:** Single HTML file, Tailwind CSS via CDN, Chart.js via CDN
- **Binding:** `0.0.0.0:5050` so accessible locally and over Tailscale

## Steps / Logic
1. Flask server reads from DB + JSONL/CSVs on each `/api/*` request (no caching needed at this scale)
2. `/api/progress` — queries SQLite for per-model job counts
3. `/api/metrics` — reads `by_model.csv`, `by_difficulty.csv`, `hallucination_rate.csv`
4. Frontend polls `/api/progress` every 5s; other data loaded once on page load with a manual refresh button
5. Charts rendered with Chart.js
6. Model order fixed: 3B → 72B (same as monitor script)

## Edge Cases
- Models not yet started: show as "pending" in progress, omit from metric charts
- DB locked (workers writing): use `timeout=5` on SQLite connect
- Missing metric fields (NaN from CSV): treat as null, skip in charts
- JSONL still being written: read with line-by-line error handling

## Dependencies
- Python: `flask`, standard library (`sqlite3`, `csv`, `json`)
- Frontend: Tailwind CSS CDN, Chart.js CDN (no npm)

## Project Structure
```
dashboard/
├── server.py        # Flask app + API routes
└── index.html       # Single-page frontend
```
