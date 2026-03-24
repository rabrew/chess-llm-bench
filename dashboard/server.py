"""Chess LLM Benchmark Dashboard — Flask server."""
import csv
import json
import math
import os
import sqlite3
from pathlib import Path

from flask import Flask, jsonify, send_from_directory

BASE_DIR = Path(__file__).parent.parent
DB_PATH = "/mnt/shared/chess-llm-bench/jobs/jobs.db"
METRICS_DIR = str(BASE_DIR / "results" / "metrics")

app = Flask(__name__, static_folder=str(Path(__file__).parent))

MODEL_ORDER = [
    ("llama3.2:3b",    "3B"),
    ("gemma3:4b",      "4B"),
    ("qwen2.5:7b",     "7B"),
    ("mistral:7b",     "7B"),
    ("deepseek-r1:7b", "7B"),
    ("wizardlm2:7b",   "7B"),
    ("llama3.1:8b",    "8B"),
    ("gemma3:12b",     "12B"),
    ("qwen2.5:14b",    "14B"),
    ("phi4:14b",       "14B"),
    ("deepseek-r1:14b","14B"),
    ("solar:10.7b",    "11B"),
    ("qwen2.5:32b",    "32B"),
    ("codellama:34b",  "34B"),
    ("yi:34b",         "34B"),
    ("command-r:35b",  "35B"),
    ("mixtral:8x7b",   "47B"),
    ("llama3.3:70b",   "70B"),
    ("qwen2.5:72b",    "72B"),
]

MODEL_SIZE_MAP = {m: s for m, s in MODEL_ORDER}
MODEL_RANK = {m: i for i, (m, _) in enumerate(MODEL_ORDER)}


def _null(val):
    """Return None for NaN/inf floats so JSON serialises cleanly."""
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _read_csv(filename):
    path = os.path.join(METRICS_DIR, filename)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@app.route("/")
def index():
    return send_from_directory(str(Path(__file__).parent), "index.html")


@app.route("/api/progress")
def api_progress():
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        cur = conn.cursor()
        cur.execute("SELECT model, status, COUNT(*) FROM jobs GROUP BY model, status")
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    from collections import defaultdict
    counts = defaultdict(lambda: defaultdict(int))
    for model, status, n in rows:
        counts[model][status] = n

    all_models = set(counts.keys()) | {m for m, _ in MODEL_ORDER}
    total_done = total_jobs = 0

    model_list = []
    for model, size in MODEL_ORDER:
        if model not in counts:
            continue
        d = counts[model]
        done      = d.get("done", 0)
        failed    = d.get("failed", 0)
        in_prog   = d.get("in_progress", 0)
        pending   = d.get("pending", 0)
        total     = done + failed + in_prog + pending
        total_done += done
        total_jobs += total
        model_list.append({
            "model":       model,
            "size":        size,
            "done":        done,
            "failed":      failed,
            "in_progress": in_prog,
            "pending":     pending,
            "total":       total,
            "pct":         round(100 * done / total, 1) if total else 0,
            "running":     in_prog > 0,
        })

    overall_pct = round(100 * total_done / total_jobs, 1) if total_jobs else 0
    return jsonify({
        "total":       total_jobs,
        "done":        total_done,
        "overall_pct": overall_pct,
        "models":      model_list,
    })


@app.route("/api/metrics")
def api_metrics():
    by_model_rows = _read_csv("by_model.csv")
    by_diff_rows  = _read_csv("by_difficulty.csv")
    halluc_rows   = _read_csv("hallucination_rate.csv")

    FLOAT_COLS_MODEL = [
        "t1_absolute_error_mean", "t1_direction_correct_mean",
        "t2_legal_mean", "t2_cpl_mean", "t3_score_mean",
    ]
    FLOAT_COLS_DIFF = [
        "t1_absolute_error", "t2_cpl", "t2_legal", "t3_score",
    ]

    def clean_model_row(r):
        out = {"model": r["model"], "size": MODEL_SIZE_MAP.get(r["model"], "?")}
        for col in FLOAT_COLS_MODEL:
            out[col] = _null(r.get(col))
        return out

    def clean_diff_row(r):
        out = {"model": r["model"], "difficulty": r.get("difficulty", "")}
        for col in FLOAT_COLS_DIFF:
            out[col] = _null(r.get(col))
        return out

    def clean_halluc_row(r):
        return {
            "model":            r["model"],
            "difficulty":       r.get("difficulty", ""),
            "hallucination_rate": _null(r.get("hallucination_rate")),
        }

    by_model = sorted(
        [clean_model_row(r) for r in by_model_rows],
        key=lambda r: MODEL_RANK.get(r["model"], 999),
    )
    by_diff  = [clean_diff_row(r)  for r in by_diff_rows]
    halluc   = [clean_halluc_row(r) for r in halluc_rows]

    return jsonify({
        "by_model":    by_model,
        "by_difficulty": by_diff,
        "hallucination": halluc,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
