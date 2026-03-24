"""Chess LLM Benchmark Dashboard — Flask server."""
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

from flask import Flask, jsonify, send_from_directory

BASE_DIR = Path(__file__).parent.parent
DB_PATH = "/mnt/shared/chess-llm-bench/jobs/jobs.db"
EVALUATIONS_PATH = str(BASE_DIR / "results" / "evaluations.jsonl")

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


def _mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _compute_metrics():
    """Read evaluations.jsonl and compute per-model and per-model×difficulty metrics."""
    # Accumulators: model -> difficulty -> field -> [values]
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    try:
        with open(EVALUATIONS_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                model = r.get("model")
                diff  = r.get("difficulty", "unknown")
                if not model:
                    continue
                a = acc[model][diff]
                v = r.get("t1_direction_correct")
                if v is not None:
                    a["t1_direction_correct"].append(float(v))
                v = r.get("t2_legal")
                if v is not None:
                    a["t2_legal"].append(float(v))
                v = r.get("t2_cpl")
                if v is not None:
                    a["t2_cpl"].append(float(v))
                v = r.get("t3_score")
                if v is not None:
                    a["t3_score"].append(float(v))
    except FileNotFoundError:
        pass

    DIFFICULTIES = ["easy", "medium", "hard", "extreme"]

    by_model = []
    by_difficulty = []
    hallucination = []

    for model, size in MODEL_ORDER:
        if model not in acc:
            continue
        # Flatten all difficulties for per-model aggregates
        all_t1, all_legal, all_cpl, all_t3 = [], [], [], []
        for diff in acc[model]:
            a = acc[model][diff]
            all_t1    += a["t1_direction_correct"]
            all_legal += a["t2_legal"]
            all_cpl   += a["t2_cpl"]
            all_t3    += a["t3_score"]

        by_model.append({
            "model": model,
            "size":  size,
            "t1_direction_correct_mean": _null(_mean(all_t1)),
            "t2_legal_mean":             _null(_mean(all_legal)),
            "t2_cpl_mean":               _null(_mean(all_cpl)),
            "t3_score_mean":             _null(_mean(all_t3)),
        })

        for diff in DIFFICULTIES:
            if diff not in acc[model]:
                continue
            a = acc[model][diff]
            legal_vals = a["t2_legal"]
            legal_mean = _mean(legal_vals)
            halluc_rate = _null(1.0 - legal_mean) if legal_mean is not None else None
            by_difficulty.append({
                "model":      model,
                "difficulty": diff,
                "t2_legal":   _null(legal_mean),
                "t2_cpl":     _null(_mean(a["t2_cpl"])),
                "t3_score":   _null(_mean(a["t3_score"])),
                "t1_direction_correct": _null(_mean(a["t1_direction_correct"])),
            })
            hallucination.append({
                "model":             model,
                "difficulty":        diff,
                "hallucination_rate": halluc_rate,
            })

    return by_model, by_difficulty, hallucination


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
    by_model, by_difficulty, hallucination = _compute_metrics()
    return jsonify({
        "by_model":      by_model,
        "by_difficulty": by_difficulty,
        "hallucination": hallucination,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
