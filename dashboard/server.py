"""Chess LLM Benchmark Dashboard — Flask server."""
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

from flask import Flask, jsonify, send_from_directory

BASE_DIR = Path(__file__).parent.parent
DB_PATH = str(BASE_DIR / "jobs" / "db" / "jobs.db")
EVALUATIONS_PATH = str(BASE_DIR / "results" / "evaluations.jsonl")

TIERS = ["easy", "medium", "hard", "extreme"]
JOBS_PER_TIER = 5900
DIRECTION_THRESHOLD = 150  # centipawns — only call it White/Black if advantage > 1.5 pawns


def _direction(eval_cp, threshold=DIRECTION_THRESHOLD):
    if eval_cp is None:
        return None
    if eval_cp > threshold:
        return "White"
    if eval_cp < -threshold:
        return "Black"
    return "Equal"

app = Flask(__name__, static_folder=str(Path(__file__).parent), static_url_path="")

MODEL_ORDER = [
    ("llama3.2:3b",      "3B"),
    ("gemma3:4b",        "4B"),
    ("qwen2.5:7b",       "7B"),
    ("mistral:7b",       "7B"),
    ("wizardlm2:7b",     "7B"),
    ("llama3.1:8b",      "8B"),
    ("gemma3:12b",       "12B"),
    ("qwen2.5:14b",      "14B"),
    ("phi4:14b",         "14B"),
    ("solar:10.7b",      "11B"),
    ("gemma4:e4b",       "4B"),
    ("qwen2.5:32b",      "32B"),
    ("codellama:34b",    "34B"),
    ("yi:34b",           "34B"),
    ("command-r:35b",    "35B"),
    ("gemma4:26b",       "26B"),
    ("gemma4:31b",       "31B"),
    ("llama3.3:70b",     "70B"),
    ("gemma4:e2b",       "2B"),
    ("mixtral:8x7b",     "47B"),
    ("qwen2.5:72b",      "72B"),
    ("deepseek-r1:7b",   "7B"),
    ("deepseek-r1:14b",  "14B"),
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
                fmt = r.get("prompt_format", "")
                # Only score t1 direction on formats that actually request an eval.
                # Recompute from raw evals using current threshold (stored t1_direction_correct
                # used the old 50cp threshold).
                if fmt not in ("move_only", "explanation_only"):
                    sf = r.get("t1_stockfish_eval")
                    me = r.get("t1_model_eval")
                    if sf is not None and me is not None:
                        v = float(_direction(me) == _direction(sf))
                        a["t1_direction_correct"].append(v)
                    ae = r.get("t1_absolute_error")
                    if ae is not None:
                        a["t1_absolute_error"].append(float(ae))
                # Only score t2_legal on formats that actually request a move
                if fmt not in ("eval_only", "explanation_only"):
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
        all_t1, all_legal, all_cpl, all_t3, all_mae = [], [], [], [], []
        for diff in acc[model]:
            a = acc[model][diff]
            all_t1    += a["t1_direction_correct"]
            all_legal += a["t2_legal"]
            all_cpl   += a["t2_cpl"]
            all_t3    += a["t3_score"]
            all_mae   += a["t1_absolute_error"]

        by_model.append({
            "model": model,
            "size":  size,
            "t1_direction_correct_mean": _null(_mean(all_t1)),
            "t2_legal_mean":             _null(_mean(all_legal)),
            "t2_cpl_mean":               _null(_mean(all_cpl)),
            "t3_score_mean":             _null(_mean(all_t3)),
            "t1_mae_mean":               _null(_mean(all_mae)),
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
                "t1_mae":     _null(_mean(a["t1_absolute_error"])),
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


def _compute_pipeline_progress():
    """Count completed tiers from evaluations.jsonl for full pipeline view."""
    counts = defaultdict(lambda: defaultdict(int))
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
                diff = r.get("difficulty")
                if model and diff:
                    counts[model][diff] += 1
    except FileNotFoundError:
        pass

    total_tiers = len(MODEL_ORDER) * len(TIERS)
    done_tiers = 0
    model_tiers = {}
    for model, _ in MODEL_ORDER:
        tier_status = {}
        for tier in TIERS:
            n = counts[model][tier]
            complete = n >= JOBS_PER_TIER
            if complete:
                done_tiers += 1
            tier_status[tier] = {"count": n, "complete": complete}
        model_tiers[model] = tier_status

    return {
        "total_tiers": total_tiers,
        "done_tiers": done_tiers,
        "pipeline_pct": round(100 * done_tiers / total_tiers, 1) if total_tiers else 0,
        "model_tiers": model_tiers,
    }


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
    pipeline = _compute_pipeline_progress()
    return jsonify({
        "total":       total_jobs,
        "done":        total_done,
        "overall_pct": overall_pct,
        "models":      model_list,
        "pipeline":    pipeline,
    })


@app.route("/api/metrics")
def api_metrics():
    by_model, by_difficulty, hallucination = _compute_metrics()
    return jsonify({
        "by_model":      by_model,
        "by_difficulty": by_difficulty,
        "hallucination": hallucination,
    })


@app.route("/api/current")
def api_current():
    """Per-tier metrics + ETA for the currently active model."""
    from datetime import datetime, timezone, timedelta

    TOTAL_PER_MODEL = JOBS_PER_TIER * len(TIERS)

    counts = defaultdict(lambda: defaultdict(int))
    tier_acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    last_ts = {}

    try:
        with open(EVALUATIONS_PATH, "rb") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    m = r.get("model")
                    tier = r.get("difficulty")
                    if not m or not tier:
                        continue
                    counts[m][tier] += 1
                    ts = r.get("timestamp")
                    if ts and (m not in last_ts or ts > last_ts[m]):
                        last_ts[m] = ts
                    fmt = r.get("prompt_format", "")
                    if fmt not in ("eval_only", "explanation_only"):
                        v = r.get("t2_legal")
                        if v is not None:
                            tier_acc[m][tier]["legal"].append(float(v))
                    if fmt not in ("move_only", "explanation_only"):
                        sf = r.get("t1_stockfish_eval")
                        me = r.get("t1_model_eval")
                        if sf is not None and me is not None:
                            tier_acc[m][tier]["direction"].append(
                                float(_direction(me) == _direction(sf))
                            )
                    v = r.get("t3_score")
                    if v is not None:
                        tier_acc[m][tier]["t3"].append(float(v))
                except Exception:
                    continue
    except FileNotFoundError:
        pass

    # Find active model via DB, fallback to most-recently-updated incomplete
    active_model = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        cur = conn.cursor()
        cur.execute("SELECT model FROM jobs WHERE status='in_progress' LIMIT 1")
        row = cur.fetchone()
        conn.close()
        if row:
            active_model = row[0]
    except Exception:
        pass

    if not active_model:
        candidates = [
            (ts, m) for m, ts in last_ts.items()
            if sum(counts[m].values()) < TOTAL_PER_MODEL
        ]
        if candidates:
            active_model = max(candidates)[1]

    if not active_model:
        return jsonify({"model": None})

    active_size = next((s for m, s in MODEL_ORDER if m == active_model), "?")
    total_done = sum(counts[active_model].values())
    remaining = TOTAL_PER_MODEL - total_done

    tier_data = {}
    for tier in TIERS:
        n = counts[active_model].get(tier, 0)
        a = tier_acc[active_model][tier]
        legal = _mean(a.get("legal", []))
        direction = _mean(a.get("direction", []))
        t3 = _mean(a.get("t3", []))
        tier_data[tier] = {
            "done": n,
            "complete": n >= JOBS_PER_TIER,
            "legal": _null(legal * 100) if legal is not None else None,
            "direction": _null(direction * 100) if direction is not None else None,
            "t3": _null(t3 * 100) if t3 is not None else None,
        }

    # Throughput from last ~600 entries for this model
    recent_ts_list = []
    try:
        with open(EVALUATIONS_PATH, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 400_000))
            tail = f.read().split(b"\n")
        for raw in tail[-700:]:
            raw = raw.strip()
            if not raw:
                continue
            try:
                r = json.loads(raw)
                if r.get("model") == active_model:
                    ts = r.get("timestamp")
                    if ts:
                        recent_ts_list.append(
                            datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        )
            except Exception:
                continue
    except Exception:
        pass

    jobs_per_min = None
    eta_secs = None
    if len(recent_ts_list) >= 10:
        recent_ts_list.sort()
        span = (recent_ts_list[-1] - recent_ts_list[0]).total_seconds()
        if span > 0:
            rate = len(recent_ts_list) / span
            jobs_per_min = _null(rate * 60)
            if remaining > 0:
                eta_secs = int(remaining / rate)

    return jsonify({
        "model": active_model,
        "size": active_size,
        "total_done": total_done,
        "total_target": TOTAL_PER_MODEL,
        "pct": round(100 * total_done / TOTAL_PER_MODEL, 1) if TOTAL_PER_MODEL else 0,
        "jobs_per_min": jobs_per_min,
        "eta_secs": eta_secs,
        "tier_data": tier_data,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
