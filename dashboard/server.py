"""Chess LLM Benchmark Dashboard — Flask server."""
import json
import math
import sqlite3
import sys
from collections import defaultdict
from functools import lru_cache, wraps
from pathlib import Path
from threading import RLock

from flask import Flask, jsonify, send_from_directory

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.metrics import (  # noqa: E402
    COMBINED_PROMPT_FORMATS, MATE_SCORE_THRESHOLD_CP, MOVE_PROMPT_FORMATS,
    _white_to_move_from_fen, compute_clamped_cpl, rescore_t3_theme,
)

DB_PATH = str(BASE_DIR / "jobs" / "db" / "jobs.db")
EVALUATIONS_PATH = str(BASE_DIR / "results" / "evaluations.jsonl")

TIERS = ["easy", "medium", "hard", "extreme"]
JOBS_PER_TIER = 5900
# Direction threshold matches `metrics.DIRECTION_THRESHOLDS_CP` and the
# "1 pawn = decisive" convention used as primary in docs/FINDINGS.md.
# Earlier versions of this dashboard used 150 cp, which diverged from the
# published findings. Keep this in sync with the docs.
DIRECTION_THRESHOLD = 100
T3_MAX_SCORE = 2
METRIC_DEFINITIONS = {
    "t2_cpl_mean": {"metric": "t2_cpl_clamped", "reference": "stored t2_eval_before", "range": [0, 2000]},
    "t3_score_mean": {"metric": "t3_score_v2", "matcher": "whole word", "range": [0, T3_MAX_SCORE]},
    "t1_mae_mean": {"metric": "t1_abs_error_excl_mate", "reference": "stored engine evaluation"},
    "t1_direction_correct_mean": {"threshold_cp": DIRECTION_THRESHOLD},
    "t2_legal_mean": {"denominator": "move-requesting prompts, including missing requested answers"},
}
_METRICS_CACHE = {}
_METRICS_LOCK = RLock()


def _synchronized(function):
    @wraps(function)
    def locked(*args, **kwargs):
        with _METRICS_LOCK:
            return function(*args, **kwargs)
    return locked


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


@lru_cache(maxsize=10000)
def _turn(fen):
    return _white_to_move_from_fen(fen) if isinstance(fen, str) else None


def _corrected_cpl(record, quality):
    cpl = _null(record.get("t2_cpl"))
    if cpl is None:
        return None
    before = _null(record.get("t2_eval_before"))
    if before is None:
        quality["excluded_cpl_missing_reference"] += 1
        return None
    turn = _turn(record.get("fen"))
    if turn is None or cpl < 0:
        quality["excluded_cpl_invalid_input"] += 1
        return None
    after = before - cpl if turn else before + cpl
    return compute_clamped_cpl(before, after, turn)


def _corrected_t3(record, quality):
    score = _null(record.get("t3_score_v2"))
    if score is not None and 0 <= score <= T3_MAX_SCORE:
        return score
    p1 = _null(record.get("t3_p1_side_correct"))
    p2 = rescore_t3_theme(record.get("theme"), record.get("t3_explanation"))
    if p1 in (0, 1) and p2 is not None:
        return p1 + p2
    if record.get("t3_score") is not None:
        quality["excluded_t3_missing_corrected_score"] += 1
    return None


def _metric_values(record, quality):
    """Shared pure scoring helpers; never load datasets or start an engine."""
    fmt = record.get("prompt_format")
    legacy = not fmt
    if legacy:
        quality["missing_prompt_format"] += 1
    values = {}
    if legacy or fmt in (*COMBINED_PROMPT_FORMATS, "eval_only"):
        truth, predicted = _null(record.get("t1_stockfish_eval")), _null(record.get("t1_model_eval"))
        if truth is not None and predicted is not None:
            values["t1_direction_correct"] = float(_direction(predicted) == _direction(truth))
        if truth is not None and abs(truth) < MATE_SCORE_THRESHOLD_CP:
            values["t1_absolute_error"] = _null(record.get("t1_absolute_error"))
    if legacy or fmt in MOVE_PROMPT_FORMATS:
        legal = record.get("t2_legal") is True
        values["t2_legal"] = float(legal)
        values["t2_cpl"] = _corrected_cpl(record, quality) if legal else None
    if legacy or fmt in (*COMBINED_PROMPT_FORMATS, "explanation_only"):
        values["t3_score"] = _corrected_t3(record, quality)
    return values


@_synchronized
def _metric_snapshot():
    """Reuse scored data across API routes until the JSONL file changes."""
    try:
        stat = Path(EVALUATIONS_PATH).stat()
        signature = (EVALUATIONS_PATH, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    except FileNotFoundError:
        signature = (EVALUATIONS_PATH, None)
    if _METRICS_CACHE.get("signature") == signature:
        return _METRICS_CACHE["snapshot"]
    snapshot = _read_metric_snapshot()
    _METRICS_CACHE.update(signature=signature, snapshot=snapshot)
    return snapshot


def _read_metric_snapshot():
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    counts = defaultdict(lambda: defaultdict(int))
    last_ts, quality = {}, defaultdict(int)
    try:
        with open(EVALUATIONS_PATH, "rb") as source:
            while line := source.readline():
                if not line.endswith(b"\n"):
                    break  # Leave an in-flight append for the next refresh.
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(record, dict) or not record.get("model"):
                    continue
                model, tier = record["model"], record.get("difficulty", "unknown")
                counts[model][tier] += 1
                timestamp = record.get("timestamp")
                if timestamp and timestamp > last_ts.get(model, ""):
                    last_ts[model] = timestamp
                for field, value in _metric_values(record, quality).items():
                    if value is not None:
                        acc[model][tier][field].append(value)
    except FileNotFoundError:
        pass
    return {"acc": acc, "counts": counts, "last_ts": last_ts, "quality": dict(quality)}


def _metric_metadata():
    quality = _metric_snapshot()["quality"]
    explanations = {
        "excluded_cpl_missing_reference": "CPL without stored T2 reference excluded; no T1 fallback is used.",
        "excluded_cpl_invalid_input": "CPL with invalid FEN or negative loss excluded.",
        "excluded_t3_missing_corrected_score": "Legacy T3 scores without rescorable evidence excluded.",
        "missing_prompt_format": "Legacy records without prompt format retain uncertain task applicability.",
    }
    return {"metric_definitions": METRIC_DEFINITIONS, "metric_quality": quality,
            "warnings": [text for key, text in explanations.items() if quality.get(key, 0)]}


@_synchronized
def _compute_metrics():
    """Read evaluations.jsonl and compute per-model and per-model×difficulty metrics."""
    # Accumulators: model -> difficulty -> field -> [values]
    acc = _metric_snapshot()["acc"]

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


@_synchronized
def _compute_pipeline_progress():
    """Count completed tiers from evaluations.jsonl for full pipeline view."""
    counts = _metric_snapshot()["counts"]

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
@_synchronized
def api_metrics():
    by_model, by_difficulty, hallucination = _compute_metrics()
    return jsonify({
        "by_model":      by_model,
        "by_difficulty": by_difficulty,
        "hallucination": hallucination,
        **_metric_metadata(),
    })


@app.route("/api/current")
@_synchronized
def api_current():
    """Per-tier metrics + ETA for the currently active model."""
    from datetime import datetime, timezone, timedelta

    TOTAL_PER_MODEL = JOBS_PER_TIER * len(TIERS)

    snapshot = _metric_snapshot()
    counts, tier_acc, last_ts = snapshot["counts"], snapshot["acc"], snapshot["last_ts"]

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
        legal = _mean(a.get("t2_legal", []))
        direction = _mean(a.get("t1_direction_correct", []))
        t3 = _mean(a.get("t3_score", []))
        tier_data[tier] = {
            "done": n,
            "complete": n >= JOBS_PER_TIER,
            "legal": _null(legal * 100) if legal is not None else None,
            "direction": _null(direction * 100) if direction is not None else None,
            "t3": _null(t3 / T3_MAX_SCORE * 100) if t3 is not None else None,
        }

    # Throughput from jobs completed in the last 10 minutes
    recent_ts_list = []
    now_utc = datetime.now(timezone.utc)
    window = timedelta(minutes=10)
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
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        if now_utc - dt <= window:
                            recent_ts_list.append(dt)
            except Exception:
                continue
    except Exception:
        pass

    jobs_per_min = None
    eta_secs = None
    if len(recent_ts_list) >= 2:
        recent_ts_list.sort()
        span = (recent_ts_list[-1] - recent_ts_list[0]).total_seconds()
        if span > 0:
            rate = (len(recent_ts_list) - 1) / span
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
        **_metric_metadata(),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
