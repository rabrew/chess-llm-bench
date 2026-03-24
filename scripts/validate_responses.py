"""
Validate stored LLM responses against the evaluator requirements.

Reads evaluations.jsonl (and optional backup), processes one model at a time,
and re-scores each record to verify T1/T2/T3 correctness.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

# Allow running from any directory
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluator import score_t1, score_t2, score_t3

RESULTS_FILE = ROOT / "results" / "evaluations.jsonl"
BACKUP_FILE  = ROOT / "results" / "evaluations.jsonl.bak_20260319"

EVAL_RANGE = (-2000, 2000)


def load_records(*paths):
    """Load and merge records from one or more jsonl files, dedup by job_id."""
    seen = {}
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    seen[r["job_id"]] = r
                except json.JSONDecodeError:
                    pass
    return list(seen.values())


def validate_record(r):
    """Re-score one record and return a dict of issues found."""
    issues = []

    # --- T1 re-score ---
    t1 = score_t1(
        model_eval=r.get("t1_model_eval"),
        stockfish_eval=r["t1_stockfish_eval"],
        eval_range=EVAL_RANGE,
    )
    if t1["t1_absolute_error"] != r.get("t1_absolute_error"):
        issues.append(
            f"T1 absolute_error mismatch: stored={r.get('t1_absolute_error')} "
            f"recalc={t1['t1_absolute_error']}"
        )
    if t1["t1_direction_correct"] != r.get("t1_direction_correct"):
        issues.append(
            f"T1 direction_correct mismatch: stored={r.get('t1_direction_correct')} "
            f"recalc={t1['t1_direction_correct']}"
        )

    # --- T2 re-score (legality only — no engine for CPL) ---
    t2 = score_t2(
        model_move=r.get("t2_move"),
        fen=r["fen"],
        stockfish_best_move=r.get("t2_best_move") or "",
        stockfish_eval=r["t1_stockfish_eval"],
        engine=None,
    )
    if t2["t2_legal"] != r.get("t2_legal"):
        issues.append(
            f"T2 legal mismatch: stored={r.get('t2_legal')} "
            f"recalc={t2['t2_legal']}  move={r.get('t2_move')}  fen={r['fen']}"
        )

    # --- T3 re-score ---
    t3 = score_t3(
        explanation=r.get("t3_explanation"),
        side_claimed=r.get("t3_side_claimed"),
        stockfish_eval=r["t1_stockfish_eval"],
        theme=r.get("theme", ""),
    )
    if t3["t3_p1_side_correct"] != r.get("t3_p1_side_correct"):
        issues.append(
            f"T3 p1_side_correct mismatch: stored={r.get('t3_p1_side_correct')} "
            f"recalc={t3['t3_p1_side_correct']}"
        )
    if t3["t3_p2_theme_correct"] != r.get("t3_p2_theme_correct"):
        issues.append(
            f"T3 p2_theme_correct mismatch: stored={r.get('t3_p2_theme_correct')} "
            f"recalc={t3['t3_p2_theme_correct']}"
        )

    return issues, t2["t2_legal"], t1["t1_direction_correct"], t3["t3_score"]


def validate_model(model_name, records):
    """Validate all records for a single model and print a report."""
    total = len(records)
    if total == 0:
        print(f"  [SKIP] No records found")
        return

    mismatches = 0
    legal_moves = 0
    direction_correct = 0
    t3_scores = []
    all_issues = []

    for r in records:
        issues, legal, direction, t3_score = validate_record(r)
        if legal:
            legal_moves += 1
        if direction:
            direction_correct += 1
        if t3_score is not None:
            t3_scores.append(t3_score)
        if issues:
            mismatches += 1
            for issue in issues:
                all_issues.append(f"    job={r['job_id']}: {issue}")

    t2_pct  = 100 * legal_moves / total
    t1_pct  = 100 * direction_correct / total
    t3_avg  = sum(t3_scores) / len(t3_scores) if t3_scores else 0.0

    status = "OK" if mismatches == 0 else f"MISMATCH ({mismatches} records)"
    tiers  = sorted(set(r.get("difficulty", "?") for r in records))

    print(f"  Records : {total}  |  Tiers: {', '.join(tiers)}")
    print(f"  T1 dir% : {t1_pct:5.1f}%   (model picks correct winning side)")
    print(f"  T2 legal: {t2_pct:5.1f}%   (move is legal on the board)")
    print(f"  T3 avg  : {t3_avg:5.2f}/2  (side + theme correctness)")
    print(f"  Score   : {status}")
    if all_issues:
        print(f"  Issues:")
        for line in all_issues[:10]:   # cap output
            print(line)
        if len(all_issues) > 10:
            print(f"    ... and {len(all_issues) - 10} more")


def main():
    print("Loading records from evaluations.jsonl + backup...\n")
    records = load_records(RESULTS_FILE, BACKUP_FILE)
    print(f"Total records loaded: {len(records)}\n")

    # Group by model
    by_model = defaultdict(list)
    for r in records:
        by_model[r["model"]].append(r)

    # Ordered by model size (config order)
    config_order = [
        "llama3.2:3b", "gemma3:4b", "qwen2.5:7b", "mistral:7b",
        "deepseek-r1:7b", "wizardlm2:7b",
        "llama3.1:8b", "gemma3:12b", "qwen2.5:14b", "phi4:14b",
        "deepseek-r1:14b", "solar:10.7b",
        "qwen2.5:32b", "codellama:34b", "yi:34b", "command-r:35b",
        "mixtral:8x7b", "llama3.3:70b", "qwen2.5:72b",
    ]

    found_any   = False
    missing     = []

    for model in config_order:
        recs = by_model.get(model, [])
        if not recs:
            missing.append(model)
            continue

        found_any = True
        print(f"{'='*60}")
        print(f"Model: {model}")
        validate_model(model, recs)
        print()

    print(f"{'='*60}")
    print(f"Models with no data ({len(missing)}):")
    for m in missing:
        print(f"  - {m}")


if __name__ == "__main__":
    main()
