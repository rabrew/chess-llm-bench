"""Compare base model vs fine-tuned model benchmark results.

Loads:
  - results/evaluations.jsonl           (base models)
  - results/finetuning_experiment/evaluations.jsonl  (fine-tuned models)

Joins on (position_id, prompt_format) after stripping -board-ft suffix.
Reports per-model delta for T1 direction, T2 legality, T2 clamped CPL, T3 score.
Also computes T0: board description accuracy via verify_board_description().

Output: results/finetuning_experiment/comparison.csv

Usage:
    python fine-tuning/scripts/compare_finetuning_results.py
    python fine-tuning/scripts/compare_finetuning_results.py --base results/evaluations.jsonl
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
FT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FT_SRC))

from board_state import verify_board_description
from src.utils import setup_logging

BASE_RESULTS = "results/evaluations.jsonl"
FT_RESULTS = "results/finetuning_experiment/evaluations.jsonl"
OUTPUT_CSV = "results/finetuning_experiment/comparison.csv"

# Fields used for per-position metrics
T1_FIELD = "t1_correct"
T2_LEGAL_FIELD = "t2_legal"
T2_CPL_FIELD = "t2_cpl"
T3_FIELD = "t3_score"
MODEL_FIELD = "model"
FEN_FIELD = "fen"
POS_ID_FIELD = "position_id"
PROMPT_FIELD = "prompt_format"


def strip_ft_suffix(model_tag: str) -> str:
    """Remove -board-ft suffix to get the base model name."""
    return model_tag.removesuffix("-board-ft")


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # skip malformed lines silently
    return records


def compute_board_accuracy(record: dict) -> float | None:
    """Compute T0 board description accuracy for a record that has a FEN.

    Uses the model's raw response text if available, otherwise returns None.
    """
    fen = record.get(FEN_FIELD)
    response = record.get("response") or record.get("t1_response") or record.get("raw_response")
    if not fen or not response:
        return None
    try:
        result = verify_board_description(fen, str(response))
        return result["accuracy"]
    except Exception:
        return None


def aggregate_model_metrics(records: list[dict]) -> dict[str, dict]:
    """Aggregate per-model metrics from a list of records.

    Returns: {model_tag: {t1_acc, t2_legal_rate, mean_cpl, t3_mean, t0_mean, n}}
    """
    buckets: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for rec in records:
        model = rec.get(MODEL_FIELD, "unknown")
        buckets[model]["t1"].append(1 if rec.get(T1_FIELD) else 0)
        buckets[model]["t2_legal"].append(1 if rec.get(T2_LEGAL_FIELD) else 0)
        cpl = rec.get(T2_CPL_FIELD)
        if cpl is not None:
            buckets[model]["t2_cpl"].append(min(int(cpl), 2000))
        t3 = rec.get(T3_FIELD)
        if t3 is not None:
            buckets[model]["t3"].append(float(t3))
        t0 = compute_board_accuracy(rec)
        if t0 is not None:
            buckets[model]["t0"].append(t0)

    result = {}
    for model, data in buckets.items():
        n = len(data["t1"])
        result[model] = {
            "t1_acc": sum(data["t1"]) / n if n else None,
            "t2_legal_rate": sum(data["t2_legal"]) / n if n else None,
            "mean_cpl": sum(data["t2_cpl"]) / len(data["t2_cpl"]) if data["t2_cpl"] else None,
            "t3_mean": sum(data["t3"]) / len(data["t3"]) if data["t3"] else None,
            "t0_mean": sum(data["t0"]) / len(data["t0"]) if data["t0"] else None,
            "n": n,
        }
    return result


def compute_deltas(
    base_metrics: dict[str, dict],
    ft_metrics: dict[str, dict],
) -> list[dict]:
    """Compute per-model deltas between fine-tuned and base results."""
    rows = []
    for ft_model, ft_data in sorted(ft_metrics.items()):
        base_model = strip_ft_suffix(ft_model)
        base_data = base_metrics.get(base_model, {})

        def delta(key: str) -> float | None:
            ft_val = ft_data.get(key)
            base_val = base_data.get(key)
            if ft_val is not None and base_val is not None:
                return round(ft_val - base_val, 4)
            return None

        rows.append({
            "base_model": base_model,
            "ft_model": ft_model,
            "base_n": base_data.get("n", 0),
            "ft_n": ft_data.get("n", 0),
            # T0 — board accuracy (new metric, ft only has meaningful values)
            "ft_t0_mean": round(ft_data.get("t0_mean") or 0, 4),
            "base_t0_mean": round(base_data.get("t0_mean") or 0, 4),
            "delta_t0": delta("t0_mean"),
            # T1 — direction accuracy
            "base_t1_acc": base_data.get("t1_acc"),
            "ft_t1_acc": ft_data.get("t1_acc"),
            "delta_t1_acc": delta("t1_acc"),
            # T2 — legality
            "base_t2_legal": base_data.get("t2_legal_rate"),
            "ft_t2_legal": ft_data.get("t2_legal_rate"),
            "delta_t2_legal": delta("t2_legal_rate"),
            # T2 — CPL (lower is better, so negative delta = improvement)
            "base_mean_cpl": base_data.get("mean_cpl"),
            "ft_mean_cpl": ft_data.get("mean_cpl"),
            "delta_cpl": delta("mean_cpl"),
            # T3 — explanation score
            "base_t3_mean": base_data.get("t3_mean"),
            "ft_t3_mean": ft_data.get("t3_mean"),
            "delta_t3": delta("t3_mean"),
        })
    return rows


def write_csv(rows: list[dict], output_path: str) -> None:
    if not rows:
        return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict], logger: Any) -> None:
    logger.info("\n=== Fine-tuning experiment summary ===")
    logger.info(f"{'Model':<25} {'T0 (board)':<12} {'ΔT1':<10} {'ΔT2 legal':<12} {'ΔCPL':<10} {'ΔT3':<8}")
    logger.info("-" * 77)
    for row in rows:
        model = row["base_model"][:24]
        t0 = f"{row['ft_t0_mean']:.3f}" if row["ft_t0_mean"] else "—"
        dt1 = f"{row['delta_t1_acc']:+.3f}" if row["delta_t1_acc"] is not None else "—"
        dt2 = f"{row['delta_t2_legal']:+.3f}" if row["delta_t2_legal"] is not None else "—"
        dcpl = f"{row['delta_cpl']:+.0f}" if row["delta_cpl"] is not None else "—"
        dt3 = f"{row['delta_t3']:+.3f}" if row["delta_t3"] is not None else "—"
        logger.info(f"{model:<25} {t0:<12} {dt1:<10} {dt2:<12} {dcpl:<10} {dt3:<8}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare base vs fine-tuned benchmark results."
    )
    parser.add_argument(
        "--base", default=BASE_RESULTS,
        help=f"Base results JSONL (default: {BASE_RESULTS})",
    )
    parser.add_argument(
        "--ft", default=FT_RESULTS,
        help=f"Fine-tuned results JSONL (default: {FT_RESULTS})",
    )
    parser.add_argument(
        "--output", default=OUTPUT_CSV,
        help=f"Output CSV path (default: {OUTPUT_CSV})",
    )
    args = parser.parse_args()

    logger = setup_logging("compare_finetuning")

    if not Path(args.base).exists():
        logger.error(f"Base results not found: {args.base}")
        sys.exit(1)
    if not Path(args.ft).exists():
        logger.error(f"FT results not found: {args.ft}")
        sys.exit(1)

    logger.info(f"Loading base results: {args.base}")
    base_records = load_jsonl(args.base)
    logger.info(f"  {len(base_records)} records loaded.")

    logger.info(f"Loading FT results: {args.ft}")
    ft_records = load_jsonl(args.ft)
    logger.info(f"  {len(ft_records)} records loaded.")

    base_metrics = aggregate_model_metrics(base_records)
    ft_metrics = aggregate_model_metrics(ft_records)

    rows = compute_deltas(base_metrics, ft_metrics)

    if not rows:
        logger.warning("No matching model pairs found. Check that FT tags end with -board-ft.")
        sys.exit(1)

    write_csv(rows, args.output)
    logger.info(f"\nComparison CSV written: {args.output}")

    print_summary(rows, logger)


if __name__ == "__main__":
    main()
