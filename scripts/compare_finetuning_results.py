"""Compare base model vs fine-tuned model benchmark results.

Loads:
  - results/evaluations.jsonl           (base models)
  - results/finetuning_experiment/evaluations.jsonl  (fine-tuned models)

Joins on (position_id, prompt_format) after stripping -board-ft suffix.
Reports per-model delta for T1 direction, T2 legality, T2 clamped CPL, T3 score.
Also computes T0: board description accuracy via verify_board_description().

Output: results/finetuning_experiment/comparison.csv

Usage:
    python scripts/compare_finetuning_results.py
    python scripts/compare_finetuning_results.py --base results/evaluations.jsonl
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.board_state import verify_board_description
from src.utils import setup_logging
from src.metrics import derive_endpoint_metrics, rescore_t3_theme

BASE_RESULTS = "results/evaluations.jsonl"
FT_RESULTS = "results/finetuning_experiment/evaluations.jsonl"
OUTPUT_CSV = "results/finetuning_experiment/comparison.csv"

# Fields used for per-position metrics
T1_FIELD = "t1_direction_correct"
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
            except json.JSONDecodeError as e:
                print(f"WARNING: skipping malformed JSON at line {lineno}: {e}", flush=True)
    return records


MOVE_FORMATS = {"cot", "fen_only", "pgn+fen", "move_only"}
EVAL_FORMATS = {"cot", "fen_only", "pgn+fen", "eval_only"}
EXPLANATION_FORMATS = {"cot", "fen_only", "pgn+fen", "explanation_only"}


def compute_board_accuracy(record: dict) -> float | None:
    """Return board F1 only for explicitly recorded board-description output."""
    response = record.get("t0_response")
    if response is None and record.get(PROMPT_FIELD) == "board_description":
        response = record.get("response")
    if record.get(FEN_FIELD) is None or response is None:
        return None
    if not isinstance(response, str):
        raise ValueError("T0 response must be text")
    return verify_board_description(record[FEN_FIELD], response)["f1"]


def _mean(values):
    return sum(values) / len(values) if values else None


def _explanation_score(record):
    if record.get("t3_score_v2") is not None:
        return record["t3_score_v2"]
    side_score = record.get("t3_p1_side_correct")
    theme_score = rescore_t3_theme(record.get("theme"), record.get("t3_explanation"))
    return side_score + theme_score if side_score is not None and theme_score is not None else None


def _record_values(record):
    prompt = record.get(PROMPT_FIELD)
    if prompt not in MOVE_FORMATS | EVAL_FORMATS | EXPLANATION_FORMATS | {"board_description"}:
        raise ValueError(f"Unknown prompt_format: {prompt!r}")
    direction = record.get(T1_FIELD)
    legal = record.get(T2_LEGAL_FIELD)
    if direction is not None and not isinstance(direction, bool):
        raise ValueError("t1_direction_correct must be boolean or null")
    if legal is not None and not isinstance(legal, bool):
        raise ValueError("t2_legal must be boolean or null")
    return {
        "t1": int(direction) if prompt in EVAL_FORMATS and direction is not None else None,
        "t2_legal": int(legal is True) if prompt in MOVE_FORMATS else None,
        "t2_cpl": derive_endpoint_metrics(record)["t2_cpl_clamped"] if prompt in MOVE_FORMATS and legal is True else None,
        "t3": _explanation_score(record) if prompt in EXPLANATION_FORMATS else None,
        "t0": compute_board_accuracy(record),
    }


def aggregate_model_metrics(records: list[dict]) -> dict[str, dict]:
    """Aggregate only applicable, measured tasks; report every denominator."""
    buckets = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(int)
    for record in records:
        model = record.get(MODEL_FIELD)
        if not isinstance(model, str) or not model:
            raise ValueError("Missing model")
        counts[model] += 1
        for task, value in _record_values(record).items():
            if value is not None:
                buckets[model][task].append(float(value))
    output = {}
    for model, count in counts.items():
        values = buckets[model]
        output[model] = {
            "t1_acc": _mean(values["t1"]),
            "t2_legal_rate": _mean(values["t2_legal"]),
            "mean_cpl": _mean(values["t2_cpl"]),
            "t3_mean": _mean(values["t3"]),
            "t0_mean": _mean(values["t0"]),
            "n": count,
            **{task + "_n": len(values[task]) for task in ("t1", "t0", "t3")},
            "t2_n": len(values["t2_legal"]),
            "cpl_n": len(values["t2_cpl"]),
        }
    return output


def _index_observations(records, *, tuned):
    indexed = {}
    for record in records:
        if record.get("job_type", "standard") != "standard":
            continue
        model = record.get(MODEL_FIELD)
        if not isinstance(model, str) or not model:
            raise ValueError("Missing model")
        if tuned and not model.endswith("-board-ft"):
            raise ValueError(f"Fine-tuned model lacks -board-ft suffix: {model}")
        position = record.get(POS_ID_FIELD)
        if position is None:
            raise ValueError("Missing position_id")
        key = (strip_ft_suffix(model), str(position), record.get(PROMPT_FIELD),
               record.get("attempt", record.get("trial", 1)))
        if key in indexed:
            raise ValueError(f"Duplicate comparison observation: {key}")
        indexed[key] = record
    return indexed


def compare_matched_records(base_records, ft_records):
    """Compare identical position/prompt/trial cohorts, reporting exclusions."""
    base_index = _index_observations(base_records, tuned=False)
    ft_index = _index_observations(ft_records, tuned=True)
    pairs = defaultdict(list)
    for key in sorted(base_index.keys() & ft_index.keys(), key=str):
        base, tuned = base_index[key], ft_index[key]
        if base.get(FEN_FIELD) != tuned.get(FEN_FIELD):
            raise ValueError(f"Conflicting FEN for matched observation: {key}")
        for truth_field in ("t1_stockfish_eval", "stockfish_eval"):
            if base.get(truth_field) is not None and tuned.get(truth_field) is not None:
                if base[truth_field] != tuned[truth_field]:
                    raise ValueError(f"Conflicting reference evaluation for matched observation: {key}")
        pairs[key[0]].append((base, tuned))
    rows = []
    for model, cohort in sorted(pairs.items()):
        base_metrics = aggregate_model_metrics([base for base, _ in cohort])
        tuned_metrics = aggregate_model_metrics([tuned for _, tuned in cohort])
        row = compute_deltas(base_metrics, tuned_metrics)[0]
        row.update(t3_metric="t3_score_v2", cpl_metric="recorded_endpoint_clamped_loss", matched_n=len(cohort),
                   base_unmatched_n=sum(key[0] == model for key in base_index) - len(cohort),
                   ft_unmatched_n=sum(key[0] == model for key in ft_index) - len(cohort))
        _add_paired_metrics(row, cohort)
        rows.append(row)
    return rows


def _assistance_policy(record):
    prompt = record.get(PROMPT_FIELD)
    if prompt == "move_only":
        return "legal_list_scan" if record.get("t2_rescue_path") == "legal_list_scan" else None
    if prompt not in MOVE_FORMATS:
        return None
    enabled = record.get("t2_rescue_enabled")
    return ("combined_rescue_enabled" if enabled else "combined_rescue_disabled") if isinstance(enabled, bool) else None


def _legality_pair_exclusion(base, tuned):
    base_policy, tuned_policy = _assistance_policy(base), _assistance_policy(tuned)
    if base_policy is None or tuned_policy is None:
        return "unknown_assistance"
    return None if base_policy == tuned_policy else "different_assistance_policy"


def _add_paired_metrics(row, cohort):
    # Availability differs from record coverage: restrict each metric to pairs
    # where both sides were measured, rather than comparing unequal subsets.
    task_keys = {"t1": ("t1_acc", "delta_t1_acc"),
                 "t2_legal": ("t2_legal", "delta_t2_legal"),
                 "t2_cpl": ("mean_cpl", "delta_cpl"),
                 "t3": ("t3_mean", "delta_t3"),
                 "t0": ("t0_mean", "delta_t0")}
    paired_values = []
    legality_exclusions = defaultdict(int)
    provenance_fields = ("t2_reference_engine", "t2_reference_budget",
                         "t2_reference_units", "t2_scoring_version")
    for base, tuned in cohort:
        base_values, ft_values = _record_values(base), _record_values(tuned)
        if base.get(PROMPT_FIELD) in MOVE_FORMATS:
            exclusion = _legality_pair_exclusion(base, tuned)
            if exclusion:
                base_values["t2_legal"] = ft_values["t2_legal"] = None
                legality_exclusions[exclusion] += 1
        if not all(base.get(field) is not None and base.get(field) == tuned.get(field)
                   for field in provenance_fields):
            base_values["t2_cpl"] = ft_values["t2_cpl"] = None
        paired_values.append((base_values, ft_values))
    row["t2_legal_comparability"] = (
        json.dumps(dict(sorted(legality_exclusions.items()))) if legality_exclusions
        else "compatible_recorded_assistance_policy"
    )
    for task, (column, delta_column) in task_keys.items():
        available = [(base[task], tuned[task]) for base, tuned in paired_values
                     if base[task] is not None and tuned[task] is not None]
        base_mean = _mean([base for base, _ in available])
        ft_mean = _mean([tuned for _, tuned in available])
        row["base_" + column] = base_mean
        row["ft_" + column] = ft_mean
        row[delta_column] = round(ft_mean - base_mean, 4) if available else None
        row[task + "_paired_n"] = len(available)


def compute_deltas(
    base_metrics: dict[str, dict],
    ft_metrics: dict[str, dict],
) -> list[dict]:
    """Format deltas for already-matched metric inputs; CLI joins records first."""
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
            "ft_t0_mean": ft_data.get("t0_mean"),
            "base_t0_mean": base_data.get("t0_mean"),
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
        t0 = f"{row['ft_t0_mean']:.3f}" if row["ft_t0_mean"] is not None else "—"
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

    rows = compare_matched_records(base_records, ft_records)

    if not rows:
        logger.warning("No matching model pairs found. Check that FT tags end with -board-ft.")
        sys.exit(1)

    write_csv(rows, args.output)
    logger.info(f"\nComparison CSV written: {args.output}")

    print_summary(rows, logger)


if __name__ == "__main__":
    main()
