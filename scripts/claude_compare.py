"""Offline C1/C4 comparison of saved Claude and local plain-board responses."""

import argparse
import json
import math
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import plain_board_compare as notation
from src.metrics import EVAL_CLAMP_CP, compute_best_move_vs_random, load_results_df
from src.provenance import build_provenance, file_sha256
from src.utils import load_config

FULL_VARIANTS = ("claude-haiku:5.5", "claude-opus:5.5-low")
LOCAL_MODEL = "gemma4:31b"
FORMATS = ("fen_only", "board_text", "board_text+legal")
TIERS = ("easy", "medium", "hard", "extreme")
C1_MIN_DROP = 0.05
FLOAT_TOLERANCE = 1e-12
TOKENS_PER_MILLION = 1_000_000
MAX_CLAMPED_CPL = 2 * EVAL_CLAMP_CP
FIRST_ROUND = 1
RETRY_ROUND = 2
DEFAULT_BATCH_DISCOUNT = 0.5
PENDING = "pending"
CLEARLY_BETTER = "frontier model plays clearly better"
NO_CLEAR_DIFFERENCE = "no clear difference"
KEYS = ["model", "position_id", "prompt_format"]
NUMERIC_COLUMNS = ("t1_model_eval", "t1_stockfish_eval", "t2_cpl", "t2_cpl_clamped",
                   "t2_eval_before", "t2_first_attempt_legal")
LIMITATIONS = [
    "C4's registered absolute board-text rho threshold does not itself establish improvement over FEN.",
    "Blunder comparisons condition on legal/scored paired moves; coverage is reported separately.",
    "Claude model sizes are unknown; hosted model outputs cannot be reproduced exactly.",
    "Claude used API default sampling settings; local move retries used temperature zero.",
    "The positions are public, so training-data contamination cannot be excluded.",
]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and math.isfinite(value)


def _frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for key in (*KEYS, "fen", "t2_legal", "t2_move", "t2_best_move", "t3_side_claimed"):
        if key not in frame:
            frame[key] = pd.Series(index=frame.index, dtype=object)
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame.get(column, pd.Series(index=frame.index, dtype=float)), errors="coerce")
    if "job_type" in frame:
        frame = frame[frame["job_type"].fillna("standard") == "standard"]
    return frame


def _dedupe(frame: pd.DataFrame) -> pd.DataFrame:
    return _frame(frame).dropna(subset=KEYS).drop_duplicates(KEYS, keep="first").reset_index(drop=True)


def _valid(frame: pd.DataFrame, measure: str) -> pd.Series:
    column = notation.MEASURE_COLUMNS[measure]
    if measure == "side_kappa":
        return frame[column].isin(("White", "Black", "Equal")) & np.isfinite(frame["t1_stockfish_eval"])
    valid = np.isfinite(frame[column])
    if measure == "t1_rho":
        valid &= np.isfinite(frame["t1_stockfish_eval"])
    elif measure == "first_answer_legal":
        valid &= frame[column].isin((0, 1))
    elif measure == "blunder_rate":
        valid &= frame["t2_legal"].eq(True) & np.isfinite(frame["t2_eval_before"]) & frame[column].between(0, MAX_CLAMPED_CPL)
    return valid


def _safe_interval(frame: pd.DataFrame, measure) -> tuple[float, float] | None:
    if frame.empty:
        return None
    # Constant resamples can make rho undefined; only finite values are meaningful.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        interval = notation.bootstrap_ci(frame, measure)
    return interval if all(_finite(value) for value in interval) else None


def _paired_measure(frame: pd.DataFrame, name: str, left: str, right: str) -> dict[str, Any]:
    valid = frame[_valid(frame, name)]
    first, second = notation.pair_rows(valid, left, right, notation.MEASURE_COLUMNS[name])
    same_fen = first.fen.eq(second.fen) & first.fen.map(lambda value: isinstance(value, str)).astype(bool)
    mismatches = int((~same_fen).sum())
    first, second = first[same_fen], second[same_fen]
    measure = notation.MEASURES[name]
    first_value = measure(first) if len(first) else math.nan
    second_value = measure(second) if len(second) else math.nan
    return {
        "n_pairs": len(first), "left": left, "right": right,
        "left_value": first_value, "right_value": second_value,
        "left_ci": _safe_interval(first, measure), "right_ci": _safe_interval(second, measure),
        "difference": second_value - first_value,
        "left_usable": int((valid.prompt_format == left).sum()),
        "right_usable": int((valid.prompt_format == right).sum()),
        "fen_mismatches": mismatches,
    }


def _complete_formats(frame: pd.DataFrame, formats: tuple[str, ...], expected: int | None) -> bool:
    if expected is None or expected <= 0:
        return False
    positions = [set(frame.loc[frame.prompt_format == fmt, "position_id"]) for fmt in formats]
    same_fen = frame.groupby("position_id").fen.nunique(dropna=False).eq(1).all()
    has_fen = frame.fen.map(lambda value: isinstance(value, str) and bool(value.strip())).all()
    return bool(positions) and same_fen and has_fen and all(
        len(group) == expected and group == positions[0] for group in positions
    )


def _all_legal_scored(frame: pd.DataFrame) -> bool:
    return bool((~frame.t2_legal.eq(True) | _valid(frame, "blunder_rate")).all())


def c1_verdict(local_rate: float, claude_rate: float, local_ci, claude_ci) -> str:
    if not all(_finite(value) for value in (local_rate, claude_rate)) or local_ci is None or claude_ci is None:
        return PENDING
    drop = local_rate - claude_rate
    meets_size = drop >= C1_MIN_DROP or math.isclose(drop, C1_MIN_DROP, abs_tol=FLOAT_TOLERANCE, rel_tol=0)
    return CLEARLY_BETTER if meets_size and claude_ci[1] < local_ci[0] else NO_CLEAR_DIFFERENCE


def c1_result(claude: pd.DataFrame, local: pd.DataFrame, expected_positions: int | None) -> dict[str, Any]:
    claude, local = _dedupe(claude), _dedupe(local)
    left = local[local.prompt_format == "fen_only"].assign(model="paired", prompt_format="local")
    right = claude[claude.prompt_format == "fen_only"].assign(model="paired", prompt_format="claude")
    combined = pd.concat([left, right], ignore_index=True)
    comparison = _paired_measure(combined, "blunder_rate", "local", "claude")
    complete = _complete_formats(combined, ("local", "claude"), expected_positions)
    scoring_complete = _all_legal_scored(combined)
    intervals_valid = comparison["left_ci"] is not None and comparison["right_ci"] is not None
    result = c1_verdict(comparison["left_value"], comparison["right_value"],
                        comparison["left_ci"], comparison["right_ci"])
    if not complete or not scoring_complete or not intervals_valid:
        result = PENDING
    return {"status": "complete" if result != PENDING else PENDING, "verdict": result,
            "local_model": LOCAL_MODEL, "comparison": comparison,
            "blunder_drop": comparison["left_value"] - comparison["right_value"],
            "cohort_complete": complete, "scoring_complete": scoring_complete,
            "expected_positions": expected_positions}


def _c4_inputs(primary: dict[str, dict]) -> dict[str, Any]:
    blunder = primary["blunder_rate"]
    left_ci, right_ci = blunder["left_ci"], blunder["right_ci"]
    return {
        "blunder_drop": blunder["left_value"] - blunder["right_value"],
        "blunder_cis_separate": bool(right_ci[1] < left_ci[0]) if left_ci and right_ci else None,
        "t1_rho": primary["t1_rho"]["right_value"],
        "side_kappa": primary["side_kappa"]["right_value"],
        "first_answer_legal": primary["first_answer_legal"]["right_value"],
    }


def c4_result(rows: pd.DataFrame, expected_positions: int | None) -> dict[str, Any]:
    rows = _dedupe(rows)
    primary = {name: _paired_measure(rows, name, "fen_only", "board_text") for name in notation.MEASURES}
    values = _c4_inputs(primary)
    complete = _complete_formats(rows, FORMATS, expected_positions)
    available = values["blunder_cis_separate"] is not None and all(
        _finite(values[key]) for key in ("blunder_drop", "t1_rho", "side_kappa", "first_answer_legal")
    )
    scoring_complete = _all_legal_scored(rows)
    outcome = notation.verdict(values) if complete and available and scoring_complete else PENDING
    return {"status": "complete" if outcome != PENDING else PENDING, "verdict": outcome,
            "primary": primary, "verdict_inputs": values, "cohort_complete": complete,
            "scoring_complete": scoring_complete,
            "limitations": LIMITATIONS}


def _mean(values: pd.Series) -> float:
    return float(values.mean()) if values.notna().any() else math.nan


def _format_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"records": len(rows), "positions": int(rows.position_id.nunique())}
    for name, measure in notation.MEASURES.items():
        usable = rows[_valid(rows, name)]
        result[name] = measure(usable) if len(usable) else math.nan
        result[f"{name}_n"] = len(usable)
    result["final_legal_rate"] = _mean(rows.t2_legal.eq(True).astype(float))
    best = compute_best_move_vs_random(rows)
    result["best_move_vs_random"] = best.iloc[0].to_dict() if len(best) else None
    return result


def _request_cost(row: dict, config: dict) -> tuple[float | None, bool]:
    stored = row.get("cost_usd")
    if _finite(stored) and stored >= 0:
        return float(stored), False
    price = config.get("claude", {}).get("prices_per_mtok", {}).get(row.get("api_model"))
    tokens = (row.get("input_tokens"), row.get("output_tokens"))
    if price is None or not all(_finite(value) and value >= 0 for value in tokens):
        return None, False
    total = tokens[0] * price["input"] + tokens[1] * price["output"]
    return total * config["claude"].get("batch_discount", DEFAULT_BATCH_DISCOUNT) / TOKENS_PER_MILLION, True


def summarize_cost(raw_responses: list[dict] | None, tag: str, config: dict) -> dict[str, Any]:
    if raw_responses is None:
        return {"status": "unavailable", "cost_usd": None, "source": "raw response ledger missing"}
    selected = [row for row in raw_responses if row.get("tag") == tag]
    if not selected:
        return {"status": "unavailable", "cost_usd": None,
                "source": "no raw transactions recorded for variant", "requests": 0}
    unique, unidentified = {}, 0
    for index, row in enumerate(selected):
        identifier = row.get("custom_id")
        unidentified += identifier is None
        unique[(row.get("batch_id"), identifier if identifier is not None else index)] = row
    amounts = [_request_cost(row, config) for row in unique.values()]
    missing = sum(amount is None for amount, _ in amounts)
    return {"status": "complete" if not missing and not unidentified else "incomplete",
            "cost_usd": sum(amount for amount, _ in amounts if amount is not None) if not missing else None,
            "source": "unique raw response transactions", "requests": len(unique),
            "duplicate_ledger_records": len(selected) - len(unique), "unidentified_requests": unidentified,
            "reconstructed_requests": sum(reconstructed for _, reconstructed in amounts),
            "missing_cost_requests": missing,
            "first_reply_max_tokens": sum(row.get("round") == FIRST_ROUND and row.get("stop_reason") == "max_tokens" for row in unique.values()),
            "retry_max_tokens": sum(row.get("round") == RETRY_ROUND and row.get("stop_reason") == "max_tokens" for row in unique.values())}


def expected_positions(config: dict) -> int | None:
    benchmark = config.get("benchmark", {})
    caps = [benchmark.get(key) for key in ("sample_per_tier", "max_positions_per_tier")]
    positive = [int(value) for value in caps if isinstance(value, (int, float)) and value > 0]
    return min(positive) * len(TIERS) if positive else None


def _variant_summary(frame, local, tag, expected, raw, config):
    original = frame[frame.model == tag]
    rows = _dedupe(original)
    return {
        "coverage": {"observed_records": len(original), "unique_records": len(rows),
                     "duplicate_records": len(original) - len(rows),
                     "expected_positions_per_format": expected,
                     "by_format": {fmt: int((rows.prompt_format == fmt).sum()) for fmt in FORMATS}},
        "C1": c1_result(rows, local, expected), "C4": c4_result(rows, expected),
        "by_format": {fmt: _format_metrics(rows[rows.prompt_format == fmt]) for fmt in FORMATS},
        "cost": summarize_cost(raw, tag, config),
    }


def build_summary(claude_df: pd.DataFrame, local_df: pd.DataFrame, config: dict,
                  raw_responses: list[dict] | None = None) -> dict[str, Any]:
    claude, local = _frame(claude_df), _frame(local_df)
    local = local[local.model == LOCAL_MODEL]
    expected = expected_positions(config)
    pilots = claude[~claude.model.isin(FULL_VARIANTS)]
    return json_safe({
        "seed": notation.SEED, "resamples": notation.RESAMPLES,
        "full_variants": FULL_VARIANTS, "local_comparator": LOCAL_MODEL,
        "variants": {tag: _variant_summary(claude, local, tag, expected, raw_responses, config)
                     for tag in FULL_VARIANTS},
        "excluded_pilots": pilots.model.value_counts().to_dict(),
        "pilot_costs": {tag: summarize_cost(raw_responses, tag, config) for tag in pilots.model.dropna().unique()},
        "C2": {"status": "not run", "reason": "Full Sonnet thinking comparison deferred"},
        "C3": {"status": "not run", "reason": "Full Sonnet thinking comparison deferred"},
        "limitations": LIMITATIONS,
    })


def read_jsonl(path: Path) -> tuple[list[dict], dict[str, int]]:
    records, malformed = [], 0
    if not path.is_file():
        return records, {"records": 0, "malformed_lines": 0, "file_missing": True}
    with path.open() as source:
        for line in source:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(row, dict):
                records.append(row)
            else:
                malformed += 1
    return records, {"records": len(records), "malformed_lines": malformed, "file_missing": False}


def read_scored(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    records, audit = read_jsonl(Path(path))
    audit["missing_fen"] = sum(not isinstance(row.get("fen"), str) for row in records)
    if not records:
        return _frame(pd.DataFrame()), audit
    # Existing records carry FEN. Avoid reading multi-gigabyte original tier files.
    with tempfile.TemporaryDirectory(prefix="claude-analysis-") as temporary:
        clean_path = Path(temporary) / "scored.jsonl"
        clean_path.write_text("".join(json.dumps(row) + "\n" for row in records))
        frame = load_results_df(str(clean_path), data_dir=temporary)
    frame = _frame(frame)
    audit["cpl_without_stored_reference"] = int((frame.t2_cpl.notna() & frame.t2_eval_before.isna()).sum())
    return frame, audit


def comparison_table(summary: dict) -> pd.DataFrame:
    rows = []
    for tag, variant in summary["variants"].items():
        for fmt, metrics in variant["by_format"].items():
            best = metrics.get("best_move_vs_random") or {}
            rows.append({"variant": tag, "prompt_format": fmt,
                         **{key: value for key, value in metrics.items() if key != "best_move_vs_random"},
                         "best_move_rate": best.get("best_move_rate"), "random_best_move_rate": best.get("random_best_move_rate"),
                         "best_move_lift": best.get("ratio_vs_random"),
                         "cost_usd_variant": variant["cost"]["cost_usd"],
                         "C1": variant["C1"]["verdict"], "C4": variant["C4"]["verdict"]})
    return pd.DataFrame(rows)


def write_report(summary: dict, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "claude_summary.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n")
    comparison_table(summary).to_csv(output / "claude_comparison.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config_claude.yaml")
    parser.add_argument("--results")
    parser.add_argument("--responses")
    parser.add_argument("--local-results", default="results/plain_board/evaluations.jsonl")
    parser.add_argument("--local-config", default="config/config_plain_board.yaml")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = config.get("paths", {})
    results = args.results or paths.get("results_file", "results/claude/evaluations.jsonl")
    responses = args.responses or paths.get("responses_file")
    claude, claude_audit = read_scored(results)
    local, local_audit = read_scored(args.local_results)
    raw, raw_audit = read_jsonl(Path(responses)) if responses else (None, {"file_missing": True})
    summary = build_summary(claude, local, config, raw if not raw_audit.get("file_missing") else None)
    summary["input_audit"] = {"claude": claude_audit, "local": local_audit, "raw_responses": raw_audit}
    summary["provenance"] = {
        "claude": build_provenance(args.config, results),
        "local": build_provenance(args.local_config, args.local_results),
        "responses_file": responses, "responses_sha256": file_sha256(Path(responses)) if responses else None,
    }
    write_report(summary, args.output_dir or paths.get("metrics_dir", "results/claude/metrics"))


if __name__ == "__main__":
    main()
