"""The notation check: plain-board prompts against a same-run fen_only control.

Pairs records on (model, position_id), computes the pre-registered measures N1–N4 with bootstrap
CIs over positions, and applies the decision rule fixed in specs/plain-board-prompts.md.

    python scripts/plain_board_compare.py --config config/config_plain_board.yaml
"""

import argparse
import json
import logging
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import chess
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_client import BOARD_TEXT_FORMAT, BOARD_TEXT_LEGAL_FORMAT, parse_response  # noqa: E402
from src.metrics import (  # noqa: E402
    BLUNDER_CPL_CP,
    _side_mark_rows,
    _side_mark_stats,
    _spearman,
    load_results_df,
)
from src.provenance import build_provenance  # noqa: E402
from src.utils import load_config, setup_logging  # noqa: E402

logger = logging.getLogger("chess_llm_bench")

CONTROL = "fen_only"
SEED = 42
RESAMPLES = 1000
CI_TAIL = 0.025  # 95% percentile interval

# Pre-registered thresholds (specs/plain-board-prompts.md, "Success criteria").
BLUNDER_DROP_LIMIT = 0.05
RHO_NOT_BARRIER = 0.20
RHO_MAJOR_BARRIER = 0.30
KAPPA_NOT_BARRIER = 0.20
FIRST_LEGAL_NOT_BARRIER = 0.25

NOT_BARRIER = "notation is not the main barrier"
MAJOR_BARRIER = "notation was a major barrier"
MIXED = "mixed"
PENDING = "pending: incomplete measures"
SIDE_LABELS = ("White", "Black", "Equal")
TIERS = ("easy", "medium", "hard", "extreme")
COMPARISON_FORMATS = (CONTROL, BOARD_TEXT_FORMAT, BOARD_TEXT_LEGAL_FORMAT)
INTERPRETATION_WARNINGS = [
    "The registered major-barrier rule uses absolute board-text correlation; it does not establish improvement over FEN.",
    "Pooled correlation is sensitive to model output scales; inspect the descriptive per-model results.",
    "Blunder rates and side agreement condition on usable paired outputs; inspect measure-specific coverage.",
]

PAIR_KEYS = ["model", "position_id"]
# Each measure is paired on the column it needs, so a pair counts only when both sides have a value.
MEASURE_COLUMNS = {
    "blunder_rate": "t2_cpl_clamped",
    "first_answer_legal": "t2_first_attempt_legal",
    "t1_rho": "t1_model_eval",
    "side_kappa": "t3_side_claimed",
}


# ---------- measures ----------

def _blunder_rate(rows: pd.DataFrame) -> float:
    cpl = rows["t2_cpl_clamped"].dropna()
    return float((cpl >= BLUNDER_CPL_CP).mean()) if len(cpl) else float("nan")


def _first_answer_legal(rows: pd.DataFrame) -> float:
    legal = rows["t2_first_attempt_legal"].dropna()
    return float(legal.mean()) if len(legal) else float("nan")


def _t1_rho(rows: pd.DataFrame) -> float:
    return _spearman(rows.dropna(subset=["t1_model_eval", "t1_stockfish_eval"]))


def _side_kappa(rows: pd.DataFrame) -> float:
    marked = _side_mark_rows(rows)
    return float(_side_mark_stats(marked)["cohen_kappa"]) if len(marked) else float("nan")


MEASURES: dict[str, Callable[[pd.DataFrame], float]] = {
    "blunder_rate": _blunder_rate,
    "first_answer_legal": _first_answer_legal,
    "t1_rho": _t1_rho,
    "side_kappa": _side_kappa,
}


def pooled_measures(rows: pd.DataFrame) -> dict[str, float]:
    return {name: measure(rows) for name, measure in MEASURES.items() if MEASURE_COLUMNS[name] in rows}


# ---------- pairing and bootstrap ----------

def pair_rows(df: pd.DataFrame, left_format: str, right_format: str, column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows of each format whose (model, position) also has a value in the other format."""
    def valued(fmt: str) -> pd.DataFrame:
        rows = _format_rows(df, fmt).drop_duplicates(PAIR_KEYS)
        return rows[_valid_measure(rows, column)].set_index(PAIR_KEYS)

    left, right = valued(left_format), valued(right_format)
    shared = left.index.intersection(right.index).sort_values()
    return left.loc[shared].reset_index(), right.loc[shared].reset_index()


def _column(rows: pd.DataFrame, column: str) -> pd.Series:
    return rows[column] if column in rows else pd.Series(np.nan, index=rows.index)


def _finite(rows: pd.DataFrame, column: str) -> pd.Series:
    return np.isfinite(pd.to_numeric(_column(rows, column), errors="coerce"))


def _valid_measure(rows: pd.DataFrame, column: str) -> pd.Series:
    """Validity shared by pairing and coverage, including each statistic's reference."""
    if column == "t3_side_claimed":
        return _column(rows, column).isin(SIDE_LABELS) & _finite(rows, "t1_stockfish_eval")
    valid = _finite(rows, column)
    if column == "t1_model_eval":
        return valid & _finite(rows, "t1_stockfish_eval")
    if column == "t2_first_attempt_legal":
        return valid & _column(rows, column).isin([0, 1])
    if column == "t2_cpl_clamped":
        valid &= pd.to_numeric(_column(rows, column), errors="coerce") >= 0
        if "t2_legal" in rows:
            valid &= rows["t2_legal"].eq(True)
    return valid


def _format_rows(df: pd.DataFrame, fmt: str) -> pd.DataFrame:
    rows = df[_column(df, "prompt_format") == fmt].copy()
    for key in PAIR_KEYS:
        if key not in rows:
            rows[key] = pd.Series(dtype=object)
    return rows.dropna(subset=PAIR_KEYS)


def _coverage_side(df: pd.DataFrame, fmt: str, column: str, n_pairs: int) -> dict[str, int]:
    rows = _format_rows(df, fmt)
    unique = rows.drop_duplicates(PAIR_KEYS)
    usable = int(_valid_measure(unique, column).sum())
    return {"observed": len(rows), "unique": len(unique), "duplicates": len(rows) - len(unique),
            "usable": usable, "missing_or_invalid": len(unique) - usable,
            "paired": n_pairs, "unmatched_usable": usable - n_pairs}


def measure_coverage(df: pd.DataFrame, left: str, right: str, column: str, n_pairs: int) -> dict[str, Any]:
    coverage: dict[str, Any] = {
        "left": _coverage_side(df, left, column, n_pairs),
        "right": _coverage_side(df, right, column, n_pairs), "by_model": {},
    }
    for model, rows in df.groupby("model") if "model" in df else []:
        paired, _ = pair_rows(rows, left, right, column)
        coverage["by_model"][model] = {
            "n_pairs": len(paired), "left": _coverage_side(rows, left, column, len(paired)),
            "right": _coverage_side(rows, right, column, len(paired)),
        }
    return coverage


def bootstrap_ci(rows: pd.DataFrame, measure: Callable[[pd.DataFrame], float],
                 seed: int = SEED, resamples: int = RESAMPLES) -> tuple[float, float]:
    """Percentile CI from resampling positions (all models' rows for a position move together)."""
    positions = rows["position_id"].to_numpy()
    unique = np.unique(positions)
    row_ids = {pid: np.flatnonzero(positions == pid) for pid in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        values.append(measure(rows.iloc[np.concatenate([row_ids[pid] for pid in drawn])]))
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return (math.nan, math.nan)
    low, high = np.quantile(finite, [CI_TAIL, 1 - CI_TAIL])
    return float(low), float(high)


def compare_measure(df: pd.DataFrame, name: str, left_format: str, right_format: str) -> dict[str, Any]:
    left, right = pair_rows(df, left_format, right_format, MEASURE_COLUMNS[name])
    measure = MEASURES[name]
    left_value = measure(left) if len(left) else math.nan
    right_value = measure(right) if len(right) else math.nan
    return {
        "measure": name, "left": left_format, "right": right_format, "n_pairs": len(left),
        "left_value": left_value, "left_ci": bootstrap_ci(left, measure) if len(left) else None,
        "right_value": right_value, "right_ci": bootstrap_ci(right, measure) if len(right) else None,
        "difference": right_value - left_value,
        "coverage": measure_coverage(df, left_format, right_format, MEASURE_COLUMNS[name], len(left)),
    }


# ---------- decision rule ----------

def verdict(values: dict[str, Any]) -> str:
    """Apply the pre-registered rule to the board_text-vs-fen_only measures."""
    numeric = ("blunder_drop", "t1_rho", "side_kappa", "first_answer_legal")
    if any(not _is_finite(values.get(key)) for key in numeric) or values.get("blunder_cis_separate") is None:
        return PENDING
    major = (values["blunder_drop"] >= BLUNDER_DROP_LIMIT and values["blunder_cis_separate"]) \
        or values["t1_rho"] >= RHO_MAJOR_BARRIER
    if major:
        return MAJOR_BARRIER
    holds = (values["blunder_drop"] < BLUNDER_DROP_LIMIT and values["t1_rho"] < RHO_NOT_BARRIER
             and values["side_kappa"] < KAPPA_NOT_BARRIER
             and values["first_answer_legal"] < FIRST_LEGAL_NOT_BARRIER)
    return NOT_BARRIER if holds else MIXED


def verdict_inputs(primary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    blunder = primary["blunder_rate"]
    left_ci, right_ci = blunder["left_ci"], blunder["right_ci"]
    cis_valid = left_ci is not None and right_ci is not None and all(
        _is_finite(value) for value in (*left_ci, *right_ci)
    )
    return {
        "blunder_drop": blunder["left_value"] - blunder["right_value"],
        "blunder_cis_separate": bool(right_ci[1] < left_ci[0]) if cis_valid else None,
        "t1_rho": primary["t1_rho"]["right_value"],
        "side_kappa": primary["side_kappa"]["right_value"],
        "first_answer_legal": primary["first_answer_legal"]["right_value"],
    }


# ---------- wrong-side answers ----------

def legal_for_other_side(fen: str, san: Any) -> bool:
    """Would this move be legal if the other side were to move?"""
    if not isinstance(san, str) or not san:
        return False
    board = chess.Board(fen)
    board.turn = not board.turn
    board.ep_square = None
    try:
        return board.parse_san(san) in board.legal_moves
    except ValueError:
        return False


def wrong_side_share(rows: pd.DataFrame) -> float:
    """Share of illegal first answers that are legal moves for the side not to move."""
    illegal = rows[_column(rows, "t2_first_attempt_legal") == 0]
    if illegal.empty or not {"raw_response", "fen"} <= set(illegal):
        return float("nan")
    moves = illegal["raw_response"].map(lambda raw: parse_response(raw)["move"] if isinstance(raw, str) else None)
    return float(np.mean([legal_for_other_side(fen, move) for fen, move in zip(illegal["fen"], moves)]))


# ---------- tables and summary ----------

def per_model_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per model and plain format: each measure on the control and the plain format, paired."""
    rows = []
    for model, model_df in df.groupby("model", sort=True) if "model" in df else []:
        for fmt in (BOARD_TEXT_FORMAT, BOARD_TEXT_LEGAL_FORMAT):
            for name, column in MEASURE_COLUMNS.items():
                left, right = pair_rows(model_df, CONTROL, fmt, column)
                rows.append({"model": model, "format": fmt, "measure": name, "n_pairs": len(left),
                             "control": MEASURES[name](left) if len(left) else np.nan,
                             "plain": MEASURES[name](right) if len(right) else np.nan})
    table = pd.DataFrame(rows, columns=["model", "format", "measure", "n_pairs", "control", "plain"])
    table["difference"] = table["plain"] - table["control"]
    return table


def dropped_pairs(df: pd.DataFrame) -> dict[str, int]:
    """Per model: positions with a control record but no matching plain record, or the reverse."""
    counts = {}
    for model, model_df in df.groupby("model", sort=True) if "model" in df else []:
        by_format = {fmt: set(rows["position_id"]) for fmt, rows in model_df.groupby("prompt_format")}
        control = by_format.get(CONTROL, set())
        counts[model] = sum(len(control ^ by_format.get(fmt, set()))
                            for fmt in (BOARD_TEXT_FORMAT, BOARD_TEXT_LEGAL_FORMAT))
    return counts


def ollama_digests() -> dict[str, str] | None:
    """Model tag -> digest from `ollama list`; None (logged) if Ollama can't be asked."""
    try:
        listing = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning(f"Could not read model digests from ollama list: {exc}")
        return None
    return {cols[0]: cols[1] for cols in (line.split() for line in listing.splitlines()[1:]) if len(cols) > 1}


def cohort_coverage(df: pd.DataFrame, expected_models: list[str] | None,
                    expected_positions: int | None) -> dict[str, Any]:
    observed = set(_column(df, "model").dropna())
    expected = set(expected_models) if expected_models is not None else observed
    counts = {
        model: {fmt: int(_format_rows(df[_column(df, "model") == model], fmt)["position_id"].nunique())
                for fmt in COMPARISON_FORMATS}
        for model in sorted(expected | observed)
    }
    position_sets = [
        set(_format_rows(df[_column(df, "model") == model], fmt)["position_id"])
        for model in sorted(expected | observed) for fmt in COMPARISON_FORMATS
    ]
    shared_positions = set.intersection(*position_sets) if position_sets else set()
    verified = expected_models is not None and expected_positions is not None
    complete = verified and observed == expected and len(shared_positions) == expected_positions and all(
        count == expected_positions for formats in counts.values() for count in formats.values()
    )
    return {"status": "complete" if complete else "incomplete" if verified else "cohort_unverified",
            "complete": bool(complete) if verified else None, "expected_models": expected_models,
            "expected_positions_per_format": expected_positions, "by_model": counts,
            "positions_shared_across_models_and_formats": len(shared_positions),
            "missing_models": sorted(expected - observed), "unexpected_models": sorted(observed - expected)}


def expected_sample_size(config: dict[str, Any]) -> int | None:
    """The generator clips the second sample to the first positive tier cap."""
    benchmark = config.get("benchmark", {})
    caps = [int(benchmark.get(key) or 0) for key in ("sample_per_tier", "max_positions_per_tier")]
    positive = [cap for cap in caps if cap > 0]
    return min(positive) * len(TIERS) if positive else None


def build_summary(df: pd.DataFrame, expected_models: list[str] | None = None,
                  expected_positions: int | None = None) -> dict[str, Any]:
    primary = {name: compare_measure(df, name, CONTROL, BOARD_TEXT_FORMAT) for name in MEASURES}
    secondary = {name: compare_measure(df, name, BOARD_TEXT_FORMAT, BOARD_TEXT_LEGAL_FORMAT)
                 for name in ("first_answer_legal", "blunder_rate")}
    inputs = verdict_inputs(primary)
    by_format = {fmt: rows for fmt, rows in df.groupby("prompt_format")} if "prompt_format" in df else {}
    pending = [name for name, result in primary.items()
               if not all(_is_finite(result[key]) for key in ("left_value", "right_value"))]
    cohort = cohort_coverage(df, expected_models, expected_positions)
    outcome = PENDING if cohort["complete"] is False else verdict(inputs)
    return json_safe({
        "seed": SEED, "resamples": RESAMPLES,
        "positions": int(_column(df, "position_id").nunique()),
        "models": sorted(_column(df, "model").dropna().unique()),
        "primary": primary, "verdict_inputs": inputs, "verdict": outcome, "cohort": cohort,
        "pending_measures": pending, "interpretation_warnings": INTERPRETATION_WARNINGS,
        "per_model": per_model_table(df).to_dict(orient="records"),
        "secondary": secondary,
        "wrong_side_share": {fmt: wrong_side_share(rows) for fmt, rows in by_format.items()},
        "dropped_pairs": dropped_pairs(df),
        "dropped_pairs_definition": "Absent records only; see each measure's coverage for missing or invalid values.",
    })


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and math.isfinite(value)


def json_safe(value: Any) -> Any:
    """Represent unavailable statistics as JSON null, including nested tables."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Plain-board notation check (specs/plain-board-prompts.md).")
    parser.add_argument("--config", default="config/config_plain_board.yaml")
    args = parser.parse_args()
    setup_logging()
    config = load_config(args.config)
    paths = config["paths"]
    df = load_results_df(paths["results_file"], paths.get("data_dir", "data"))
    out_dir = Path(paths["metrics_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    per_model_table(df).to_csv(out_dir / "plain_board_comparison.csv", index=False)
    summary = build_summary(df, expected_models=config.get("models"), expected_positions=expected_sample_size(config))
    summary["provenance"] = {**build_provenance(args.config, paths["results_file"]),
                             "ollama_digests": ollama_digests()}
    (out_dir / "plain_board_summary.json").write_text(json.dumps(summary, indent=2, default=str, allow_nan=False) + "\n")
    print(json.dumps({key: summary[key] for key in ("positions", "verdict_inputs", "verdict")}, indent=2))


if __name__ == "__main__":
    main()
