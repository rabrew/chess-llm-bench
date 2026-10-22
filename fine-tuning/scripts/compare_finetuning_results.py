"""Compare base model vs fine-tuned model benchmark results.

Pairs first-attempt (job_type == "standard") records on
(base model, position_id, prompt_format) and reports, per model pair, each
side's metrics and the fine-tuned − base delta:

  T1  Spearman ρ, Cohen's κ at ±50 cp, MAE and MAE relative to always answering 0
  T2  legal-move rate over prompts that ask for a move, clamped CPL
  T3  mean score

Error alone is not evidence of skill: most positions are near level, so a model
that answers small numbers scores a low MAE without reading the board. Read
t1_mae next to t1_mae_ratio and t1_spearman (specs/t1-zero-baseline-and-ft-comparison.md).

Board-reading accuracy (T0) is measured by board_reading_test.py, not here.

Output: results/finetuning_experiment/comparison.csv

Usage:
    python fine-tuning/scripts/compare_finetuning_results.py
    python fine-tuning/scripts/compare_finetuning_results.py --base results/evaluations.jsonl
"""

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.metrics import (  # noqa: E402
    EVAL_CLAMP_CP,
    MATE_SCORE_THRESHOLD_CP,
    compute_direction_agreement,
    compute_eval_error_vs_zero,
    compute_eval_rank_correlation,
    load_results_df,
)
from src.utils import load_config, setup_logging  # noqa: E402

logger = logging.getLogger("chess_llm_bench")

BASE_CONFIG = "config/config.yaml"
FT_CONFIG = "config/config_finetuned.yaml"
OUTPUT_CSV = "results/finetuning_experiment/comparison.csv"
DATA_DIR = str(REPO_ROOT / "data")

FT_SUFFIX = "-board-ft"
STANDARD_JOB = "standard"
KEY_COLS = ["position_id", "prompt_format"]
MOVE_FORMATS = ("fen_only", "pgn+fen", "cot", "move_only")
KAPPA_THRESHOLD_CP = 50
CPL_CLAMP_CP = EVAL_CLAMP_CP

METRICS = (
    "t1_spearman",
    "t1_kappa_t50",
    "t1_mae",
    "t1_mae_ratio",
    "t2_legal_rate",
    "t2_mean_cpl_clamped",
    "t3_mean",
)


def resolve_results_path(explicit: str | None, config_path: str) -> str:
    """Use the explicit path if given, else paths.results_file from the config."""
    if explicit:
        return explicit
    path = load_config(config_path).get("paths", {}).get("results_file")
    if not path:
        raise ValueError(f"{config_path} has no paths.results_file")
    return path


def load_records(path: str) -> pd.DataFrame:
    """Load results with the standard derived columns, first-attempt jobs only."""
    df = load_results_df(path, data_dir=DATA_DIR)
    if df.empty:
        return df
    job_type = df["job_type"] if "job_type" in df.columns else STANDARD_JOB
    return df[pd.Series(job_type, index=df.index).fillna(STANDARD_JOB) == STANDARD_JOB]


def _dedupe(rows: pd.DataFrame, model: str) -> pd.DataFrame:
    duplicated = rows.duplicated(subset=KEY_COLS, keep="first")
    if duplicated.any():
        logger.warning(f"{model}: {int(duplicated.sum())} duplicate (position, format) records; keeping the first")
    return rows[~duplicated]


def _pair(base_df: pd.DataFrame, ft_df: pd.DataFrame, ft_model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_model = ft_model.removesuffix(FT_SUFFIX)
    base_rows = _dedupe(base_df[base_df["model"] == base_model], base_model)
    ft_rows = _dedupe(ft_df[ft_df["model"] == ft_model], ft_model)
    shared = base_rows[KEY_COLS].merge(ft_rows[KEY_COLS], on=KEY_COLS)

    def keep_shared(rows: pd.DataFrame) -> pd.DataFrame:
        return rows.merge(shared, on=KEY_COLS).sort_values(KEY_COLS).reset_index(drop=True)

    return keep_shared(base_rows), keep_shared(ft_rows)


def _first_value(table: pd.DataFrame, column: str) -> float:
    return float(table[column].iloc[0]) if not table.empty else math.nan


def _mean(values: pd.Series) -> float:
    return float(values.mean()) if values.notna().any() else math.nan


def side_metrics(rows: pd.DataFrame) -> dict[str, float]:
    """All comparison metrics for one model's paired rows."""
    agreement = compute_direction_agreement(rows)
    if not agreement.empty:
        agreement = agreement[agreement["threshold"] == KAPPA_THRESHOLD_CP]
    error = compute_eval_error_vs_zero(rows)
    move_rows = rows[rows["prompt_format"].isin(MOVE_FORMATS)]
    return {
        "t1_spearman": _first_value(compute_eval_rank_correlation(rows), "spearman_rho"),
        "t1_kappa_t50": _first_value(agreement, "cohen_kappa"),
        "t1_mae": _first_value(error, "mae"),
        "t1_mae_ratio": _first_value(error, "mae_ratio"),
        "t2_legal_rate": _mean(move_rows["t2_legal"].fillna(False).astype(float)),
        "t2_mean_cpl_clamped": _mean(pd.to_numeric(rows["t2_cpl_clamped"], errors="coerce")),
        "t3_mean": _mean(pd.to_numeric(rows["t3_score"], errors="coerce")),
    }


def shared_zero_mae(base_rows: pd.DataFrame, ft_rows: pd.DataFrame) -> float:
    """MAE of always answering 0, on non-mate rows where both sides gave an eval."""
    both = base_rows[KEY_COLS + ["t1_model_eval", "t1_stockfish_eval"]].merge(
        ft_rows[KEY_COLS + ["t1_model_eval"]], on=KEY_COLS, suffixes=("_base", "_ft"),
    )
    truth = both["t1_stockfish_eval"].astype(float)
    usable = both["t1_model_eval_base"].notna() & both["t1_model_eval_ft"].notna() & (truth.abs() < MATE_SCORE_THRESHOLD_CP)
    return _mean(truth[usable].abs())


def compare(base_df: pd.DataFrame, ft_df: pd.DataFrame) -> pd.DataFrame:
    """One row per fine-tuned model with a base partner: base_, ft_ and delta_ per metric."""
    if base_df.empty or ft_df.empty:
        return pd.DataFrame()
    rows = []
    for ft_model in sorted(ft_df["model"].unique()):
        base_rows, ft_rows = _pair(base_df, ft_df, ft_model)
        if ft_rows.empty:
            logger.warning(f"{ft_model}: no records pair with {ft_model.removesuffix(FT_SUFFIX)}; skipped")
            continue
        base_metrics, ft_metrics = side_metrics(base_rows), side_metrics(ft_rows)
        row: dict[str, Any] = {
            "base_model": ft_model.removesuffix(FT_SUFFIX),
            "ft_model": ft_model,
            "n_pairs": len(ft_rows),
            "zero_mae": shared_zero_mae(base_rows, ft_rows),
        }
        for metric in METRICS:
            row[f"base_{metric}"] = base_metrics[metric]
            row[f"ft_{metric}"] = ft_metrics[metric]
            row[f"delta_{metric}"] = ft_metrics[metric] - base_metrics[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt(value: float, spec: str) -> str:
    return "—" if math.isnan(value) else format(value, spec)


def print_summary(table: pd.DataFrame) -> None:
    logger.info("=== Fine-tuning experiment: base → fine-tuned (Δ) ===")
    logger.info(f"{'Model':<16} {'pairs':>6} {'T1 ρ':>16} {'T1 MAE/zero':>18} {'T2 legal':>18} {'T3 mean':>18}")
    for _, row in table.iterrows():
        cells = [
            f"{_fmt(row[f'base_{m}'], spec)}→{_fmt(row[f'ft_{m}'], spec)}"
            for m, spec in (("t1_spearman", ".2f"), ("t1_mae_ratio", ".2f"),
                            ("t2_legal_rate", ".1%"), ("t3_mean", ".2f"))
        ]
        logger.info(f"{row['base_model'][:16]:<16} {row['n_pairs']:>6} " + " ".join(f"{c:>18}" for c in cells))
    logger.info("T1 MAE/zero < 1 beats always answering 0 cp; ρ near 0 means evals ignore the position.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned benchmark results.")
    parser.add_argument("--base-config", default=BASE_CONFIG, help=f"Config naming the base results (default: {BASE_CONFIG})")
    parser.add_argument("--ft-config", default=FT_CONFIG, help=f"Config naming the fine-tuned results (default: {FT_CONFIG})")
    parser.add_argument("--base", help="Base results JSONL (overrides --base-config)")
    parser.add_argument("--ft", help="Fine-tuned results JSONL (overrides --ft-config)")
    parser.add_argument("--output", default=OUTPUT_CSV, help=f"Output CSV path (default: {OUTPUT_CSV})")
    args = parser.parse_args()

    setup_logging()

    try:
        base_path = resolve_results_path(args.base, args.base_config)
        ft_path = resolve_results_path(args.ft, args.ft_config)
    except (OSError, ValueError) as exc:
        logger.error(f"Cannot resolve results paths: {exc}")
        sys.exit(1)

    for label, path in (("Base", base_path), ("Fine-tuned", ft_path)):
        if not Path(path).exists():
            logger.error(f"{label} results not found: {path}")
            sys.exit(1)

    base_df, ft_df = load_records(base_path), load_records(ft_path)
    logger.info(f"Loaded {len(base_df)} base and {len(ft_df)} fine-tuned first-attempt records")

    table = compare(base_df, ft_df)
    if table.empty:
        logger.error(f"No model pairs found. Fine-tuned tags must be '<base>{FT_SUFFIX}'.")
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    logger.info(f"Comparison CSV written: {args.output}")
    print_summary(table)


if __name__ == "__main__":
    main()
