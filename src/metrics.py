"""Metrics aggregation and analysis for benchmark results."""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .result_writer import load_results

logger = logging.getLogger("chess_llm_bench")


# Floor (in centipawns) for the relative-error denominator. ~1 pawn — small
# enough that near-zero positions are penalised meaningfully but large enough
# to avoid divide-by-zero blow-up.
RELATIVE_ERROR_FLOOR_CP = 100


def compute_relative_error(
    model_eval: float, stockfish_eval: float, floor: int = RELATIVE_ERROR_FLOOR_CP
) -> float:
    """Position-magnitude-invariant alternative to absolute error.

    relative_error = |model - truth| / max(|truth|, floor)

    The floor prevents division blow-up on near-zero positions and makes a
    "100 cp miss on an equal position" register as 1.0 — comparable to a
    "1000 cp miss on a +1000 cp position."
    """
    if floor <= 0:
        raise ValueError(f"floor must be positive, got {floor}")
    return abs(model_eval - stockfish_eval) / max(abs(stockfish_eval), floor)


def load_results_df(results_file: str = "results/evaluations.jsonl") -> pd.DataFrame:
    """Load results into a pandas DataFrame.

    Args:
        results_file: Path to JSONL results file

    Returns:
        DataFrame with all results, including the derived
        ``t1_relative_error`` column for hypothesis-test calibration.
    """
    results = load_results(results_file)
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)

    # Derived: position-magnitude-invariant T1 error. NaN where either input
    # is missing (model failed to produce a parseable number).
    if "t1_model_eval" in df.columns and "t1_stockfish_eval" in df.columns:
        valid = df["t1_model_eval"].notna() & df["t1_stockfish_eval"].notna()
        df["t1_relative_error"] = np.where(
            valid,
            np.abs(df["t1_model_eval"] - df["t1_stockfish_eval"])
            / np.maximum(np.abs(df["t1_stockfish_eval"]), RELATIVE_ERROR_FLOOR_CP),
            np.nan,
        )
    return df


def aggregate_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by model.

    Args:
        df: Results DataFrame

    Returns:
        Aggregated metrics per model
    """
    if df.empty:
        return pd.DataFrame()

    agg = df.groupby("model").agg({
        # T1 metrics
        "t1_absolute_error": ["mean", "std", "median"],
        "t1_direction_correct": "mean",

        # T2 metrics
        "t2_legal": "mean",
        "t2_cpl": ["mean", "std", "median"],

        # T3 metrics
        "t3_p1_side_correct": "mean",
        "t3_p2_theme_correct": "mean",
        "t3_score": ["mean", "std"],

        # Counts
        "job_id": "count",
        "inference_ms": ["mean", "median"],
    })

    # Flatten column names
    agg.columns = ["_".join(col).strip("_") for col in agg.columns]

    return agg.reset_index()


def aggregate_by_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by difficulty tier.

    Args:
        df: Results DataFrame

    Returns:
        Aggregated metrics per difficulty
    """
    if df.empty:
        return pd.DataFrame()

    return df.groupby(["model", "difficulty"]).agg({
        "t1_absolute_error": "mean",
        "t2_cpl": "mean",
        "t2_legal": "mean",
        "t3_score": "mean",
        "job_id": "count",
    }).reset_index()


def aggregate_by_phase(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by game phase.

    Args:
        df: Results DataFrame

    Returns:
        Aggregated metrics per phase
    """
    if df.empty:
        return pd.DataFrame()

    return df.groupby(["model", "phase"]).agg({
        "t1_absolute_error": "mean",
        "t2_cpl": "mean",
        "t3_score": "mean",
        "job_id": "count",
    }).reset_index()


def aggregate_by_source(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by position source.

    Args:
        df: Results DataFrame

    Returns:
        Aggregated metrics per source
    """
    if df.empty:
        return pd.DataFrame()

    return df.groupby(["model", "source"]).agg({
        "t1_absolute_error": "mean",
        "t2_cpl": "mean",
        "t2_legal": "mean",
        "t3_score": "mean",
        "job_id": "count",
    }).reset_index()


def aggregate_by_model_family(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by model family and size for scaling analysis.

    Args:
        df: Results DataFrame

    Returns:
        Aggregated metrics per model family and size
    """
    if df.empty:
        return pd.DataFrame()

    return df.groupby(["model_family", "model_size_b"]).agg({
        "t1_absolute_error": "mean",
        "t2_cpl": "mean",
        "t2_legal": "mean",
        "t3_score": "mean",
        "job_id": "count",
    }).reset_index()


def calculate_hallucination_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate illegal move (hallucination) rate by model and difficulty.

    Args:
        df: Results DataFrame

    Returns:
        Hallucination rates
    """
    if df.empty:
        return pd.DataFrame()

    # Hallucination = not legal move
    df = df.copy()
    df["hallucination"] = ~df["t2_legal"].fillna(False)

    return df.groupby(["model", "difficulty"]).agg({
        "hallucination": "mean",
        "job_id": "count",
    }).reset_index().rename(columns={"hallucination": "hallucination_rate"})


def calculate_learning_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate learning deltas from correction loop results.

    Args:
        df: Results DataFrame

    Returns:
        Learning deltas per model
    """
    if df.empty:
        return pd.DataFrame()

    # Get correction and control results
    correction_df = df[df["job_type"] == "correction"].copy()
    control_df = df[df["job_type"] == "control"].copy()

    if correction_df.empty or control_df.empty:
        return pd.DataFrame()

    # Get parent job results
    standard_df = df[df["job_type"] == "standard"].copy()

    results = []

    for _, corr_row in correction_df.iterrows():
        parent_id = corr_row.get("parent_job_id")
        if not parent_id:
            continue

        # Find parent result
        parent_rows = standard_df[standard_df["job_id"] == parent_id]
        if parent_rows.empty:
            continue
        parent_row = parent_rows.iloc[0]

        # Find control result
        ctrl_rows = control_df[control_df["parent_job_id"] == parent_id]
        if ctrl_rows.empty:
            continue
        ctrl_row = ctrl_rows.iloc[0]

        # Calculate deltas
        cpl_1 = parent_row.get("t2_cpl")
        cpl_corr = corr_row.get("t2_cpl")
        cpl_ctrl = ctrl_row.get("t2_cpl")

        if pd.notna(cpl_1) and pd.notna(cpl_corr) and pd.notna(cpl_ctrl):
            delta_correction = cpl_1 - cpl_corr
            delta_control = cpl_1 - cpl_ctrl
            net_effect = delta_correction - delta_control

            results.append({
                "model": corr_row["model"],
                "parent_job_id": parent_id,
                "cpl_attempt_1": cpl_1,
                "cpl_correction": cpl_corr,
                "cpl_control": cpl_ctrl,
                "delta_correction": delta_correction,
                "delta_control": delta_control,
                "net_feedback_effect": net_effect,
            })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


def compute_hypothesis_tests(df: pd.DataFrame) -> dict[str, Any]:
    """Test the pre-registered hypotheses.

    Args:
        df: Results DataFrame

    Returns:
        Dictionary with hypothesis test results
    """
    results = {}

    if df.empty:
        return {"error": "No data available"}

    # H1: T1 error increases with difficulty.
    #
    # Reported under three metrics. Absolute error is kept for transparency
    # because it surfaces a well-known artefact (Lichess "easy" puzzles tend
    # to have decisive evals, inflating absolute errors when the model guesses
    # near zero). Relative error is the primary metric — invariant to position
    # magnitude. Direction accuracy is a sanity-check on the qualitative call.
    difficulty_order = ["easy", "medium", "hard", "extreme"]

    def _trend_supported(values: list[float], expect: str) -> bool:
        """Weakly monotonic in `expect` direction AND strict overall.

        A flat series ([1, 1, 1, 1]) is not evidence of an increase or
        decrease, so we require the endpoints to differ in the predicted
        direction in addition to weak monotonicity at each step.
        """
        cleaned = [v for v in values if v is not None and not pd.isna(v)]
        if len(cleaned) < 2:
            return False
        if expect == "increasing":
            weak = all(cleaned[i] <= cleaned[i + 1] for i in range(len(cleaned) - 1))
            return weak and cleaned[0] < cleaned[-1]
        weak = all(cleaned[i] >= cleaned[i + 1] for i in range(len(cleaned) - 1))
        return weak and cleaned[0] > cleaned[-1]

    def _h1_metric(column: str, expect: str) -> dict[str, Any]:
        data = df.groupby("difficulty")[column].mean()
        values_dict = {d: (None if d not in data.index else float(data.loc[d]))
                       for d in difficulty_order}
        ordered = [values_dict[d] for d in difficulty_order]
        return {
            "supported": _trend_supported(ordered, expect),
            "values": values_dict,
            "expect": expect,
        }

    h1_metrics = {
        "absolute_error": _h1_metric("t1_absolute_error", "increasing"),
    }
    if "t1_relative_error" in df.columns:
        h1_metrics["relative_error"] = _h1_metric("t1_relative_error", "increasing")
    if "t1_direction_correct" in df.columns:
        h1_metrics["direction_accuracy"] = _h1_metric("t1_direction_correct", "decreasing")

    primary_metric = "relative_error" if "relative_error" in h1_metrics else "absolute_error"
    primary = h1_metrics[primary_metric]
    results["H1"] = {
        "description": "T1 error increases with difficulty",
        "primary_metric": primary_metric,
        "primary_supported": primary["supported"],
        "metrics": h1_metrics,
        # Backwards-compat surface (existing dashboard/test consumers expect
        # `supported` and `values` at the top level). Mirrors the primary.
        "supported": primary["supported"],
        "values": primary["values"],
    }

    # H2: T2 CPL increases with difficulty
    h2_data = df.groupby("difficulty")["t2_cpl"].mean()
    h2_values = [h2_data.get(d) for d in difficulty_order if d in h2_data]
    h2_increasing = len(h2_values) >= 2 and all(
        h2_values[i] <= h2_values[i + 1]
        for i in range(len(h2_values) - 1)
        if h2_values[i] is not None and h2_values[i + 1] is not None
    )
    results["H2"] = {
        "description": "T2 CPL increases with difficulty",
        "supported": h2_increasing,
        "values": {d: h2_data.get(d) for d in difficulty_order},
    }

    # H3: T3 score decreases with difficulty
    h3_data = df.groupby("difficulty")["t3_score"].mean()
    h3_values = [h3_data.get(d) for d in difficulty_order if d in h3_data]
    h3_decreasing = len(h3_values) >= 2 and all(
        h3_values[i] >= h3_values[i + 1]
        for i in range(len(h3_values) - 1)
        if h3_values[i] is not None and h3_values[i + 1] is not None
    )
    results["H3"] = {
        "description": "T3 score decreases with difficulty",
        "supported": h3_decreasing,
        "values": {d: h3_data.get(d) for d in difficulty_order},
    }

    # H4: Larger models perform better (within family)
    h4_results = {}
    for family in df["model_family"].dropna().unique():
        family_df = df[df["model_family"] == family]
        if len(family_df["model_size_b"].dropna().unique()) < 2:
            continue

        size_metrics = family_df.groupby("model_size_b").agg({
            "t1_absolute_error": "mean",
            "t2_cpl": "mean",
            "t3_score": "mean",
        })

        # Check if larger size has lower error, lower CPL, higher T3
        sizes = sorted(size_metrics.index)
        if len(sizes) >= 2:
            smaller, larger = sizes[0], sizes[-1]
            better_t1 = (
                size_metrics.loc[larger, "t1_absolute_error"]
                < size_metrics.loc[smaller, "t1_absolute_error"]
            )
            better_t2 = (
                size_metrics.loc[larger, "t2_cpl"]
                < size_metrics.loc[smaller, "t2_cpl"]
            )
            better_t3 = (
                size_metrics.loc[larger, "t3_score"]
                > size_metrics.loc[smaller, "t3_score"]
            )
            h4_results[family] = {
                "better_t1": better_t1,
                "better_t2": better_t2,
                "better_t3": better_t3,
                "all_better": better_t1 and better_t2 and better_t3,
            }

    results["H4"] = {
        "description": "Larger models perform better within family",
        "supported": all(r.get("all_better", False) for r in h4_results.values()),
        "by_family": h4_results,
    }

    # H5: Better T3 than T2 on openings, worse on endgames (relative performance)
    # This requires normalizing scores - simplified version
    phase_data = df.groupby("phase").agg({
        "t2_legal": "mean",
        "t3_score": "mean",
    })

    results["H5"] = {
        "description": "Better T3 vs T2 on openings, worse on endgames",
        "phase_data": phase_data.to_dict() if not phase_data.empty else {},
        "note": "Requires detailed relative performance analysis",
    }

    return results


def generate_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Generate a complete summary of benchmark results.

    Args:
        df: Results DataFrame

    Returns:
        Summary dictionary
    """
    if df.empty:
        return {"error": "No results available"}

    summary = {
        "total_jobs": len(df),
        "models_tested": df["model"].nunique(),
        "positions_tested": df["position_id"].nunique(),

        "overall_metrics": {
            "t1_mean_error": df["t1_absolute_error"].mean(),
            "t1_direction_accuracy": df["t1_direction_correct"].mean(),
            "t2_legal_rate": df["t2_legal"].mean(),
            "t2_mean_cpl": df["t2_cpl"].mean(),
            "t3_mean_score": df["t3_score"].mean(),
        },

        "by_model": aggregate_by_model(df).to_dict(orient="records"),
        "by_difficulty": aggregate_by_difficulty(df).to_dict(orient="records"),
        "by_phase": aggregate_by_phase(df).to_dict(orient="records"),
        "by_source": aggregate_by_source(df).to_dict(orient="records"),
        "hypothesis_tests": compute_hypothesis_tests(df),
    }

    return summary


def save_metrics(
    df: pd.DataFrame,
    output_dir: str = "results/metrics",
) -> None:
    """Save all metric aggregations to files.

    Args:
        df: Results DataFrame
        output_dir: Output directory for metric files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if df.empty:
        logger.warning("No results to save metrics for")
        return

    # Save aggregations as CSV
    aggregate_by_model(df).to_csv(output_path / "by_model.csv", index=False)
    aggregate_by_difficulty(df).to_csv(output_path / "by_difficulty.csv", index=False)
    aggregate_by_phase(df).to_csv(output_path / "by_phase.csv", index=False)
    aggregate_by_source(df).to_csv(output_path / "by_source.csv", index=False)
    aggregate_by_model_family(df).to_csv(
        output_path / "by_model_family.csv", index=False
    )
    calculate_hallucination_rate(df).to_csv(
        output_path / "hallucination_rate.csv", index=False
    )

    # Save learning deltas if available
    learning_df = calculate_learning_deltas(df)
    if not learning_df.empty:
        learning_df.to_csv(output_path / "learning_deltas.csv", index=False)

    # Save summary as JSON
    summary = generate_summary(df)
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Saved metrics to {output_path}")
