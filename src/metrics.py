"""Metrics aggregation and analysis for benchmark results."""

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pandas as pd

from .evaluator import match_t3_theme
from .result_writer import load_results
from .utils import parse_model_info
from .metric_audit import (direction_baselines, direction_diagnostics, regime_split,
                           cpl_diagnostics, analysis_provenance, publication_columns, json_safe, MOVE_FORMATS)

logger = logging.getLogger("chess_llm_bench")


# Floor (in centipawns) for the relative-error denominator. ~1 pawn — small
# enough that near-zero positions are penalised meaningfully but large enough
# to avoid divide-by-zero blow-up.
RELATIVE_ERROR_FLOOR_CP = 100

# Centipawn clamp for CPL — Lichess analysis convention. See
# evaluator.EVAL_CLAMP_CP for rationale.
EVAL_CLAMP_CP = 1000

# Threshold above which a Stockfish eval is treated as a mate-encoded score
# and excluded from absolute-error aggregations. Engine_wrapper encodes
# mate-in-N as ±10000 - mate_in*10, so anything ≥ 9000 is unambiguously a
# mate score. Mate-truth rows inflate t1_absolute_error by ~9966 each because
# the model's eval output is itself clamped to ±2000.
MATE_SCORE_THRESHOLD_CP = 9000

# Direction-accuracy thresholds (in centipawns). The original headline used
# ±50 only; that turns out to be the worst-case threshold because it forces a
# White/Black/Equal call right at the boundary where humans and engines also
# disagree. Reporting at multiple thresholds gives a much fuller picture:
# at ±100 ("one pawn = decisive") and ±200 ("clearly winning"), models agree
# with Stockfish much more often.
DIRECTION_THRESHOLDS_CP = (0, 50, 100, 200)


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


def _white_to_move_from_fen(fen: str) -> bool | None:
    """Return True if it's White to move in the given FEN. None on parse error."""
    try:
        return chess.Board(fen).turn == chess.WHITE
    except Exception:
        return None


def _win_probability(cp: float) -> float:
    """Sigmoid mapping centipawn eval → expected score. Standard k=400."""
    return 1.0 / (1.0 + math.exp(-cp / 400.0))


def compute_clamped_cpl(
    stockfish_eval: float,
    eval_after: float,
    white_to_move: bool,
    clamp_cp: int = EVAL_CLAMP_CP,
) -> float:
    """CPL with both endpoints clamped to ±clamp_cp.

    Lichess convention. See evaluator.EVAL_CLAMP_CP for rationale.
    """
    sf_c = max(-clamp_cp, min(clamp_cp, stockfish_eval))
    ea_c = max(-clamp_cp, min(clamp_cp, eval_after))
    cpl = (sf_c - ea_c) if white_to_move else (ea_c - sf_c)
    return max(0.0, cpl)


def compute_wp_loss(
    stockfish_eval: float,
    eval_after: float,
    white_to_move: bool,
    clamp_cp: int = EVAL_CLAMP_CP,
) -> float:
    """Win-probability loss × 1000 (milli-WP).

    Bounded to [0, 1000]. A value of 300 means the move dropped the side's
    win probability by 0.30. Less prone to mate-encoding artefacts than CPL
    because the sigmoid saturates outside the clamp window.
    """
    sf_c = max(-clamp_cp, min(clamp_cp, stockfish_eval))
    ea_c = max(-clamp_cp, min(clamp_cp, eval_after))
    wp_before = _win_probability(sf_c) if white_to_move else 1.0 - _win_probability(sf_c)
    wp_after = _win_probability(ea_c) if white_to_move else 1.0 - _win_probability(ea_c)
    return max(0.0, (wp_before - wp_after) * 1000.0)


def _verified_best_move_zero(record: dict[str, Any]) -> bool:
    """Validate the versioned reference-best-move scoring convention."""
    budget = record.get("t2_reference_budget")
    nodes = budget.get("nodes") if isinstance(budget, dict) else None
    move = record.get("t2_move")
    return (
        record.get("t2_legal") is True
        and record.get("t2_best_move_zero_convention") is True
        and record.get("t2_scoring_version") == "lc0_endpoints_v1_best_move_zero"
        and record.get("t2_reference_engine") == "Lc0Engine"
        and isinstance(nodes, int) and not isinstance(nodes, bool) and nodes > 0
        and record.get("t2_reference_units") == "centipawns_white"
        and isinstance(move, str) and bool(move)
        and move == record.get("t2_best_move")
    )


def derive_endpoint_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """Score recorded endpoint pairs or an explicitly versioned best-move zero.

    Historical t2_cpl may already have been clamped or use a different reference
    from T1. It cannot identify either original endpoint.
    """
    empty = {"t2_cpl_clamped": None, "t2_wp_loss": None, "t2_endpoint_status": "unavailable"}
    if _verified_best_move_zero(record):
        return {"t2_cpl_clamped": 0.0, "t2_wp_loss": 0.0,
                "t2_endpoint_status": "best_move_zero_convention"}
    before = record.get("t2_reference_eval_before")
    after = record.get("t2_reference_eval_after")
    if before is None or after is None or pd.isna(before) or pd.isna(after):
        return empty
    turn = record.get("t2_reference_white_to_move")
    if not isinstance(turn, (bool, np.bool_)):
        fen = record.get("fen")
        turn = _white_to_move_from_fen(fen) if isinstance(fen, str) else None
    if turn is None:
        return empty
    try:
        before, after = float(before), float(after)
    except (TypeError, ValueError):
        return {**empty, "t2_endpoint_status": "invalid"}
    if not math.isfinite(before) or not math.isfinite(after):
        return {**empty, "t2_endpoint_status": "invalid"}
    return {"t2_cpl_clamped": compute_clamped_cpl(before, after, bool(turn)),
            "t2_wp_loss": compute_wp_loss(before, after, bool(turn)),
            "t2_endpoint_status": "recorded"}


def rescore_t3_theme(theme: str | None, explanation: str | None) -> int | None:
    """Rescore stored explanation text using the shared versioned keyword matcher.

    This is lexical theme presence, not a semantic explanation-quality grade.
    Raw responses and existing stored legacy scores remain unchanged.
    """
    return match_t3_theme(theme, explanation)



def load_results_df(
    results_file: str = "results/evaluations.jsonl",
    data_dir: str = "data",
) -> pd.DataFrame:
    """Load results into a pandas DataFrame with derived metric columns.

    Args:
        results_file: Path to JSONL results file
        data_dir: Directory containing position JSON files (for FEN lookup —
            needed to recover side-to-move when reconstructing eval_after)

    Returns:
        DataFrame with all results plus derived columns:
            t1_relative_error      magnitude-invariant T1 error
            t1_abs_error_excl_mate t1_absolute_error with mate-truth rows NaN'd
            t2_cpl_clamped         CPL recomputed with ±EVAL_CLAMP_CP clamp
            t2_wp_loss             Δ win-probability × 1000 (bounded [0,1000])
            t3_p2_theme_correct_v2 theme-correctness rescored with new matcher
    """
    results = load_results(results_file)
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    _mask_prompt_applicability(df)

    # ----- T1 derived columns ---------------------------------------------
    if "t1_model_eval" in df.columns and "t1_stockfish_eval" in df.columns:
        valid = df["t1_model_eval"].notna() & df["t1_stockfish_eval"].notna()
        df["t1_relative_error"] = np.where(
            valid,
            np.abs(df["t1_model_eval"] - df["t1_stockfish_eval"])
            / np.maximum(np.abs(df["t1_stockfish_eval"]), RELATIVE_ERROR_FLOOR_CP),
            np.nan,
        )
        # T1 absolute error with mate-truth rows excluded. Mate-truth rows
        # contribute ~9966 cp of "error" automatically because the model's
        # output is itself clamped to ±2000 and the truth can be ±16000.
        if "t1_absolute_error" in df.columns:
            mate_truth = (
                df["t1_stockfish_eval"].notna()
                & (df["t1_stockfish_eval"].abs() >= MATE_SCORE_THRESHOLD_CP)
            )
            df["t1_abs_error_excl_mate"] = np.where(
                mate_truth, np.nan, df["t1_absolute_error"]
            )

    # Historical CPL and a separately computed T1 score do not identify the
    # original reference-engine endpoints. Only use recorded matched endpoints.
    if "t2_cpl" in df or "t2_reference_eval_before" in df:
        endpoint_rows = [derive_endpoint_metrics(record) for record in results]
        for column in ("t2_cpl_clamped", "t2_wp_loss", "t2_endpoint_status"):
            df[column] = [record[column] for record in endpoint_rows]
        for column in ("t2_cpl_clamped", "t2_wp_loss"):
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # ----- T3 rescored theme-correctness with the new matcher -------------
    if "theme" in df.columns and "t3_explanation" in df.columns:
        df["t3_p2_theme_correct_v2"] = [
            rescore_t3_theme(t, e)
            for t, e in zip(df["theme"], df["t3_explanation"])
        ]
        # Combined T3 score under the new theme matcher (P1 unchanged).
        if "t3_p1_side_correct" in df.columns:
            df["t3_score_v2"] = (
                pd.to_numeric(df["t3_p1_side_correct"], errors="coerce")
                + pd.to_numeric(df["t3_p2_theme_correct_v2"], errors="coerce")
            )

    # ----- T2 legal-move rate, computed only on move-asking attempts -----
    # `score_t2` returns t2_legal=False when t2_move is None, conflating
    # "model produced an illegal move" with "this prompt didn't ask for a
    # move at all" (eval_only and explanation_only prompts). The corrected
    # column is True/False only when the model attempted a move; NaN when
    # no move was attempted, so pandas .mean() automatically skips it.
    if "t2_legal" in df.columns and "t2_move" in df.columns:
        attempted = df["t2_move"].notna()
        df["t2_legal_attempted"] = np.where(attempted, df["t2_legal"], np.nan)

    if "t2_legal" in df and "prompt_format" in df:
        df["t2_legal_move_prompt"] = pd.to_numeric(df.t2_legal, errors="coerce").where(df.prompt_format.isin(MOVE_FORMATS))

    # ----- T1 direction accuracy at multiple thresholds -----
    # The headline `t1_direction_correct` field uses ±50 cp, which empirically
    # is the worst-case threshold for this dataset (forces the call right at
    # the boundary where everyone disagrees). Reporting at 0/50/100/200 cp
    # gives a more honest picture of model evaluation skill.
    if "t1_model_eval" in df.columns and "t1_stockfish_eval" in df.columns:
        for thresh in DIRECTION_THRESHOLDS_CP:
            df[f"t1_direction_correct_t{thresh}"] = _compute_direction_correct(
                df["t1_model_eval"], df["t1_stockfish_eval"], thresh
            )

    return df


def _mask_prompt_applicability(frame: pd.DataFrame) -> None:
    """Ignore unsolicited task fragments in prompts that do not request a task."""
    if "prompt_format" not in frame:
        return
    requested = {
        "t1_": ("cot", "fen_only", "pgn+fen", "eval_only"),
        "t3_": ("cot", "fen_only", "pgn+fen", "explanation_only"),
    }
    for prefix, formats in requested.items():
        not_requested = ~frame.prompt_format.isin(formats)
        for column in frame:
            if column.startswith(prefix) and column != "t1_stockfish_eval":
                frame[column] = frame[column].astype(object)
                frame.loc[not_requested, column] = np.nan
    for column in ("t1_model_eval", "t1_absolute_error", "t1_stockfish_eval"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _compute_direction_correct(
    model_eval: pd.Series, stockfish_eval: pd.Series, threshold: int
) -> pd.Series:
    """Vectorised direction-correctness at a given centipawn threshold.

    Returns True/False where both values are present, NaN otherwise.
    """
    valid = model_eval.notna() & stockfish_eval.notna()
    me_dir = np.where(model_eval > threshold, "W",
                       np.where(model_eval < -threshold, "B", "E"))
    sf_dir = np.where(stockfish_eval > threshold, "W",
                       np.where(stockfish_eval < -threshold, "B", "E"))
    correct = (me_dir == sf_dir).astype(object)
    return pd.Series(np.where(valid, correct, np.nan), index=model_eval.index)


def _build_fen_lookup(data_dir: str) -> dict[int, str]:
    """Map position_id → fen by reading data/{difficulty}.json files.

    Cached at module level after first call so repeated metrics computations
    don't re-read the JSON files.
    """
    global _FEN_LOOKUP_CACHE
    if _FEN_LOOKUP_CACHE is not None:
        return _FEN_LOOKUP_CACHE
    lookup: dict[int, str] = {}
    base = Path(data_dir)
    if not base.exists():
        logger.warning(f"data_dir {data_dir} not found; skipping FEN lookup")
        _FEN_LOOKUP_CACHE = lookup
        return lookup
    for tier in ("easy", "medium", "hard", "extreme"):
        path = base / f"{tier}.json"
        if not path.exists():
            continue
        try:
            with open(path) as f:
                positions = json.load(f)
            for p in positions:
                pid = p.get("id")
                fen = p.get("fen")
                if pid is not None and isinstance(fen, str):
                    lookup[pid] = fen
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
    _FEN_LOOKUP_CACHE = lookup
    return lookup


_FEN_LOOKUP_CACHE: dict[int, str] | None = None


def aggregate_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by model.

    Args:
        df: Results DataFrame

    Returns:
        Aggregated metrics per model
    """
    if df.empty:
        return pd.DataFrame()

    spec: dict[str, Any] = {
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
    }

    # Optional derived columns (added by load_results_df). Including them via
    # spec only when present keeps aggregate_by_model() callable from tests
    # that pass a hand-rolled DataFrame.
    optional_cols = {
        "t1_relative_error": "mean",
        "t1_abs_error_excl_mate": ["mean", "median"],
        "t2_cpl_clamped": ["mean", "median"],
        "t2_wp_loss": ["mean", "median"],
        "t2_legal_attempted": "mean",
        "t2_legal_move_prompt": "mean",
        "t3_p2_theme_correct_v2": "mean",
        "t3_score_v2": "mean",
    }
    for thresh in DIRECTION_THRESHOLDS_CP:
        col = f"t1_direction_correct_t{thresh}"
        if col in df.columns:
            optional_cols[col] = "mean"
    for col, ops in optional_cols.items():
        if col in df.columns:
            spec[col] = ops

    agg = df.groupby("model").agg(spec)

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

    spec: dict[str, Any] = {
        "t1_absolute_error": "mean",
        "t2_cpl": "mean",
        "t2_legal": "mean",
        "t3_score": "mean",
        "job_id": "count",
    }
    for col in (
        "t1_relative_error",
        "t1_abs_error_excl_mate",
        "t2_cpl_clamped",
        "t2_wp_loss",
        "t2_legal_attempted",
        "t2_legal_move_prompt",
        "t3_score_v2",
    ):
        if col in df.columns:
            spec[col] = "mean"
    return df.groupby(["model", "difficulty"]).agg(spec).reset_index()


def aggregate_by_phase(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by game phase.

    Args:
        df: Results DataFrame

    Returns:
        Aggregated metrics per phase
    """
    if df.empty:
        return pd.DataFrame()

    spec = {
        "t1_absolute_error": "mean",
        "t2_cpl": "mean",
        "t3_score": "mean",
        "job_id": "count",
    }
    for column in ("t1_abs_error_excl_mate", "t2_cpl_clamped", "t2_wp_loss", "t2_legal_move_prompt", "t3_score_v2"):
        if column in df:
            spec[column] = "mean"
    return df.groupby(["model", "phase"]).agg(spec).reset_index()


def aggregate_by_source(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by position source.

    Args:
        df: Results DataFrame

    Returns:
        Aggregated metrics per source
    """
    if df.empty:
        return pd.DataFrame()

    spec = {
        "t1_absolute_error": "mean",
        "t2_cpl": "mean",
        "t2_legal": "mean",
        "t3_score": "mean",
        "job_id": "count",
    }
    for column in ("t1_abs_error_excl_mate", "t2_cpl_clamped", "t2_wp_loss", "t2_legal_move_prompt", "t3_score_v2"):
        if column in df:
            spec[column] = "mean"
    return df.groupby(["model", "source"]).agg(spec).reset_index()


def aggregate_by_model_family(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by model family and size for scaling analysis.

    Args:
        df: Results DataFrame

    Returns:
        Aggregated metrics per model family and size
    """
    if df.empty:
        return pd.DataFrame()

    enriched = df.copy()
    metadata = {tag: parse_model_info(tag) for tag in enriched.model.unique()}
    enriched["model_family"] = enriched.model.map(lambda tag: metadata[tag]["family"])
    enriched["model_size_b"] = enriched.model.map(lambda tag: metadata[tag]["size_b"])
    spec = {column: "mean" for column in (
        "t1_absolute_error", "t2_cpl", "t2_legal", "t3_score",
        "t1_abs_error_excl_mate", "t2_cpl_clamped", "t2_wp_loss",
        "t2_legal_attempted", "t2_legal_move_prompt", "t3_score_v2") if column in enriched}
    spec["job_id"] = "count"
    aggregated = enriched.groupby(["model", "model_family", "model_size_b"]).agg(spec).reset_index()
    if len(aggregated) != enriched.model.nunique():
        raise ValueError("Family aggregation lost model identity")
    aggregated["scaling_eligible"] = aggregated.model.map(lambda tag: metadata[tag]["scaling_eligible"])
    return aggregated



def calculate_hallucination_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate illegal move (hallucination) rate by model and difficulty.

    Args:
        df: Results DataFrame

    Returns:
        Hallucination rates
    """
    if df.empty:
        return pd.DataFrame()

    failures = df.copy()
    if "t2_legal_move_prompt" in failures:
        failures["move_prompt_failure_rate"] = 1 - failures.t2_legal_move_prompt
        return failures.groupby(["model", "difficulty"]).agg(
            move_prompt_failure_rate=("move_prompt_failure_rate", "mean"),
            move_prompt_n=("move_prompt_failure_rate", "count")).reset_index()
    # Backwards compatibility for hand-built legacy frames. Never interpret
    # this unfiltered denominator as model hallucination in exported reports.
    failures["hallucination"] = ~failures["t2_legal"].fillna(False)
    return failures.groupby(["model", "difficulty"]).agg({
        "hallucination": "mean", "job_id": "count",
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

    # H2: T2 CPL increases with difficulty.
    #
    # Same artefact pattern as H1 — raw CPL is dominated by mate-encoding
    # leakage on easy puzzles (which are mostly mate-in-N tactics). Reported
    # under three metrics with t2_cpl_clamped as the primary.
    def _h2_metric(column: str) -> dict[str, Any]:
        if column not in df:
            return dict(supported=None, values={}, expect="increasing", missing=True,
                        available_n=0, publication_status="insufficient_data")
        numeric = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        means = numeric.groupby(df.difficulty).mean()
        values = {difficulty: None if pd.isna(means.get(difficulty)) else float(means[difficulty])
                  for difficulty in difficulty_order}
        complete = all(value is not None for value in values.values())
        return dict(supported=_trend_supported(list(values.values()), "increasing") if complete else None,
                    available_n=int(numeric.notna().sum()), values=values, expect="increasing",
                    publication_status="available" if complete else "insufficient_data")

    h2_metrics = {"absolute_cpl": _h2_metric("t2_cpl"),
                  "clamped_cpl": _h2_metric("t2_cpl_clamped"),
                  "wp_loss": _h2_metric("t2_wp_loss")}
    primary_h2 = h2_metrics["clamped_cpl"]
    results["H2"] = dict(description="T2 CPL increases with difficulty",
                         primary_metric="clamped_cpl", primary_supported=primary_h2["supported"],
                         metrics=h2_metrics, supported=primary_h2["supported"],
                         publication_status=primary_h2["publication_status"], values=primary_h2["values"])

    # Corrected metrics are primary; legacy values remain explicitly identified.
    h3_column = "t3_score_v2" if "t3_score_v2" in df else "t3_score"
    h3_metrics = {column: _h1_metric(column, "decreasing")
                  for column in ("t3_score", "t3_score_v2") if column in df}
    results["H3"] = dict(description="T3 score decreases with difficulty",
                         primary_metric=h3_column, metrics=h3_metrics,
                         supported=h3_metrics[h3_column]["supported"],
                         values=h3_metrics[h3_column]["values"])
    h4_columns = ["t1_abs_error_excl_mate", "t2_cpl_clamped", "t3_score_v2"]
    # Metadata is re-parsed instead of trusting legacy stored family/size values.
    scaling = df.copy()
    metadata = {tag: parse_model_info(tag) for tag in scaling.model.unique()}
    scaling = scaling.loc[scaling.model.map(lambda tag: metadata[tag]["scaling_eligible"])]
    scaling["model_family"] = scaling.model.map(lambda tag: metadata[tag]["family"])
    scaling["model_size_b"] = scaling.model.map(lambda tag: metadata[tag]["size_b"])
    for column in h4_columns:
        scaling[column] = pd.to_numeric(scaling[column], errors="coerce").replace([np.inf, -np.inf], np.nan) if column in scaling else np.nan
    h4_results = {}
    for family, family_frame in scaling.groupby("model_family"):
        by_size = family_frame.groupby("model_size_b")[h4_columns].mean()
        if len(by_size) < 2:
            continue
        smaller, larger = min(by_size.index), max(by_size.index)
        counts = family_frame.groupby("model_size_b")[h4_columns].count()
        available = {column: {"smaller": int(counts.loc[smaller, column]),
                              "larger": int(counts.loc[larger, column])} for column in h4_columns}
        comparisons = {}
        for column in h4_columns:
            if not all(available[column].values()):
                comparisons[column] = None
            elif column == "t3_score_v2":
                comparisons[column] = bool(by_size.loc[larger, column] > by_size.loc[smaller, column])
            else:
                comparisons[column] = bool(by_size.loc[larger, column] < by_size.loc[smaller, column])
        complete = all(value is not None for value in comparisons.values())
        h4_results[family] = dict(metrics=comparisons, available_n=available,
                                  all_better=all(comparisons.values()) if complete else None,
                                  publication_status="available" if complete else "insufficient_data")
    complete_h4 = bool(h4_results) and all(row["publication_status"] == "available" for row in h4_results.values())
    results["H4"] = dict(description="Larger dense models perform better within family",
                         primary_metric="t2_cpl_clamped", metrics=h4_columns,
                         supported=all(row["all_better"] for row in h4_results.values()) if complete_h4 else None,
                         publication_status="available" if complete_h4 else "insufficient_data",
                         by_family=h4_results,
                         excluded_models=[tag for tag, metadata_row in metadata.items() if not metadata_row["scaling_eligible"]])

    # H5: Better T3 than T2 on openings, worse on endgames (relative performance).
    #
    # We measure the per-phase gap between normalised T3 and T2 performance.
    # T3 score is on [0, 2] (P1 + P2); we normalise to [0, 1] by dividing by 2.
    # T2 legality (`t2_legal_attempted` if present, else `t2_legal`) is on
    # [0, 1] already. The hypothesis predicts:
    #   gap(opening) > gap(middlegame) > gap(endgame)
    # i.e. the T3-vs-T2 advantage shrinks (or reverses) toward the endgame.
    phase_order = ["opening", "middlegame", "endgame"]
    t2_col = "t2_legal_move_prompt" if "t2_legal_move_prompt" in df else (
        "t2_legal_attempted" if "t2_legal_attempted" in df.columns else "t2_legal"
    )
    t3_col = "t3_score_v2" if "t3_score_v2" in df.columns else "t3_score"
    phase_metrics = df.groupby("phase").agg({t2_col: "mean", t3_col: "mean"})
    phase_values = {}
    gaps = {}
    for ph in phase_order:
        if ph in phase_metrics.index:
            t2_norm = float(phase_metrics.loc[ph, t2_col])
            # t3 normalised to [0, 1] (max possible v2 score is 2)
            t3_norm = float(phase_metrics.loc[ph, t3_col]) / 2.0
            gap = t3_norm - t2_norm
            phase_values[ph] = {
                "t2_legal_normalised": t2_norm,
                "t3_score_normalised": t3_norm,
                "gap_t3_minus_t2": gap,
            }
            gaps[ph] = gap

    ordered_gaps = [gaps[ph] for ph in phase_order if ph in gaps]
    h5_supported = (
        len(ordered_gaps) == 3
        and ordered_gaps[0] > ordered_gaps[1]
        and ordered_gaps[1] > ordered_gaps[2]
    )
    results["H5"] = {
        "description": "Exploratory material-phase contrast; historical legality procedure is uncertain",
        "publication_status": "descriptive_only",
        "supported": h5_supported,
        "phase_values": phase_values,
        "gap_order": ordered_gaps,
        "metrics_used": {
            "t2": t2_col + " (normalised to [0,1])",
            "t3": t3_col + " (divided by 2 to normalise to [0,1])",
        },
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
            "t1_mean_error_excl_mate": (
                df["t1_abs_error_excl_mate"].mean()
                if "t1_abs_error_excl_mate" in df.columns else None
            ),
            "t1_relative_error": (
                df["t1_relative_error"].mean()
                if "t1_relative_error" in df.columns else None
            ),
            "t1_direction_accuracy_t50": (df["t1_direction_correct_t50"].mean()
                if "t1_direction_correct_t50" in df else None),
            "t1_direction_accuracy_by_threshold": {
                f"t{thresh}": (
                    df[f"t1_direction_correct_t{thresh}"].mean()
                    if f"t1_direction_correct_t{thresh}" in df.columns
                    else None
                )
                for thresh in DIRECTION_THRESHOLDS_CP
            },
            "t2_mean_cpl_raw": df["t2_cpl"].mean(),
            "t2_mean_cpl_clamped": (
                df["t2_cpl_clamped"].mean()
                if "t2_cpl_clamped" in df.columns else None
            ),
            "t2_mean_wp_loss": (
                df["t2_wp_loss"].mean()
                if "t2_wp_loss" in df.columns else None
            ),
            "t3_mean_score": df["t3_score"].mean(),
            "t3_mean_score_v2": (
                df["t3_score_v2"].mean()
                if "t3_score_v2" in df.columns else None
            ),
        },

        "by_model": publication_columns(aggregate_by_model(df)).to_dict(orient="records"),
        "by_difficulty": publication_columns(aggregate_by_difficulty(df)).to_dict(orient="records"),
        "by_phase": publication_columns(aggregate_by_phase(df)).to_dict(orient="records"),
        "by_source": publication_columns(aggregate_by_source(df)).to_dict(orient="records"),
        "hypothesis_tests": compute_hypothesis_tests(df),
    }

    summary["quality_status"] = {"primary_metric": "blunder_rate",
        "quality_records": int(df["t2_cpl_clamped"].notna().sum()) if "t2_cpl_clamped" in df else 0,
        "endpoint_records": int(df["t2_endpoint_status"].eq("recorded").sum()) if "t2_endpoint_status" in df else 0,
        "convention_zero_records": int(df["t2_endpoint_status"].eq("best_move_zero_convention").sum()) if "t2_endpoint_status" in df else 0,
        "historical_note": "Historical CPL does not identify original reference endpoints; corrected quality requires recorded endpoints or a verified explicit best-move-zero convention"}
    summary["legality_by_regime"] = regime_split(df).to_dict(orient="records")
    summary["provenance"] = analysis_provenance()
    return summary


def save_metrics(
    df: pd.DataFrame,
    output_dir: str = "results/metrics",
    *, collection_provenance: dict[str, Any] | None = None,
    config_snapshot: dict[str, Any] | None = None,
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
        for filename in ("by_model.csv", "by_difficulty.csv", "by_phase.csv", "by_source.csv",
                         "by_model_family.csv", "hallucination_rate.csv", "direction_baselines.csv",
                         "direction_diagnostics.csv", "regime_split.csv", "cpl_quality.csv",
                         "cpl_buckets.csv", "learning_deltas.csv"):
            (output_path / filename).unlink(missing_ok=True)
        (output_path / "summary.json").write_text(json.dumps({"total_jobs": 0, "error": "No results available"}))
        return

    tables = {
        "by_model.csv": aggregate_by_model(df),
        "by_difficulty.csv": aggregate_by_difficulty(df),
        "by_phase.csv": aggregate_by_phase(df),
        "by_source.csv": aggregate_by_source(df),
        "by_model_family.csv": aggregate_by_model_family(df),
        "hallucination_rate.csv": calculate_hallucination_rate(df),
        "direction_baselines.csv": direction_baselines(df),
        "direction_diagnostics.csv": direction_diagnostics(df),
        "regime_split.csv": regime_split(df),
    }
    quality, buckets = cpl_diagnostics(df)
    tables.update({"cpl_quality.csv": quality, "cpl_buckets.csv": buckets})
    for filename, table in tables.items():
        publication_columns(table).to_csv(output_path / filename, index=False)
    learning_df = calculate_learning_deltas(df)
    if learning_df.empty:
        (output_path / "learning_deltas.csv").unlink(missing_ok=True)
    else:
        learning_df.to_csv(output_path / "learning_deltas.csv", index=False)
    summary = generate_summary(df)
    summary["provenance"] = analysis_provenance(collection_provenance)
    summary["provenance"]["analysis_config_snapshot"] = config_snapshot or {}
    with open(output_path / "summary.json", "w") as stream:
        json.dump(json_safe(summary), stream, indent=2, allow_nan=False)

    logger.info(f"Saved metrics to {output_path}")
