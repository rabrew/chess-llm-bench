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

from .evaluator import theme_mentioned
from .result_writer import load_results
from .utils import parse_model_info

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

# Prompt formats that ask the model for a move (T2). eval_only and
# explanation_only do not.
MOVE_PROMPT_FORMATS = ("fen_only", "pgn+fen", "cot", "move_only")


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


def rescore_t3_theme(theme: str | None, explanation: str | None) -> int | None:
    """Recompute the T3 theme-correctness component using the new matcher.

    Postprocessing of the stored t3_explanation field — the underlying LLM
    output is unchanged. Keyed by exact Lichess theme label first, with
    camelCase-to-words fallback, matched as whole words.
    """
    if theme is None or explanation is None:
        return None
    if not isinstance(theme, str) or not isinstance(explanation, str):
        return None

    return 1 if theme_mentioned(theme, explanation) else 0


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

    # ----- Model family / size --------------------------------------------
    # Stored values came from the old guessing parser (D3); recompute them.
    if "model" in df.columns:
        info = {tag: parse_model_info(tag) for tag in df["model"].dropna().unique()}
        df["model_family"] = df["model"].map(lambda tag: info[tag]["family"] if tag in info else None)
        df["model_size_b"] = df["model"].map(lambda tag: info[tag]["size_b"] if tag in info else None)

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
        # output is itself clamped to ±2000 and the truth can be ±10000.
        if "t1_absolute_error" in df.columns:
            mate_truth = (
                df["t1_stockfish_eval"].notna()
                & (df["t1_stockfish_eval"].abs() >= MATE_SCORE_THRESHOLD_CP)
            )
            df["t1_abs_error_excl_mate"] = np.where(
                mate_truth, np.nan, df["t1_absolute_error"]
            )

    # ----- T2 derived columns: clamped CPL + win-probability loss ---------
    # Reconstruct eval_after from (t1_stockfish_eval, t2_cpl, fen):
    #   cpl = max(0, sf - eval_after)  if white_to_move else
    #         max(0, eval_after - sf)
    # so eval_after = sf - cpl  if white_to_move else  sf + cpl.
    if "t2_cpl" in df.columns and "t1_stockfish_eval" in df.columns:
        fen_lookup = _build_fen_lookup(data_dir)
        if "fen" in df.columns:
            fens = df["fen"]
        elif "position_id" in df.columns:
            fens = df["position_id"].map(fen_lookup)
        else:
            fens = pd.Series([None] * len(df))

        white_to_move = fens.map(
            lambda f: _white_to_move_from_fen(f) if isinstance(f, str) else None
        )

        def _derive(row_sf, row_cpl, row_wtm):
            if pd.isna(row_sf) or pd.isna(row_cpl) or row_wtm is None:
                return (np.nan, np.nan)
            sf = float(row_sf)
            cpl = float(row_cpl)
            ea = sf - cpl if row_wtm else sf + cpl
            return (
                compute_clamped_cpl(sf, ea, bool(row_wtm)),
                compute_wp_loss(sf, ea, bool(row_wtm)),
            )

        derived = [
            _derive(sf, cpl, wtm)
            for sf, cpl, wtm in zip(
                df["t1_stockfish_eval"], df["t2_cpl"], white_to_move
            )
        ]
        df["t2_cpl_clamped"] = [d[0] for d in derived]
        df["t2_wp_loss"] = [d[1] for d in derived]

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

    # ----- T2 legal-move rate, computed only on move-asking prompts -----
    # `score_t2` returns t2_legal=False for eval_only / explanation_only
    # prompts, which never ask for a move; those rows are NaN here so .mean()
    # skips them. The attempt is defined by the prompt, not by t2_move: the
    # worker nulls any move it could not make legal, so a null move on a
    # move-asking prompt IS an illegal move and must count as one.
    if "t2_legal" in df.columns and "t2_move" in df.columns:
        if "prompt_format" in df.columns:
            attempted = df["prompt_format"].isin(MOVE_PROMPT_FORMATS)
            legal = df["t2_legal"].astype("boolean").fillna(False).astype(float)
        else:
            logger.warning("No prompt_format column; t2_legal_attempted skips null moves (undercounts illegal moves)")
            attempted = df["t2_move"].notna()
            legal = df["t2_legal"]
        df["t2_legal_attempted"] = np.where(attempted, legal, np.nan)

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


def _compute_direction_correct(
    model_eval: pd.Series, stockfish_eval: pd.Series, threshold: int
) -> pd.Series:
    """Vectorised direction-correctness at a given centipawn threshold.

    Returns True/False where both values are present, NaN otherwise.
    """
    valid = model_eval.notna() & stockfish_eval.notna()
    me_dir = _direction_labels(model_eval, threshold)
    sf_dir = _direction_labels(stockfish_eval, threshold)
    correct = (me_dir == sf_dir).astype(object)
    return pd.Series(np.where(valid, correct, np.nan), index=model_eval.index)


# ---------------------------------------------------------------------------
# Chance-corrected direction metrics (specs/direction-baseline-kappa.md).
#
# Raw direction accuracy rewards a model that always answers 0 cp, because the
# "Equal" class grows with the threshold. These metrics compare against what
# the label frequencies alone would score.
# ---------------------------------------------------------------------------

DIRECTION_LABELS = ("W", "B", "E")
_T1_EVAL_COLS = ("t1_model_eval", "t1_stockfish_eval")


def _direction_labels(evals: pd.Series, threshold: int) -> np.ndarray:
    """Label each eval "W" (> threshold), "B" (< -threshold) or "E"."""
    return np.where(evals > threshold, "W", np.where(evals < -threshold, "B", "E"))


def _validate_label_pair(pred, truth) -> tuple[np.ndarray, np.ndarray]:
    pred_arr, truth_arr = np.asarray(pred), np.asarray(truth)
    if len(pred_arr) != len(truth_arr):
        raise ValueError(f"length mismatch: {len(pred_arr)} predictions vs {len(truth_arr)} truths")
    if len(pred_arr) == 0:
        raise ValueError("cannot score an empty set of labels")
    return pred_arr, truth_arr


def cohen_kappa(pred, truth) -> float:
    """Cohen's κ: agreement beyond what the label frequencies alone produce.

    Returns NaN when expected agreement is 1 (both sides are the same single
    constant label), where κ is mathematically undefined.
    """
    pred_arr, truth_arr = _validate_label_pair(pred, truth)
    observed = float(np.mean(pred_arr == truth_arr))
    expected = sum(
        float(np.mean(pred_arr == label)) * float(np.mean(truth_arr == label))
        for label in set(pred_arr) | set(truth_arr)
    )
    if math.isclose(expected, 1.0):
        return float("nan")
    return (observed - expected) / (1.0 - expected)


def balanced_accuracy(pred, truth) -> float:
    """Mean per-class recall over the classes present in ``truth``."""
    pred_arr, truth_arr = _validate_label_pair(pred, truth)
    recalls = [
        float(np.mean(pred_arr[truth_arr == label] == label))
        for label in np.unique(truth_arr)
    ]
    return float(np.mean(recalls))


def _valid_t1_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not all(col in df.columns for col in _T1_EVAL_COLS):
        return pd.DataFrame()
    return df.dropna(subset=list(_T1_EVAL_COLS))


def _class_shares(labels: np.ndarray) -> dict[str, float]:
    return {label: float(np.mean(labels == label)) for label in DIRECTION_LABELS}


def _majority(shares: dict[str, float]) -> tuple[str, float]:
    # max() keeps the first of tied keys, so ties resolve in W, B, E order.
    label = max(DIRECTION_LABELS, key=lambda lbl: shares[lbl])
    return label, shares[label]


def compute_direction_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Engine-label class shares and majority-class baseline per threshold."""
    valid = _valid_t1_rows(df)
    if valid.empty:
        return pd.DataFrame()
    rows = []
    for thresh in DIRECTION_THRESHOLDS_CP:
        shares = _class_shares(_direction_labels(valid["t1_stockfish_eval"], thresh))
        majority_class, majority_acc = _majority(shares)
        rows.append({
            "threshold": thresh,
            "n": len(valid),
            **{f"pct_{label}": share for label, share in shares.items()},
            "majority_class": majority_class,
            "majority_baseline_acc": majority_acc,
        })
    return pd.DataFrame(rows)


def _agreement_row(model: str, rows: pd.DataFrame, thresh: int) -> dict[str, Any]:
    pred = _direction_labels(rows["t1_model_eval"], thresh)
    truth = _direction_labels(rows["t1_stockfish_eval"], thresh)
    accuracy = float(np.mean(pred == truth))
    _, baseline = _majority(_class_shares(truth))
    return {
        "model": model,
        "threshold": thresh,
        "n": len(rows),
        "accuracy": accuracy,
        "majority_baseline_acc": baseline,
        "lift_over_baseline": accuracy - baseline,
        "cohen_kappa": cohen_kappa(pred, truth),
        "balanced_accuracy": balanced_accuracy(pred, truth),
    }


def compute_direction_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, threshold) accuracy alongside its baseline, κ and balanced accuracy.

    Each model's baseline is computed on that model's own valid rows.
    """
    valid = _valid_t1_rows(df)
    if valid.empty:
        return pd.DataFrame()
    for model in set(df["model"]) - set(valid["model"]):
        logger.warning(f"{model}: no rows with both T1 evals; omitted from direction agreement")
    records = [
        _agreement_row(model, rows, thresh)
        for model, rows in valid.groupby("model")
        for thresh in DIRECTION_THRESHOLDS_CP
    ]
    return pd.DataFrame(records)


def _spearman(rows: pd.DataFrame) -> float:
    """Spearman ρ as Pearson correlation of average ranks (tie-aware, no scipy)."""
    model, truth = rows["t1_model_eval"], rows["t1_stockfish_eval"]
    if model.nunique() < 2 or truth.nunique() < 2:
        return float("nan")
    return float(model.rank().corr(truth.rank()))


def compute_eval_rank_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman ρ between model eval and engine eval, per model (threshold-free).

    A constant model has zero variance, so its ρ is NaN (undefined).
    """
    valid = _valid_t1_rows(df)
    if valid.empty:
        return pd.DataFrame()
    records = [
        {"model": model, "n": len(rows), "spearman_rho": _spearman(rows)}
        for model, rows in valid.groupby("model")
    ]
    return pd.DataFrame(records)


def _non_mate_t1_rows(df: pd.DataFrame) -> pd.DataFrame:
    valid = _valid_t1_rows(df)
    if valid.empty:
        return valid
    return valid[valid["t1_stockfish_eval"].abs() < MATE_SCORE_THRESHOLD_CP]


def _error_vs_zero(rows: pd.DataFrame) -> dict[str, Any]:
    truth = rows["t1_stockfish_eval"].astype(float)
    abs_error = (rows["t1_model_eval"].astype(float) - truth).abs()
    mae, zero_mae = float(abs_error.mean()), float(truth.abs().mean())
    return {
        "n": len(rows),
        "mae": mae,
        "zero_mae": zero_mae,
        "mae_ratio": mae / zero_mae if zero_mae > 0 else float("nan"),
        "median_abs_error": float(abs_error.median()),
        "zero_median_abs_error": float(truth.abs().median()),
    }


def compute_eval_error_vs_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model T1 error next to the error of always answering 0 cp, on the same rows.

    Most positions are near level, so a constant small answer scores a low error
    without reading the board; mae_ratio < 1 is the bar for real skill.
    Mate-truth rows are excluded, as for t1_abs_error_excl_mate.
    """
    rows = _non_mate_t1_rows(df)
    if rows.empty:
        return pd.DataFrame()
    for model in set(df["model"]) - set(rows["model"]):
        logger.warning(f"{model}: no non-mate rows with both T1 evals; omitted from zero baseline")
    records = [
        {"model": model, **_error_vs_zero(model_rows)}
        for model, model_rows in rows.groupby("model")
    ]
    return pd.DataFrame(records)


def _overall_zero_baseline(df: pd.DataFrame) -> dict[str, Any] | None:
    rows = _non_mate_t1_rows(df)
    if rows.empty:
        return None
    pooled = _error_vs_zero(rows)
    return {key: pooled[key] for key in ("n", "mae", "zero_mae", "mae_ratio")}


def _overall_direction_chance_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Pooled (all-model) baseline, κ and ρ for summary.json."""
    valid = _valid_t1_rows(df)
    if valid.empty:
        return {
            "t1_direction_baselines": None,
            "t1_direction_kappa_by_threshold": None,
            "t1_spearman_rho": None,
        }
    baselines = compute_direction_baselines(valid)
    kappas = {
        f"t{thresh}": cohen_kappa(
            _direction_labels(valid["t1_model_eval"], thresh),
            _direction_labels(valid["t1_stockfish_eval"], thresh),
        )
        for thresh in DIRECTION_THRESHOLDS_CP
    }
    return {
        "t1_direction_baselines": {
            f"t{row['threshold']}": {k: v for k, v in row.items() if k != "threshold"}
            for row in baselines.to_dict(orient="records")
        },
        "t1_direction_kappa_by_threshold": kappas,
        "t1_spearman_rho": _spearman(valid),
    }


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

    if "model" in df.columns:
        models_per_group = df.groupby(["model_family", "model_size_b"])["model"].nunique()
        merged = models_per_group[models_per_group > 1]
        if not merged.empty:
            raise ValueError(f"Family table would merge more than one model per row: {merged.to_dict()}")

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

    # H2: T2 CPL increases with difficulty.
    #
    # Same artefact pattern as H1 — raw CPL is dominated by mate-encoding
    # leakage on easy puzzles (which are mostly mate-in-N tactics). Reported
    # under three metrics with t2_cpl_clamped as the primary.
    def _h2_metric(column: str) -> dict[str, Any]:
        if column not in df.columns:
            return {"supported": False, "values": {}, "expect": "increasing", "missing": True}
        data = df.groupby("difficulty")[column].mean()
        values_dict = {d: (None if d not in data.index else float(data.loc[d]))
                       for d in difficulty_order}
        ordered = [values_dict[d] for d in difficulty_order]
        return {
            "supported": _trend_supported(ordered, "increasing"),
            "values": values_dict,
            "expect": "increasing",
        }

    h2_metrics = {
        "absolute_cpl": _h2_metric("t2_cpl"),
    }
    if "t2_cpl_clamped" in df.columns:
        h2_metrics["clamped_cpl"] = _h2_metric("t2_cpl_clamped")
    if "t2_wp_loss" in df.columns:
        h2_metrics["wp_loss"] = _h2_metric("t2_wp_loss")

    primary_h2 = "clamped_cpl" if "clamped_cpl" in h2_metrics else "absolute_cpl"
    results["H2"] = {
        "description": "T2 CPL increases with difficulty",
        "primary_metric": primary_h2,
        "primary_supported": h2_metrics[primary_h2]["supported"],
        "metrics": h2_metrics,
        # Backwards-compat surface
        "supported": h2_metrics[primary_h2]["supported"],
        "values": h2_metrics[primary_h2]["values"],
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

    # H5: Better T3 than T2 on openings, worse on endgames (relative performance).
    #
    # We measure the per-phase gap between normalised T3 and T2 performance.
    # T3 score is on [0, 2] (P1 + P2); we normalise to [0, 1] by dividing by 2.
    # T2 legality (`t2_legal_attempted` if present, else `t2_legal`) is on
    # [0, 1] already. The hypothesis predicts:
    #   gap(opening) > gap(middlegame) > gap(endgame)
    # i.e. the T3-vs-T2 advantage shrinks (or reverses) toward the endgame.
    phase_order = ["opening", "middlegame", "endgame"]
    t2_col = (
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
        "description": "T3 advantage over T2 is largest in openings, smallest in endgames",
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
            "t1_direction_accuracy_t50": df["t1_direction_correct"].mean(),
            "t1_direction_accuracy_by_threshold": {
                f"t{thresh}": (
                    df[f"t1_direction_correct_t{thresh}"].mean()
                    if f"t1_direction_correct_t{thresh}" in df.columns
                    else None
                )
                for thresh in DIRECTION_THRESHOLDS_CP
            },
            **_overall_direction_chance_metrics(df),
            "t1_zero_baseline": _overall_zero_baseline(df),
            "t2_legal_rate_buggy_includes_no_move_prompts": df["t2_legal"].mean(),
            "t2_legal_rate": (
                df["t2_legal_attempted"].mean()
                if "t2_legal_attempted" in df.columns else None
            ),
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

        "by_model": aggregate_by_model(df).to_dict(orient="records"),
        "by_difficulty": aggregate_by_difficulty(df).to_dict(orient="records"),
        "by_phase": aggregate_by_phase(df).to_dict(orient="records"),
        "by_source": aggregate_by_source(df).to_dict(orient="records"),
        "hypothesis_tests": compute_hypothesis_tests(df),
    }

    return summary


def _write_or_remove(table: pd.DataFrame, path: Path) -> None:
    """Write a conditional output, or delete any stale copy when there is none.

    Skipping the write silently left March pilot data in learning_deltas.csv
    through every later regeneration (D11).
    """
    if table.empty:
        if path.exists():
            path.unlink()
            logger.info(f"Removed stale {path.name} (no data this run)")
        return
    table.to_csv(path, index=False)


def _save_direction_chance_tables(df: pd.DataFrame, output_path: Path) -> None:
    tables = {
        "direction_baselines.csv": compute_direction_baselines(df),
        "direction_agreement.csv": compute_direction_agreement(df),
        "t1_rank_correlation.csv": compute_eval_rank_correlation(df),
        "t1_error_vs_zero.csv": compute_eval_error_vs_zero(df),
    }
    for name, table in tables.items():
        _write_or_remove(table, output_path / name)


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
    _save_direction_chance_tables(df, output_path)

    # Save learning deltas if available
    _write_or_remove(calculate_learning_deltas(df), output_path / "learning_deltas.csv")

    # Save summary as JSON
    summary = generate_summary(df)
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Saved metrics to {output_path}")
