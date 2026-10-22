"""Tests for the T1 zero-answer error baseline (specs/t1-zero-baseline-and-ft-comparison.md)."""

import json
import math

import pandas as pd
import pytest

from src.metrics import (
    MATE_SCORE_THRESHOLD_CP,
    compute_eval_error_vs_zero,
    generate_summary,
    save_metrics,
)

TRUTH_EVALS = [-450, -120, -20, 0, 40, 130, 300]
MATE_TRUTH = MATE_SCORE_THRESHOLD_CP + 990


def _df(model_evals_by_model: dict[str, list[float | None]], truth=TRUTH_EVALS) -> pd.DataFrame:
    rows = []
    for model, evals in model_evals_by_model.items():
        for i, (me, sf) in enumerate(zip(evals, truth)):
            rows.append({
                "model": model,
                "position_id": i,
                "job_id": f"{model}_{i}",
                "job_type": "standard",
                "difficulty": "easy",
                "phase": "middlegame",
                "source": "lichess_puzzles",
                "model_family": "x",
                "model_size_b": 3,
                "t1_model_eval": me,
                "t1_stockfish_eval": sf,
                "t1_absolute_error": None if me is None else abs(me - sf),
                "t1_direction_correct": True,
                "t2_legal": True,
                "t2_cpl": 0,
                "t3_p1_side_correct": 1,
                "t3_p2_theme_correct": 0,
                "t3_score": 1,
                "inference_ms": 100,
            })
    return pd.DataFrame(rows)


def _row(table: pd.DataFrame, model: str) -> pd.Series:
    match = table[table["model"] == model]
    assert len(match) == 1
    return match.iloc[0]


def test_constant_zero_predictor_has_ratio_one():
    table = compute_eval_error_vs_zero(_df({"zero": [0] * len(TRUTH_EVALS)}))
    row = _row(table, "zero")
    assert row["mae_ratio"] == pytest.approx(1.0)
    assert row["mae"] == pytest.approx(row["zero_mae"])
    assert row["median_abs_error"] == pytest.approx(row["zero_median_abs_error"])


def test_perfect_predictor_has_zero_error():
    row = _row(compute_eval_error_vs_zero(_df({"perfect": list(TRUTH_EVALS)})), "perfect")
    assert row["mae"] == 0
    assert row["mae_ratio"] == 0


def test_zero_mae_matches_mean_absolute_truth():
    row = _row(compute_eval_error_vs_zero(_df({"m": [5] * len(TRUTH_EVALS)})), "m")
    expected = sum(abs(t) for t in TRUTH_EVALS) / len(TRUTH_EVALS)
    assert row["zero_mae"] == pytest.approx(expected)
    assert row["n"] == len(TRUTH_EVALS)


def test_mate_truth_rows_are_excluded():
    truth = TRUTH_EVALS + [MATE_TRUTH, -MATE_TRUTH]
    df = _df({"m": [0] * len(truth)}, truth=truth)
    row = _row(compute_eval_error_vs_zero(df), "m")
    assert row["n"] == len(TRUTH_EVALS)
    assert row["zero_mae"] == pytest.approx(sum(abs(t) for t in TRUTH_EVALS) / len(TRUTH_EVALS))


def test_rows_missing_an_eval_are_excluded():
    evals = [0] * len(TRUTH_EVALS)
    evals[0] = None
    row = _row(compute_eval_error_vs_zero(_df({"m": evals})), "m")
    assert row["n"] == len(TRUTH_EVALS) - 1


def test_all_zero_truth_gives_nan_ratio():
    truth = [0, 0, 0]
    row = _row(compute_eval_error_vs_zero(_df({"m": [10, 20, 30]}, truth=truth)), "m")
    assert row["zero_mae"] == 0
    assert math.isnan(row["mae_ratio"])


def test_one_row_per_model_and_empty_model_omitted():
    df = _df({
        "a": [0] * len(TRUTH_EVALS),
        "b": list(TRUTH_EVALS),
        "empty": [None] * len(TRUTH_EVALS),
    })
    table = compute_eval_error_vs_zero(df)
    assert sorted(table["model"]) == ["a", "b"]


def test_empty_input_returns_empty_frame():
    assert compute_eval_error_vs_zero(pd.DataFrame()).empty


def test_save_metrics_writes_csv_and_summary_key(tmp_path):
    df = _df({"zero": [0] * len(TRUTH_EVALS), "perfect": list(TRUTH_EVALS)})
    save_metrics(df, output_dir=str(tmp_path))

    table = pd.read_csv(tmp_path / "t1_error_vs_zero.csv")
    assert set(table["model"]) == {"zero", "perfect"}

    summary = json.loads((tmp_path / "summary.json").read_text())
    pooled = summary["overall_metrics"]["t1_zero_baseline"]
    assert pooled["n"] == 2 * len(TRUTH_EVALS)
    assert pooled["mae_ratio"] == pytest.approx(0.5)


def test_summary_key_is_none_without_valid_rows():
    df = _df({"empty": [None] * len(TRUTH_EVALS)})
    assert generate_summary(df)["overall_metrics"]["t1_zero_baseline"] is None
