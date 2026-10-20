"""Tests for chance-corrected T1 direction metrics (specs/direction-baseline-kappa.md)."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    DIRECTION_THRESHOLDS_CP,
    balanced_accuracy,
    cohen_kappa,
    compute_direction_agreement,
    compute_direction_baselines,
    compute_eval_rank_correlation,
    generate_summary,
    save_metrics,
)

# Truth evals chosen so the class mix changes with threshold:
# at ±0 most rows are W/B, at ±200 most rows are E.
TRUTH_EVALS = [-450, -180, -120, -60, -20, 0, 15, 40, 70, 130, 160, 220, 300, 600, 9990]


def _df(model_evals_by_model: dict[str, list[float | None]], truth=TRUTH_EVALS) -> pd.DataFrame:
    rows = []
    for model, evals in model_evals_by_model.items():
        for i, (me, sf) in enumerate(zip(evals, truth)):
            rows.append({
                "model": model,
                "position_id": i,
                "t1_model_eval": me,
                "t1_stockfish_eval": sf,
            })
    return pd.DataFrame(rows)


def _row(table: pd.DataFrame, model: str, threshold: int) -> pd.Series:
    match = table[(table["model"] == model) & (table["threshold"] == threshold)]
    assert len(match) == 1
    return match.iloc[0]


# ---------- cohen_kappa ----------

def test_kappa_perfect_agreement_is_one():
    labels = ["W", "B", "E", "W", "B"]
    assert cohen_kappa(labels, labels) == pytest.approx(1.0)


def test_kappa_constant_prediction_is_zero():
    truth = ["W", "B", "E", "E", "W", "E"]
    assert cohen_kappa(["E"] * len(truth), truth) == pytest.approx(0.0)


def test_kappa_known_value():
    # p_o = 0.5; p_e = 0.5*0.5 + 0.5*0.5 = 0.5 -> kappa 0
    assert cohen_kappa(["W", "W", "B", "B"], ["W", "B", "W", "B"]) == pytest.approx(0.0)
    # p_o = 0.75; p_e = 0.5 -> kappa 0.5
    assert cohen_kappa(["W", "W", "B", "B"], ["W", "W", "B", "W"]) == pytest.approx(0.5)


def test_kappa_undefined_when_both_constant_same_label():
    assert math.isnan(cohen_kappa(["E", "E", "E"], ["E", "E", "E"]))


def test_kappa_raises_on_empty():
    with pytest.raises(ValueError):
        cohen_kappa([], [])


def test_kappa_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        cohen_kappa(["W"], ["W", "B"])


# ---------- balanced_accuracy ----------

def test_balanced_accuracy_constant_prediction_is_one_over_k():
    truth = ["W", "W", "W", "W", "B", "E"]
    assert balanced_accuracy(["W"] * 6, truth) == pytest.approx(1 / 3)


def test_balanced_accuracy_perfect_is_one():
    truth = ["W", "B", "E", "E"]
    assert balanced_accuracy(truth, truth) == pytest.approx(1.0)


def test_balanced_accuracy_raises_on_empty():
    with pytest.raises(ValueError):
        balanced_accuracy([], [])


# ---------- compute_direction_baselines ----------

def test_baselines_shares_sum_to_one_and_majority_matches():
    df = _df({"m": [0] * len(TRUTH_EVALS)})
    baselines = compute_direction_baselines(df)
    assert list(baselines["threshold"]) == list(DIRECTION_THRESHOLDS_CP)
    for _, row in baselines.iterrows():
        shares = {"W": row["pct_W"], "B": row["pct_B"], "E": row["pct_E"]}
        assert sum(shares.values()) == pytest.approx(1.0)
        assert row["majority_baseline_acc"] == pytest.approx(max(shares.values()))
        assert shares[row["majority_class"]] == pytest.approx(row["majority_baseline_acc"])
        assert row["n"] == len(TRUTH_EVALS)


def test_baselines_equal_class_grows_with_threshold():
    baselines = compute_direction_baselines(_df({"m": [0] * len(TRUTH_EVALS)}))
    pct_e = list(baselines["pct_E"])
    assert pct_e == sorted(pct_e)
    assert baselines.iloc[-1]["majority_class"] == "E"


def test_baselines_exclude_missing_rows():
    evals = [0] * len(TRUTH_EVALS)
    evals[0] = None
    baselines = compute_direction_baselines(_df({"m": evals}))
    assert (baselines["n"] == len(TRUTH_EVALS) - 1).all()


# ---------- compute_direction_agreement ----------

def test_constant_zero_model_gets_kappa_zero_and_no_lift():
    table = compute_direction_agreement(_df({"always_zero": [0] * len(TRUTH_EVALS)}))
    for thresh in DIRECTION_THRESHOLDS_CP:
        row = _row(table, "always_zero", thresh)
        assert row["cohen_kappa"] == pytest.approx(0.0)
        assert row["accuracy"] <= row["majority_baseline_acc"] + 1e-12
        assert row["lift_over_baseline"] <= 1e-12


def test_constant_zero_matches_majority_baseline_when_equal_is_majority():
    table = compute_direction_agreement(_df({"always_zero": [0] * len(TRUTH_EVALS)}))
    row = _row(table, "always_zero", 200)
    assert row["accuracy"] == pytest.approx(row["majority_baseline_acc"])


def test_perfect_model_scores_kappa_one():
    table = compute_direction_agreement(_df({"oracle": list(TRUTH_EVALS)}))
    for thresh in DIRECTION_THRESHOLDS_CP:
        row = _row(table, "oracle", thresh)
        assert row["accuracy"] == pytest.approx(1.0)
        assert row["cohen_kappa"] == pytest.approx(1.0)
        assert row["balanced_accuracy"] == pytest.approx(1.0)


def test_sign_flipped_model_scores_negative_kappa():
    table = compute_direction_agreement(_df({"flipped": [-x for x in TRUTH_EVALS]}))
    row = _row(table, "flipped", 0)
    assert row["cohen_kappa"] < 0


def test_agreement_one_row_per_model_threshold():
    df = _df({"a": [0] * len(TRUTH_EVALS), "b": list(TRUTH_EVALS)})
    table = compute_direction_agreement(df)
    assert len(table) == 2 * len(DIRECTION_THRESHOLDS_CP)
    assert set(table.columns) >= {
        "model", "threshold", "n", "accuracy", "majority_baseline_acc",
        "lift_over_baseline", "cohen_kappa", "balanced_accuracy",
    }


def test_agreement_baseline_uses_each_models_own_rows():
    full = list(TRUTH_EVALS)
    # Model "partial" skips every position whose truth is outside ±50.
    partial = [x if abs(x) <= 50 else None for x in TRUTH_EVALS]
    table = compute_direction_agreement(_df({"full": full, "partial": partial}))
    row = _row(table, "partial", 50)
    assert row["n"] == sum(1 for x in partial if x is not None)
    # On its own rows every truth is Equal at ±50, so its baseline is 1.0.
    assert row["majority_baseline_acc"] == pytest.approx(1.0)
    assert _row(table, "full", 50)["majority_baseline_acc"] < 1.0


def test_agreement_empty_df_returns_empty():
    assert compute_direction_agreement(pd.DataFrame()).empty


# ---------- compute_eval_rank_correlation ----------

def test_rank_correlation_perfect_and_flipped():
    df = _df({"oracle": list(TRUTH_EVALS), "flipped": [-x for x in TRUTH_EVALS]})
    table = compute_eval_rank_correlation(df).set_index("model")
    assert table.loc["oracle", "spearman_rho"] == pytest.approx(1.0)
    assert table.loc["flipped", "spearman_rho"] == pytest.approx(-1.0)


def test_rank_correlation_is_scale_free():
    scaled = [x * 0.1 for x in TRUTH_EVALS]
    table = compute_eval_rank_correlation(_df({"scaled": scaled}))
    assert table.iloc[0]["spearman_rho"] == pytest.approx(1.0)


def test_rank_correlation_constant_model_is_nan():
    table = compute_eval_rank_correlation(_df({"always_zero": [0] * len(TRUTH_EVALS)}))
    assert math.isnan(table.iloc[0]["spearman_rho"])


# ---------- summary + save_metrics wiring ----------

def _full_df() -> pd.DataFrame:
    df = _df({"a": [0] * len(TRUTH_EVALS), "b": list(TRUTH_EVALS)})
    n = len(df)
    df["job_id"] = [f"j{i}" for i in range(n)]
    df["job_type"] = "standard"
    df["difficulty"] = "easy"
    df["phase"] = "middlegame"
    df["source"] = "lichess_puzzles"
    df["model_family"] = "x"
    df["model_size_b"] = np.where(df["model"] == "a", 3, 7)
    df["t1_absolute_error"] = (df["t1_model_eval"] - df["t1_stockfish_eval"]).abs()
    df["t1_direction_correct"] = True
    df["t2_legal"] = True
    df["t2_cpl"] = 0
    df["t3_p1_side_correct"] = 1
    df["t3_p2_theme_correct"] = 0
    df["t3_score"] = 1
    df["inference_ms"] = 100
    return df


def test_summary_has_chance_corrected_keys():
    overall = generate_summary(_full_df())["overall_metrics"]
    assert set(overall["t1_direction_baselines"]) == {f"t{t}" for t in DIRECTION_THRESHOLDS_CP}
    assert set(overall["t1_direction_kappa_by_threshold"]) == {f"t{t}" for t in DIRECTION_THRESHOLDS_CP}
    assert -1.0 <= overall["t1_spearman_rho"] <= 1.0


def test_save_metrics_writes_new_csvs(tmp_path):
    out = tmp_path / "metrics"
    save_metrics(_full_df(), str(out))
    for name in ("direction_baselines.csv", "direction_agreement.csv", "t1_rank_correlation.csv"):
        assert (out / name).exists(), name
    summary = json.loads((out / "summary.json").read_text())
    assert "t1_direction_baselines" in summary["overall_metrics"]
