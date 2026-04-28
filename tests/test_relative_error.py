"""Tests for the relative-error metric (Issue 1: H1 calibration fix)."""

import math

import pandas as pd
import pytest

from src.metrics import compute_relative_error, compute_hypothesis_tests


class TestComputeRelativeError:
    def test_zero_zero(self):
        assert compute_relative_error(0, 0) == 0.0

    def test_perfect_match(self):
        assert compute_relative_error(50, 50) == 0.0

    def test_truth_zero_model_off_by_floor(self):
        # |0 - 100| / max(0, 100) = 1.0
        assert compute_relative_error(0, 100) == 1.0

    def test_truth_above_floor(self):
        # |0 - 1000| / max(1000, 100) = 1.0
        assert compute_relative_error(0, 1000) == 1.0

    def test_truth_below_floor_uses_clamp(self):
        # |0 - 50| / max(50, 100) = 50/100 = 0.5
        assert compute_relative_error(0, 50) == 0.5

    def test_sign_mismatch(self):
        # |-100 - 100| / max(100, 100) = 200/100 = 2.0
        assert compute_relative_error(-100, 100) == 2.0

    def test_large_negative_truth(self):
        # |0 - (-2000)| / 2000 = 1.0
        assert compute_relative_error(0, -2000) == 1.0

    def test_custom_floor(self):
        # |0 - 50| / max(50, 200) = 50/200 = 0.25
        assert compute_relative_error(0, 50, floor=200) == 0.25

    def test_negative_floor_rejected(self):
        with pytest.raises(ValueError):
            compute_relative_error(0, 100, floor=-1)


class TestHypothesisH1WithRelativeError:
    @pytest.fixture
    def synthetic_df(self):
        # Build a DF where:
        #  - absolute error DECREASES with difficulty (the artefact in the real data)
        #  - relative error is constant across difficulty (the "real" picture: model is equally bad everywhere)
        #
        # easy:    truth=±2000, model=0  → abs_err=2000, rel_err=2000/2000=1.0
        # extreme: truth=±200,  model=0  → abs_err=200,  rel_err=200/200=1.0
        rows = []
        for diff, truth in [("easy", 2000), ("medium", 1000),
                            ("hard", 500), ("extreme", 200)]:
            for sign in (1, -1):
                rows.append({
                    "difficulty": diff,
                    "phase": "middlegame",
                    "model": "test_model",
                    "model_family": "test",
                    "model_size_b": 7.0,
                    "t1_model_eval": 0,
                    "t1_stockfish_eval": sign * truth,
                    "t1_absolute_error": truth,
                    "t1_direction_correct": False,
                    "t2_legal": True,
                    "t2_cpl": 0,
                    "t3_score": 0.0,
                })
        df = pd.DataFrame(rows)
        df["t1_relative_error"] = df.apply(
            lambda r: compute_relative_error(r["t1_model_eval"], r["t1_stockfish_eval"]),
            axis=1,
        )
        return df

    def test_absolute_error_artefact_visible(self, synthetic_df):
        # Sanity check: the absolute-error metric appears to reject H1 (errors decrease)
        result = compute_hypothesis_tests(synthetic_df)
        h1 = result["H1"]
        abs_metric = h1["metrics"]["absolute_error"]
        # Absolute error decreases with difficulty in this fixture
        values = abs_metric["values"]
        assert values["easy"] > values["extreme"]
        assert abs_metric["supported"] is False

    def test_relative_error_neutral(self, synthetic_df):
        # The synthetic data has CONSTANT relative error across tiers
        result = compute_hypothesis_tests(synthetic_df)
        h1 = result["H1"]
        rel_metric = h1["metrics"]["relative_error"]
        values = rel_metric["values"]
        # All tiers should have the same relative error (1.0 in this fixture)
        assert all(abs(v - 1.0) < 1e-9 for v in values.values())
        # Constant trend — strict-increasing fails, so not supported
        assert rel_metric["supported"] is False

    def test_h1_reports_all_three_metrics(self, synthetic_df):
        result = compute_hypothesis_tests(synthetic_df)
        h1 = result["H1"]
        assert "metrics" in h1
        assert set(h1["metrics"].keys()) == {
            "absolute_error", "relative_error", "direction_accuracy"
        }
        assert "primary_metric" in h1
        assert h1["primary_metric"] == "relative_error"

    def test_handles_missing_eval(self):
        # A row with NaN t1_model_eval should not blow up
        df = pd.DataFrame([
            {"difficulty": "easy", "phase": "middlegame", "model": "m",
             "model_family": "test", "model_size_b": 7.0,
             "t1_model_eval": None, "t1_stockfish_eval": 100,
             "t1_absolute_error": None, "t1_direction_correct": None,
             "t1_relative_error": None,
             "t2_legal": None, "t2_cpl": None, "t3_score": None},
            {"difficulty": "medium", "phase": "middlegame", "model": "m",
             "model_family": "test", "model_size_b": 7.0,
             "t1_model_eval": 50, "t1_stockfish_eval": 100,
             "t1_absolute_error": 50, "t1_direction_correct": True,
             "t1_relative_error": 0.5,
             "t2_legal": True, "t2_cpl": 0, "t3_score": 0.0},
        ])
        result = compute_hypothesis_tests(df)
        # Should produce a result without raising
        assert "H1" in result
