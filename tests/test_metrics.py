"""Tests for src/metrics.py"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.metrics import (
    load_results_df,
    aggregate_by_model,
    aggregate_by_difficulty,
    aggregate_by_phase,
    aggregate_by_source,
    aggregate_by_model_family,
    calculate_hallucination_rate,
    calculate_learning_deltas,
    compute_hypothesis_tests,
    generate_summary,
    save_metrics,
)


@pytest.fixture
def sample_df():
    """A minimal DataFrame with all required columns."""
    records = []
    models = [("llama3.2:3b", "llama", 3), ("qwen2.5:7b", "qwen", 7)]
    difficulties = ["easy", "medium", "hard", "extreme"]
    phases = ["opening", "middlegame", "endgame"]
    for model, family, size in models:
        for i, diff in enumerate(difficulties):
            phase = phases[i % len(phases)]
            records.append({
                "job_id": f"j_{model}_{diff}",
                "job_type": "standard",
                "position_id": i + 1,
                "model": model,
                "model_family": family,
                "model_size_b": size,
                "difficulty": diff,
                "phase": phase,
                "source": "lichess_puzzles",
                "t1_absolute_error": 100 + i * 50,
                "t1_direction_correct": True,
                "t2_legal": i % 2 == 0,
                "t2_cpl": i * 20,
                "t3_p1_side_correct": 1,
                "t3_p2_theme_correct": i % 2,
                "t3_score": 1 + (i % 2),
                "inference_ms": 1000 + i * 100,
            })
    return pd.DataFrame(records)


@pytest.fixture
def correction_df():
    """DataFrame with standard + correction + control rows."""
    return pd.DataFrame([
        {
            "job_id": "parent_1", "job_type": "standard", "model": "llama3.2:3b",
            "position_id": 1, "parent_job_id": None, "t2_cpl": 100,
            "model_family": "llama", "model_size_b": 3, "difficulty": "easy",
            "phase": "opening", "source": "lichess_puzzles",
            "t1_absolute_error": 50, "t1_direction_correct": True,
            "t2_legal": True, "t3_p1_side_correct": 1, "t3_p2_theme_correct": 0,
            "t3_score": 1, "inference_ms": 1000,
        },
        {
            "job_id": "parent_1_correction", "job_type": "correction",
            "position_id": 1, "model": "llama3.2:3b", "parent_job_id": "parent_1", "t2_cpl": 40,
            "model_family": "llama", "model_size_b": 3, "difficulty": "easy",
            "phase": "opening", "source": "lichess_puzzles",
            "t1_absolute_error": 50, "t1_direction_correct": True,
            "t2_legal": True, "t3_p1_side_correct": 1, "t3_p2_theme_correct": 0,
            "t3_score": 1, "inference_ms": 1200,
        },
        {
            "job_id": "parent_1_control", "job_type": "control",
            "position_id": 1, "model": "llama3.2:3b", "parent_job_id": "parent_1", "t2_cpl": 80,
            "model_family": "llama", "model_size_b": 3, "difficulty": "easy",
            "phase": "opening", "source": "lichess_puzzles",
            "t1_absolute_error": 50, "t1_direction_correct": True,
            "t2_legal": True, "t3_p1_side_correct": 1, "t3_p2_theme_correct": 0,
            "t3_score": 1, "inference_ms": 1100,
        },
    ])


class TestLoadResultsDf:
    def test_empty_file_returns_empty_df(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        df = load_results_df(str(f))
        assert df.empty

    def test_loads_records(self, tmp_path):
        f = tmp_path / "results.jsonl"
        f.write_text('{"job_id": "j1", "model": "x"}\n{"job_id": "j2", "model": "y"}\n')
        df = load_results_df(str(f))
        assert len(df) == 2


class TestAggregateByModel:
    def test_empty_df(self):
        assert aggregate_by_model(pd.DataFrame()).empty

    def test_returns_one_row_per_model(self, sample_df):
        result = aggregate_by_model(sample_df)
        assert len(result) == 2
        assert set(result["model"]) == {"llama3.2:3b", "qwen2.5:7b"}

    def test_has_expected_columns(self, sample_df):
        result = aggregate_by_model(sample_df)
        assert "t2_legal_mean" in result.columns
        assert "job_id_count" in result.columns


class TestAggregateByDifficulty:
    def test_empty_df(self):
        assert aggregate_by_difficulty(pd.DataFrame()).empty

    def test_returns_model_difficulty_pairs(self, sample_df):
        result = aggregate_by_difficulty(sample_df)
        assert "model" in result.columns
        assert "difficulty" in result.columns
        assert len(result) == 8  # 2 models × 4 difficulties


class TestAggregateByPhase:
    def test_empty_df(self):
        assert aggregate_by_phase(pd.DataFrame()).empty

    def test_has_phase_column(self, sample_df):
        result = aggregate_by_phase(sample_df)
        assert "phase" in result.columns


class TestAggregateBySource:
    def test_empty_df(self):
        assert aggregate_by_source(pd.DataFrame()).empty

    def test_has_source_column(self, sample_df):
        result = aggregate_by_source(sample_df)
        assert "source" in result.columns


class TestAggregateByModelFamily:
    def test_empty_df(self):
        assert aggregate_by_model_family(pd.DataFrame()).empty

    def test_groups_by_family_and_size(self, sample_df):
        result = aggregate_by_model_family(sample_df)
        assert "model_family" in result.columns
        assert "model_size_b" in result.columns


class TestCalculateHallucinationRate:
    def test_empty_df(self):
        assert calculate_hallucination_rate(pd.DataFrame()).empty

    def test_calculates_rate(self, sample_df):
        result = calculate_hallucination_rate(sample_df)
        assert "hallucination_rate" in result.columns
        assert all(0 <= r <= 1 for r in result["hallucination_rate"])

    def test_all_legal(self):
        df = pd.DataFrame([
            {"job_id": "j1", "model": "x", "difficulty": "easy", "t2_legal": True},
            {"job_id": "j2", "model": "x", "difficulty": "easy", "t2_legal": True},
        ])
        result = calculate_hallucination_rate(df)
        assert result["hallucination_rate"].iloc[0] == 0.0

    def test_all_illegal(self):
        df = pd.DataFrame([
            {"job_id": "j1", "model": "x", "difficulty": "easy", "t2_legal": False},
        ])
        result = calculate_hallucination_rate(df)
        assert result["hallucination_rate"].iloc[0] == 1.0


class TestCalculateLearningDeltas:
    def test_empty_df(self):
        assert calculate_learning_deltas(pd.DataFrame()).empty

    def test_no_correction_rows(self, sample_df):
        # sample_df only has standard rows
        assert calculate_learning_deltas(sample_df).empty

    def test_calculates_deltas(self, correction_df):
        result = calculate_learning_deltas(correction_df)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["cpl_attempt_1"] == 100
        assert row["cpl_correction"] == 40
        assert row["cpl_control"] == 80
        assert row["delta_correction"] == 60
        assert row["delta_control"] == 20
        assert row["net_feedback_effect"] == 40

    def test_skips_none_cpl(self):
        df = pd.DataFrame([
            {"job_id": "p1", "job_type": "standard", "model": "m", "parent_job_id": None, "t2_cpl": None},
            {"job_id": "p1_corr", "job_type": "correction", "model": "m", "parent_job_id": "p1", "t2_cpl": 10},
            {"job_id": "p1_ctrl", "job_type": "control", "model": "m", "parent_job_id": "p1", "t2_cpl": 20},
        ])
        result = calculate_learning_deltas(df)
        assert result.empty

    def test_skips_missing_parent(self):
        df = pd.DataFrame([
            {"job_id": "orphan_corr", "job_type": "correction", "model": "m", "parent_job_id": "ghost", "t2_cpl": 10},
            {"job_id": "orphan_ctrl", "job_type": "control", "model": "m", "parent_job_id": "ghost", "t2_cpl": 20},
        ])
        result = calculate_learning_deltas(df)
        assert result.empty


class TestTestHypotheses:
    def test_empty_df_returns_error(self):
        result = compute_hypothesis_tests(pd.DataFrame())
        assert "error" in result

    def test_returns_all_hypotheses(self, sample_df):
        result = compute_hypothesis_tests(sample_df)
        assert "H1" in result
        assert "H2" in result
        assert "H3" in result
        assert "H4" in result
        assert "H5" in result

    def test_h1_structure(self, sample_df):
        result = compute_hypothesis_tests(sample_df)
        assert "supported" in result["H1"]
        assert "values" in result["H1"]

    def test_h4_no_family_with_multiple_sizes(self):
        # Single model per family — H4 should have empty by_family
        df = pd.DataFrame([{
            "job_id": "j1", "model": "llama3.2:3b", "model_family": "llama",
            "model_size_b": 3, "difficulty": "easy", "phase": "opening",
            "source": "lichess_puzzles",
            "t1_absolute_error": 50, "t1_direction_correct": True,
            "t2_legal": True, "t2_cpl": 20,
            "t3_p1_side_correct": 1, "t3_p2_theme_correct": 0, "t3_score": 1,
            "inference_ms": 1000,
        }])
        result = compute_hypothesis_tests(df)
        assert result["H4"]["by_family"] == {}


class TestGenerateSummary:
    def test_empty_df_returns_error(self):
        result = generate_summary(pd.DataFrame())
        assert "error" in result

    def test_returns_summary_keys(self, sample_df):
        result = generate_summary(sample_df)
        assert "total_jobs" in result
        assert "models_tested" in result
        assert "overall_metrics" in result
        assert result["total_jobs"] == len(sample_df)


class TestSaveMetrics:
    def test_saves_csv_files(self, tmp_path, sample_df):
        output_dir = str(tmp_path / "metrics")
        save_metrics(sample_df, output_dir)
        assert (tmp_path / "metrics" / "by_model.csv").exists()
        assert (tmp_path / "metrics" / "by_difficulty.csv").exists()
        assert (tmp_path / "metrics" / "summary.json").exists()

    def test_empty_df_no_files(self, tmp_path):
        output_dir = str(tmp_path / "metrics")
        save_metrics(pd.DataFrame(), output_dir)
        # Directory may be created but no CSVs
        assert not (tmp_path / "metrics" / "by_model.csv").exists()

    def test_saves_learning_deltas_when_present(self, tmp_path, correction_df):
        output_dir = str(tmp_path / "metrics")
        save_metrics(correction_df, output_dir)
        assert (tmp_path / "metrics" / "learning_deltas.csv").exists()
