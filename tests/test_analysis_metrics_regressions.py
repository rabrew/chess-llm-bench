"""Post-results analysis fixes using synthetic rows and temporary datasets."""

import json

import numpy as np
import pandas as pd
import pytest

from src import metrics


def _positions():
    return pd.DataFrame({
        "model": ["m"] * 8,
        "position_id": [1] * 4 + [2] * 4,
        "t2_cpl_clamped": [0] * 4 + [1000] * 4,
    })


def test_repeating_position_responses_does_not_increase_ci_precision():
    original = _positions().iloc[[0, 4]]
    once = metrics.compute_blunder_rates(original).iloc[0]
    repeated = metrics.compute_blunder_rates(_positions()).iloc[0]
    assert (once.ci_low, once.ci_high) == (repeated.ci_low, repeated.ci_high)
    assert repeated.ci_method == "position_cluster_bootstrap"
    assert repeated.n_positions == 2
    assert repeated.ci_low == 0 and repeated.ci_high == 1


def test_pooled_blunders_cluster_shared_positions_across_models():
    one_model = _positions()
    pooled = pd.concat([one_model, one_model.assign(model="other")], ignore_index=True)
    result = metrics._overall_blunder_rate(pooled)
    assert result["n"] == 16
    assert result["n_positions"] == 2
    assert result["rate"] == 0.5
    assert result["ci_low"] == 0 and result["ci_high"] == 1
    assert result["ci_method"] == "position_cluster_bootstrap"


def test_cluster_interval_preserves_response_weighted_point_estimate():
    frame = _positions().iloc[[0, 4, 5, 6, 7]]
    row = metrics.compute_blunder_rates(frame).iloc[0]
    assert row.blunder_rate == pytest.approx(4 / 5)
    assert row.n == 5 and row.n_positions == 2
    assert (row.ci_low, row.ci_high) == (0, 1)


def test_cluster_ci_is_seeded_and_invariant_to_row_order():
    frame = _positions()
    first = metrics.compute_blunder_rates(frame).iloc[0]
    other = metrics.compute_blunder_rates(frame.iloc[::-1]).iloc[0]
    assert (first.ci_low, first.ci_high) == (other.ci_low, other.ci_high)


def test_cluster_ci_is_unchanged_when_draws_need_smaller_memory_batches(monkeypatch):
    frame = pd.DataFrame({"model": ["m"] * 40, "position_id": np.repeat(np.arange(20), 2),
                          "t2_cpl_clamped": [0, 1000, 1000, 1000] * 10})
    first = metrics.compute_blunder_rates(frame).iloc[0]
    monkeypatch.setattr(metrics, "BOOTSTRAP_MAX_DRAW_ELEMENTS", 20)
    small_batches = metrics.compute_blunder_rates(frame).iloc[0]
    assert (first.ci_low, first.ci_high) == (small_batches.ci_low, small_batches.ci_high)
    assert small_batches.ci_resamples == metrics.BOOTSTRAP_RESAMPLES
    assert small_batches.ci_seed == metrics.BOOTSTRAP_SEED


@pytest.mark.parametrize("missing_column", [True, False])
def test_missing_cluster_identifiers_get_explicit_iid_fallback(caplog, missing_column):
    frame = _positions()
    if missing_column:
        frame = frame.drop(columns="position_id")
    else:
        frame.loc[0, "position_id"] = np.nan
    row = metrics.compute_blunder_rates(frame).iloc[0]
    assert row.ci_method == "iid_response_bootstrap_missing_position_id"
    assert pd.isna(row.n_positions)
    assert "position" in caplog.text.lower() and "iid" in caplog.text.lower()


def test_overlap_output_does_not_claim_statistical_ties():
    frame = pd.concat([_positions(), _positions().assign(model="other")], ignore_index=True)
    result = metrics.compute_blunder_rates(frame)
    assert result["interval_overlap_with_neighbor"].all()
    assert "rank_tied" not in result.columns


def test_hallucination_denominator_excludes_non_move_tasks():
    frame = pd.DataFrame({
        "model": ["m"] * 5, "difficulty": ["easy"] * 5,
        "job_id": list("abcde"),
        "prompt_format": ["fen_only", "board_text", "board_text+legal", "eval_only", "explanation_only"],
        "t2_legal": [True, False, None, False, False],
    })
    row = metrics.calculate_hallucination_rate(frame).iloc[0]
    assert row.job_id == 3
    assert row.hallucination_rate == pytest.approx(2 / 3)
    assert row.denominator_method == "move_requesting_prompts"


def test_hallucination_legacy_fallback_is_explicit(caplog):
    frame = pd.DataFrame({"model": ["m"], "difficulty": ["easy"], "job_id": ["a"], "t2_legal": [True]})
    row = metrics.calculate_hallucination_rate(frame).iloc[0]
    assert row.hallucination_rate == 0
    assert row.denominator_method == "all_records_legacy_unknown_applicability"
    assert "prompt_format" in caplog.text


def test_hallucination_all_non_move_tasks_has_empty_output():
    frame = pd.DataFrame({"model": ["m"], "difficulty": ["easy"], "job_id": ["a"],
                          "prompt_format": ["eval_only"], "t2_legal": [False]})
    assert metrics.calculate_hallucination_rate(frame).empty


def test_unknown_prompt_formats_are_excluded_with_warning(caplog):
    frame = pd.DataFrame({"model": ["m", "m"], "difficulty": ["easy", "easy"],
                          "job_id": ["a", "b"], "prompt_format": ["board_text", None],
                          "t2_legal": [True, False]})
    row = metrics.calculate_hallucination_rate(frame).iloc[0]
    assert row.job_id == 1 and row.hallucination_rate == 0
    assert "1 rows with unknown prompt_format" in caplog.text


def _write_tier(directory, fen):
    directory.mkdir(exist_ok=True)
    (directory / "easy.json").write_text(json.dumps([{"id": 1, "fen": fen}]))


def test_fen_lookup_cache_tracks_dataset_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "_FEN_LOOKUP_CACHE", None)
    first, second = tmp_path / "one", tmp_path / "two"
    _write_tier(first, "first")
    _write_tier(second, "second")
    assert metrics._build_fen_lookup(str(first))[1] == "first"
    assert metrics._build_fen_lookup(str(second))[1] == "second"


def test_fen_lookup_cache_refreshes_changed_files(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "_FEN_LOOKUP_CACHE", None)
    _write_tier(tmp_path, "old")
    assert metrics._build_fen_lookup(str(tmp_path))[1] == "old"
    _write_tier(tmp_path, "replacement")
    assert metrics._build_fen_lookup(str(tmp_path))[1] == "replacement"


def test_fen_lookup_does_not_cache_absent_directory_forever(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "_FEN_LOOKUP_CACHE", None)
    directory = tmp_path / "new"
    assert metrics._build_fen_lookup(str(directory)) == {}
    _write_tier(directory, "created")
    assert metrics._build_fen_lookup(str(directory))[1] == "created"
