"""save_metrics must not leave stale conditional outputs behind (D11)."""

import pandas as pd

from src.metrics import save_metrics


def _correction_free_df() -> pd.DataFrame:
    return pd.DataFrame([{
        "job_id": "j1", "job_type": "standard", "position_id": 1, "model": "mistral:7b",
        "model_family": "mistral", "model_size_b": 7.0, "difficulty": "easy",
        "phase": "middlegame", "source": "lichess_puzzles",
        "t1_absolute_error": 10, "t1_direction_correct": True, "t2_legal": True,
        "t2_cpl": 0, "t3_p1_side_correct": 1, "t3_p2_theme_correct": 0, "t3_score": 1,
        "inference_ms": 100,
    }])


def test_empty_learning_deltas_removes_stale_file(tmp_path):
    stale = tmp_path / "learning_deltas.csv"
    stale.write_text("model,cpl_correction\nqwen2.5:7b,10\n")
    save_metrics(_correction_free_df(), str(tmp_path))
    assert not stale.exists()


def test_missing_t1_columns_removes_stale_direction_tables(tmp_path):
    names = ("direction_baselines.csv", "direction_agreement.csv", "t1_rank_correlation.csv")
    for name in names:
        (tmp_path / name).write_text("stale\n")
    save_metrics(_correction_free_df(), str(tmp_path))
    for name in names:
        assert not (tmp_path / name).exists(), name
