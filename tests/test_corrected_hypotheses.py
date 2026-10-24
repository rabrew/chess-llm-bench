"""H3, H4 and family table use corrected columns (specs/corrected-hypotheses-family.md, T3-6)."""

import pandas as pd
import pytest

from src.metrics import aggregate_by_model_family, compute_hypothesis_tests

DIFFICULTIES = ["easy", "medium", "hard", "extreme"]
TRUTH = [-400, -150, -60, 0, 60, 150, 400, 800]


def _rows(model, size, t3_v2, t3_old, perfect_t1, blunder):
    rows = []
    for d_index, difficulty in enumerate(DIFFICULTIES):
        for i, truth in enumerate(TRUTH):
            rows.append({
                "model": model, "model_family": "fam", "model_size_b": size,
                "job_id": f"{model}-{difficulty}-{i}", "position_id": i, "difficulty": difficulty,
                "phase": "middlegame", "source": "lichess_puzzles", "prompt_format": "fen_only",
                "t1_model_eval": truth if perfect_t1 else (i * 37) % 11, "t1_stockfish_eval": truth,
                "t1_absolute_error": 10.0, "t1_direction_correct": True,
                "t2_legal": True, "t2_legal_attempted": 1.0, "t2_cpl": 500.0,
                "t2_cpl_clamped": 1000.0 if blunder else 50.0,
                "t3_score": t3_old[d_index], "t3_score_v2": t3_v2[d_index],
                "t3_p1_side_correct": 1, "t3_p2_theme_correct": 0, "inference_ms": 1,
            })
    return rows


def _frame():
    falling, rising = [1.5, 1.2, 0.9, 0.6], [0.6, 0.9, 1.2, 1.5]
    return pd.DataFrame(
        _rows("small", 3, t3_v2=falling, t3_old=rising, perfect_t1=False, blunder=True)
        + _rows("large", 70, t3_v2=[x + 0.1 for x in falling], t3_old=rising, perfect_t1=True, blunder=False)
    )


def test_h3_follows_v2_score():
    h3 = compute_hypothesis_tests(_frame())["H3"]
    assert h3["supported"] is True
    assert h3["metric"] == "t3_score_v2"


def test_h4_uses_rho_blunder_rate_and_v2():
    h4 = compute_hypothesis_tests(_frame())["H4"]
    fam = h4["by_family"]["fam"]
    assert fam["better_t1"] and fam["better_t2"] and fam["better_t3"]
    assert h4["columns_used"] == {"t1": "spearman_rho", "t2": "blunder_rate", "t3": "t3_score_v2"}


def test_family_table_has_corrected_columns():
    table = aggregate_by_model_family(_frame()).set_index("model_size_b")
    assert {"t1_spearman_rho", "t2_blunder_rate", "t2_cpl_clamped_mean", "t2_legal_rate",
            "t3_score_v2_mean", "n_records"} <= set(table.columns)
    assert "t2_legal" not in table.columns
    assert table.loc[70, "t1_spearman_rho"] == pytest.approx(1.0)
    assert table.loc[3, "t2_blunder_rate"] == pytest.approx(1.0)
    assert table.loc[3, "n_records"] == len(DIFFICULTIES) * len(TRUTH)


def test_family_table_falls_back_without_derived_columns():
    frame = _frame().drop(columns=["t2_cpl_clamped", "t2_legal_attempted", "t3_score_v2"])
    table = aggregate_by_model_family(frame).set_index("model_size_b")
    assert table.loc[3, "t3_score_v2_mean"] == pytest.approx(frame[frame["model_size_b"] == 3]["t3_score"].mean())
    assert table.loc[3, "t2_legal_rate"] == pytest.approx(1.0)


def test_merge_guard_still_raises():
    frame = _frame()
    frame.loc[frame["model"] == "small", "model_size_b"] = 70
    with pytest.raises(ValueError):
        aggregate_by_model_family(frame)
