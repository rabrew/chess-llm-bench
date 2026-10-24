"""T1 perspective check split by side to move (specs/t1-side-to-move.md, T1-3)."""

import json
import math

import pandas as pd
import pytest

from src.metrics import compute_t1_by_side_to_move, load_results_df, save_metrics

WHITE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BLACK = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
TRUTH = [-450, -220, -120, -60, 0, 40, 130, 260, 600]


def _rows(model, fen, flip=False):
    return [{"model": model, "fen": fen, "t1_model_eval": -t if flip else t, "t1_stockfish_eval": t}
            for t in TRUTH]


def _row(table, model):
    return table.set_index("model").loc[model]


def test_white_pov_model_not_flagged():
    df = pd.DataFrame(_rows("good", WHITE) + _rows("good", BLACK))
    row = _row(compute_t1_by_side_to_move(df), "good")
    assert row["spearman_white"] == pytest.approx(1)
    assert row["spearman_black"] == pytest.approx(1)
    assert row["n_white"] == row["n_black"] == len(TRUTH)
    assert not row["perspective_flip_suspect"]


def test_side_to_move_pov_model_flagged():
    df = pd.DataFrame(_rows("flip", WHITE) + _rows("flip", BLACK, flip=True))
    row = _row(compute_t1_by_side_to_move(df), "flip")
    assert row["spearman_black"] == pytest.approx(-1)
    assert row["perspective_flip_suspect"]


def test_one_side_only_is_nan_and_not_flagged():
    row = _row(compute_t1_by_side_to_move(pd.DataFrame(_rows("w", WHITE))), "w")
    assert math.isnan(row["spearman_black"])
    assert not row["perspective_flip_suspect"]


def test_save_metrics_writes_csv(tmp_path):
    records = [
        {"job_id": f"j{i}_{side}", "job_type": "standard", "position_id": i, "fen": fen,
         "model": "qwen2.5:7b", "prompt_format": "eval_only",
         "difficulty": "easy", "phase": "opening", "source": "lichess_puzzles", "inference_ms": 1,
         "t1_model_eval": t, "t1_stockfish_eval": t, "t1_absolute_error": 0, "t1_direction_correct": True,
         "t2_move": None, "t2_legal": False, "t2_cpl": None,
         "t3_score": None, "t3_p1_side_correct": None, "t3_p2_theme_correct": None}
        for side, fen in (("w", WHITE), ("b", BLACK)) for i, t in enumerate(TRUTH)
    ]
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    out = tmp_path / "metrics"
    save_metrics(load_results_df(str(path), data_dir=str(tmp_path / "no-data")), output_dir=str(out))
    assert (out / "t1_by_side_to_move.csv").exists()
