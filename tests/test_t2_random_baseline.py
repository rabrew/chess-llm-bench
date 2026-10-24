"""Best-move rate against a per-position random baseline (specs/t2-random-baseline.md, T2-3)."""

import json

import chess
import pandas as pd
import pytest

from src.metrics import compute_best_move_vs_random, load_results_df, save_metrics

START = chess.STARTING_FEN  # 20 legal moves
# White king h1 in check from a rook on a1, only Kh2 / Kg2 escape squares covered except one.
ONE_MOVE = "8/8/8/8/8/6k1/r7/7K w - - 0 1"


def _n_legal(fen):
    return len(list(chess.Board(fen).legal_moves))


def _df(rows):
    return pd.DataFrame([
        {"model": model, "prompt_format": fmt, "fen": fen, "t2_move": move, "t2_best_move": best}
        for model, fmt, fen, move, best in rows
    ])


def _only(table):
    assert len(table) == 1
    return table.iloc[0]


def test_one_move_fixture_really_has_one_legal_move():
    assert _n_legal(ONE_MOVE) == 1


def test_random_rate_is_mean_of_reciprocals_not_reciprocal_of_mean():
    only_move = next(iter(chess.Board(ONE_MOVE).legal_moves))
    row = _only(compute_best_move_vs_random(_df([
        ("m", "move_only", START, "e4", "e4"),
        ("m", "move_only", ONE_MOVE, "Kg1", chess.Board(ONE_MOVE).san(only_move)),
    ])))
    expected = (1 / _n_legal(START) + 1 / _n_legal(ONE_MOVE)) / 2
    assert row["random_best_move_rate"] == pytest.approx(expected)
    assert row["random_best_move_rate"] != pytest.approx(2 / (_n_legal(START) + _n_legal(ONE_MOVE)))
    assert row["mean_legal_moves"] == pytest.approx((_n_legal(START) + _n_legal(ONE_MOVE)) / 2)


def test_best_move_rate_is_exact_san_match():
    row = _only(compute_best_move_vs_random(_df([
        ("m", "fen_only", START, "e4", "e4"),
        ("m", "fen_only", START, "d4", "e4"),
        ("m", "fen_only", START, "Nf3", "e4"),
        ("m", "fen_only", START, "e4", "e4"),
    ])))
    assert row["n"] == 4
    assert row["best_move_rate"] == pytest.approx(0.5)
    assert row["ratio_vs_random"] == pytest.approx(0.5 / row["random_best_move_rate"])


def test_excludes_rows_without_best_move_and_non_move_prompts():
    row = _only(compute_best_move_vs_random(_df([
        ("m", "move_only", START, "e4", "e4"),
        ("m", "move_only", START, None, None),
        ("m", "eval_only", START, "e4", "e4"),
    ])))
    assert row["n"] == 1


def test_bad_fen_dropped_not_raised():
    row = _only(compute_best_move_vs_random(_df([
        ("m", "move_only", START, "e4", "e4"),
        ("m", "move_only", "not a fen", "e4", "e4"),
    ])))
    assert row["n"] == 1


def test_model_without_rows_omitted_and_empty_input():
    table = compute_best_move_vs_random(_df([
        ("a", "move_only", START, "e4", "e4"),
        ("b", "move_only", START, None, None),
    ]))
    assert list(table["model"]) == ["a"]
    assert compute_best_move_vs_random(_df([("b", "move_only", START, None, None)])).empty


def test_save_metrics_writes_csv_and_summary(tmp_path):
    record = {
        "job_id": "j", "job_type": "standard", "position_id": 1, "fen": START,
        "model": "qwen2.5:7b", "prompt_format": "move_only",
        "difficulty": "easy", "phase": "opening", "source": "lichess_puzzles",
        "inference_ms": 100, "t1_model_eval": None, "t1_stockfish_eval": 0,
        "t1_absolute_error": None, "t1_direction_correct": None,
        "t2_move": "e4", "t2_best_move": "e4", "t2_legal": True, "t2_cpl": 0,
        "t3_score": None, "t3_p1_side_correct": None, "t3_p2_theme_correct": None,
    }
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps(record) + "\n")
    out = tmp_path / "metrics"
    save_metrics(load_results_df(str(path), data_dir=str(tmp_path / "no-data")), output_dir=str(out))
    assert (out / "t2_best_move_vs_random.csv").exists()
    pooled = json.loads((out / "summary.json").read_text())["overall_metrics"]["t2_best_move_vs_random"]
    assert pooled["best_move_rate"] == 1
    assert pooled["random_best_move_rate"] == pytest.approx(1 / 20)
