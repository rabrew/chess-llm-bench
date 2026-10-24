"""Lichess puzzle metadata joined back to benchmark positions (specs/puzzle-metadata-join.md)."""

import json
import math
import sys
from pathlib import Path

import chess
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import build_puzzle_meta as bpm  # noqa: E402
from src.metrics import compute_puzzle_meta_metrics, load_results_df, save_metrics  # noqa: E402

FEN_A = chess.STARTING_FEN
FEN_B = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


def _row(fen, moves, themes, puzzle_id="p1", rating="1500"):
    return {"PuzzleId": puzzle_id, "FEN": fen, "Moves": moves, "Rating": rating, "Themes": themes}


# ---------- build_meta ----------

def test_build_meta_keeps_matched_fens_with_full_themes():
    rows = [_row(FEN_A, "e2e4 e7e5", "crushing fork short"), _row("8/8/8/8/8/8/8/K6k w - - 0 1", "a1a2", "mate")]
    meta = bpm.build_meta(rows, {FEN_A})
    entry = meta[FEN_A]
    assert entry["themes_all"] == ["crushing", "fork", "short"]
    assert entry["setup_move_uci"] == "e2e4"
    assert entry["setup_move_san"] == "e4"
    assert entry["puzzle_fen"] == FEN_B
    assert entry["rating"] == 1500
    assert set(meta) == {FEN_A}


def test_duplicate_fen_keeps_first():
    rows = [_row(FEN_A, "e2e4", "fork", puzzle_id="first"), _row(FEN_A, "d2d4", "pin", puzzle_id="second")]
    assert bpm.build_meta(rows, {FEN_A})[FEN_A]["puzzle_id"] == "first"


def test_illegal_setup_move_kept_without_san():
    entry = bpm.build_meta([_row(FEN_A, "e2e5", "fork")], {FEN_A})[FEN_A]
    assert entry["setup_move_san"] is None
    assert entry["puzzle_fen"] is None


def test_low_coverage_raises():
    with pytest.raises(ValueError):
        bpm.build_meta([_row(FEN_A, "e2e4", "fork")], {FEN_A, FEN_B})


def test_main_missing_csv_exits_1(tmp_path, monkeypatch):
    results = tmp_path / "r.jsonl"
    results.write_text(json.dumps({"fen": FEN_A}) + "\n")
    monkeypatch.setattr(sys, "argv", ["bpm", "--csv", str(tmp_path / "nope.csv.zst"),
                                      "--results", str(results), "--out", str(tmp_path / "m.json")])
    with pytest.raises(SystemExit) as exc:
        bpm.main()
    assert exc.value.code == 1


# ---------- load_results_df + metrics ----------

def _record(job, move, legal, explanation=None, fmt="fen_only"):
    return {
        "job_id": job, "job_type": "standard", "position_id": 1, "fen": FEN_A,
        "model": "qwen2.5:7b", "prompt_format": fmt,
        "difficulty": "easy", "phase": "opening", "source": "lichess_puzzles", "inference_ms": 1,
        "t1_model_eval": None, "t1_stockfish_eval": 0, "t1_absolute_error": None, "t1_direction_correct": None,
        "t2_move": move, "t2_legal": legal, "t2_cpl": None, "theme": "crushing",
        "t3_explanation": explanation, "t3_score": None, "t3_p1_side_correct": None, "t3_p2_theme_correct": None,
    }


def _load(tmp_path, records, themes="crushing fork short", with_meta=True):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    if with_meta:
        meta = bpm.build_meta([_row(FEN_A, "e2e4 e7e5", themes)], {FEN_A})
        (data_dir / bpm.META_FILENAME).write_text(json.dumps({"_source": "test", "positions": meta}))
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return load_results_df(str(path), data_dir=str(data_dir))


def test_setup_move_rate_cases(tmp_path):
    df = _load(tmp_path, [_record("a", "e4", True), _record("b", "d4", True), _record("c", None, False)])
    assert list(df["t2_played_setup_move"][:2]) == [1.0, 0.0]
    assert math.isnan(df["t2_played_setup_move"].iloc[2])


def test_tactical_theme_credits_any_tactical_theme_and_ignores_crushing(tmp_path):
    df = _load(tmp_path, [
        _record("a", None, False, "White — the knight fork wins material.", fmt="explanation_only"),
        _record("b", None, False, "White — White is crushing.", fmt="explanation_only"),
    ])
    assert list(df["t3_tactical_theme_correct"]) == [1.0, 0.0]


def test_no_tactical_theme_in_list_is_nan(tmp_path):
    df = _load(tmp_path, [_record("a", None, False, "White is crushing.", fmt="explanation_only")],
               themes="crushing short")
    assert math.isnan(df["t3_tactical_theme_correct"].iloc[0])


def test_without_meta_file_columns_absent_and_summary_none(tmp_path):
    df = _load(tmp_path, [_record("a", "e4", True)], with_meta=False)
    assert "t2_played_setup_move" not in df.columns
    out = tmp_path / "metrics"
    save_metrics(df, output_dir=str(out))
    assert json.loads((out / "summary.json").read_text())["overall_metrics"]["puzzle_meta"] is None


def test_metrics_table_and_summary(tmp_path):
    df = _load(tmp_path, [
        _record("a", "e4", True), _record("b", "d4", True),
        _record("c", None, False, "White — a fork.", fmt="explanation_only"),
    ])
    row = compute_puzzle_meta_metrics(df).iloc[0]
    assert row["n_moves"] == 2
    assert row["setup_move_rate"] == pytest.approx(0.5)
    assert row["n_tactical_explanations"] == 1
    assert row["tactical_theme_rate"] == pytest.approx(1.0)
    out = tmp_path / "metrics"
    save_metrics(df, output_dir=str(out))
    assert (out / "puzzle_meta_metrics.csv").exists()
    assert json.loads((out / "summary.json").read_text())["overall_metrics"]["puzzle_meta"]["setup_move_rate"] == 0.5
