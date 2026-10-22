"""T2 legal rate counts blanked illegal moves (specs/t1-zero-baseline-and-ft-comparison.md §4)."""

import json
import math

import pytest

from src.metrics import load_results_df

FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"


def _record(fmt, move, legal):
    return {
        "job_id": f"{fmt}_{move}", "job_type": "standard", "position_id": 1, "fen": FEN,
        "model": "llama3.2:3b", "prompt_format": fmt,
        "t1_model_eval": None, "t1_stockfish_eval": 0, "t1_absolute_error": None,
        "t2_move": move, "t2_legal": legal, "t2_cpl": None, "t3_score": None,
    }


def _load(tmp_path, records, drop_format=False):
    if drop_format:
        records = [{k: v for k, v in r.items() if k != "prompt_format"} for r in records]
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return load_results_df(str(path), data_dir=str(tmp_path / "no-data"))


@pytest.mark.parametrize("fmt", ["fen_only", "pgn+fen", "cot", "move_only"])
def test_blanked_illegal_move_counts_as_illegal(tmp_path, fmt):
    df = _load(tmp_path, [_record(fmt, None, False)])
    assert df["t2_legal_attempted"].iloc[0] == 0


@pytest.mark.parametrize("fmt", ["eval_only", "explanation_only"])
def test_prompts_without_a_move_are_nan(tmp_path, fmt):
    df = _load(tmp_path, [_record(fmt, None, False)])
    assert math.isnan(df["t2_legal_attempted"].iloc[0])


def test_legal_move_counts_as_legal(tmp_path):
    df = _load(tmp_path, [_record("move_only", "Nf6", True)])
    assert df["t2_legal_attempted"].iloc[0] == 1


def test_rate_over_mixed_rows(tmp_path):
    records = [
        _record("move_only", "Nf6", True),
        _record("fen_only", None, False),
        _record("eval_only", None, False),
        _record("explanation_only", None, False),
    ]
    assert _load(tmp_path, records)["t2_legal_attempted"].mean() == pytest.approx(0.5)


def test_without_prompt_format_falls_back_to_move_presence(tmp_path):
    records = [_record("move_only", "Nf6", True), _record("fen_only", None, False)]
    df = _load(tmp_path, records, drop_format=True)
    assert df["t2_legal_attempted"].iloc[0] == 1
    assert math.isnan(df["t2_legal_attempted"].iloc[1])
