"""Pieces + legal moves dataset (fine-tuning/specs/legal-moves-finetuning.md)."""

import json
import sys
from pathlib import Path

import chess
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FT_ROOT / "src"))
sys.path.insert(0, str(FT_ROOT / "scripts"))

from board_state import (  # noqa: E402
    fen_to_board_and_legal_description,
    fen_to_board_description,
    legal_moves_san,
    parse_board_description,
    verify_board_description,
    verify_legal_moves,
)

ITALIAN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
FOOLS_MATE = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
STALEMATE = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
PROMOTION = "8/P6k/8/8/8/8/8/K7 w - - 0 1"


def _truth(fen):
    board = chess.Board(fen)
    return {board.san(m) for m in board.legal_moves}


@pytest.mark.parametrize("fen", [chess.STARTING_FEN, ITALIAN, PROMOTION])
def test_legal_moves_match_python_chess_and_are_sorted(fen):
    moves = legal_moves_san(fen)
    assert set(moves) == _truth(fen)
    assert moves == sorted(moves)


def test_promotion_san_included():
    assert "a8=Q" in legal_moves_san(PROMOTION)


@pytest.mark.parametrize("fen", [FOOLS_MATE, STALEMATE])
def test_no_legal_moves_written_as_none(fen):
    assert fen_to_board_and_legal_description(fen).endswith("Legal moves: none")


def test_description_is_board_then_legal_line():
    text = fen_to_board_and_legal_description(ITALIAN)
    board_part, legal_line = text.rsplit("\n", 1)
    assert board_part == fen_to_board_description(ITALIAN)
    assert legal_line == "Legal moves: " + ", ".join(legal_moves_san(ITALIAN))


def test_board_parser_ignores_legal_line():
    text = fen_to_board_and_legal_description(ITALIAN)
    assert parse_board_description(text) == parse_board_description(fen_to_board_description(ITALIAN))
    assert verify_board_description(ITALIAN, text)["accuracy"] == 1.0


def test_verify_legal_moves_perfect_any_order():
    moves = legal_moves_san(ITALIAN)
    result = verify_legal_moves(ITALIAN, "Side to move: Black\nLegal moves: " + ", ".join(reversed(moves)))
    assert result["recall"] == 1.0 and result["precision"] == 1.0


def test_verify_legal_moves_missing_and_invented():
    moves = legal_moves_san(ITALIAN)
    text = "Legal moves: " + ", ".join(moves[1:] + ["Qxh8"])
    result = verify_legal_moves(ITALIAN, text)
    assert result["missing"] == [moves[0]]
    assert result["extra"] == ["Qxh8"]
    assert result["recall"] == pytest.approx((len(moves) - 1) / len(moves))
    assert result["precision"] == pytest.approx((len(moves) - 1) / len(moves))


def test_verify_legal_moves_without_line():
    result = verify_legal_moves(ITALIAN, "White: King e1")
    assert result["found"] is False
    assert result["recall"] == 0.0


def test_verify_none_on_stalemate():
    assert verify_legal_moves(STALEMATE, "Legal moves: none")["recall"] == 1.0
    assert verify_legal_moves(STALEMATE, "Legal moves: Kg8")["precision"] == 0.0


# ---------- dataset generation ----------

def _board_dataset(tmp_path, fens):
    from generate_board_dataset import INSTRUCTION
    path = tmp_path / "board.jsonl"
    path.write_text("".join(json.dumps({"instruction": INSTRUCTION, "input": f"FEN: {fen}",
                                        "output": fen_to_board_description(fen)}) + "\n" for fen in fens))
    return path


def test_from_dataset_keeps_order_and_rebuilds_targets(tmp_path):
    from generate_board_dataset import LEGAL_INSTRUCTION, records_from_dataset
    fens = [ITALIAN, chess.STARTING_FEN, PROMOTION]
    records = records_from_dataset(_board_dataset(tmp_path, fens), task="board+legal")
    assert [r["input"] for r in records] == [f"FEN: {fen}" for fen in fens]
    assert all(r["instruction"] == LEGAL_INSTRUCTION for r in records)
    assert records[0]["output"] == fen_to_board_and_legal_description(ITALIAN)


def test_from_dataset_board_task_reproduces_original(tmp_path):
    from generate_board_dataset import records_from_dataset
    path = _board_dataset(tmp_path, [ITALIAN, PROMOTION])
    original = [json.loads(line) for line in path.read_text().splitlines()]
    assert records_from_dataset(path, task="board") == original


def test_from_dataset_bad_fen_raises(tmp_path):
    from generate_board_dataset import records_from_dataset
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"instruction": "i", "input": "FEN: not a fen", "output": "o"}) + "\n")
    with pytest.raises(ValueError, match="line 1"):
        records_from_dataset(path, task="board+legal")


def test_same_held_out_split_as_board_dataset(tmp_path):
    pytest.importorskip("transformers")
    from finetune_model import split_held_out
    from generate_board_dataset import records_from_dataset
    fens = [chess.Board().fen()]
    board = chess.Board()
    for move in ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d4", "exd4"]:
        board.push_san(move)
        fens.append(board.fen())
    path = _board_dataset(tmp_path, fens)
    original = [json.loads(line) for line in path.read_text().splitlines()]
    legal = records_from_dataset(path, task="board+legal")
    held_board = [r["input"] for r in split_held_out(original, n=3)[1]]
    held_legal = [r["input"] for r in split_held_out(legal, n=3)[1]]
    assert held_board == held_legal


def test_real_dataset_targets_are_exact():
    dataset = REPO_ROOT / "data" / "board_legal_dataset.jsonl"
    if not dataset.exists():
        pytest.skip("legal-moves dataset not generated yet")
    with open(dataset) as handle:
        for line_no, line in enumerate(handle):
            if line_no >= 500:
                break
            record = json.loads(line)
            fen = record["input"].removeprefix("FEN: ")
            assert verify_legal_moves(fen, record["output"])["recall"] == 1.0
            assert verify_legal_moves(fen, record["output"])["precision"] == 1.0
