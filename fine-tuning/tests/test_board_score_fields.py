"""F-8: extra score fields from verify_board_description (specs/board-reading-test.md)."""

import sys
from pathlib import Path

import pytest

FT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FT_ROOT / "src"))

from board_state import fen_to_board_description, verify_board_description  # noqa: E402

KINGS_FEN = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
PERFECT = "White: King e1\nBlack: King e8\nSide to move: White"


def test_perfect_board():
    result = verify_board_description(KINGS_FEN, PERFECT)
    assert result["correct"] is True
    assert (result["recall"], result["precision"]) == (1.0, 1.0)
    assert (result["missing_count"], result["extra_count"], result["total_pieces"]) == (0, 0, 2)
    assert result["side_to_move_correct"] is True


def test_invented_piece_lowers_precision_not_recall():
    result = verify_board_description(KINGS_FEN, "White: King e1, Queen d1\nBlack: King e8")
    assert result["accuracy"] == result["recall"] == 1.0
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["extra_count"] == 1
    assert result["correct"] is False


def test_wrong_square_counts_missing_and_extra():
    result = verify_board_description(KINGS_FEN, "White: King e2\nBlack: King e8")
    assert (result["missing_count"], result["extra_count"]) == (1, 1)
    assert result["recall"] == 0.5


def test_nothing_listed():
    result = verify_board_description(KINGS_FEN, "I am not sure.")
    assert (result["recall"], result["precision"]) == (0.0, 0.0)
    assert result["missing_count"] == 2
    assert result["side_to_move_correct"] is None


def test_wrong_side_to_move():
    result = verify_board_description(KINGS_FEN, "White: King e1\nBlack: King e8\nSide to move: Black")
    assert result["side_to_move_correct"] is False


def test_generated_description_is_perfect():
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    result = verify_board_description(fen, fen_to_board_description(fen))
    assert result["correct"] and result["precision"] == 1.0 and result["side_to_move_correct"]
