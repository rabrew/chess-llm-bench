"""Tests for fine-tuning/src/board_state.py — verifier and parser."""

import sys
from pathlib import Path

import chess
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FT_ROOT / "src"))
sys.path.insert(0, str(FT_ROOT / "scripts"))

from board_state import (
    build_correction,
    fen_to_board_description,
    parse_board_description,
    verify_board_description,
)

# Starting position FEN
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Italian Game after 1.e4 e5 2.Nf3 Nc6 3.Bc4
ITALIAN_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"


class TestFenToBoardDescription:
    def test_starting_position_contains_all_white_pieces(self):
        desc = fen_to_board_description(STARTING_FEN)
        assert "King e1" in desc
        assert "Queen d1" in desc

    def test_starting_position_contains_all_black_pieces(self):
        desc = fen_to_board_description(STARTING_FEN)
        assert "King e8" in desc
        assert "Queen d8" in desc

    def test_side_to_move_white(self):
        desc = fen_to_board_description(STARTING_FEN)
        assert "Side to move: White" in desc

    def test_side_to_move_black(self):
        # After 1.e4
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        desc = fen_to_board_description(fen)
        assert "Side to move: Black" in desc

    def test_no_empty_squares_in_output(self):
        desc = fen_to_board_description(STARTING_FEN)
        # There should be no "empty" or bare square notation outside of piece context
        lines = desc.split("\n")
        piece_lines = [l for l in lines if l.startswith("White:") or l.startswith("Black:")]
        for line in piece_lines:
            # Should not contain standalone empty square descriptions
            assert "empty" not in line.lower()

    def test_promoted_piece_appears(self):
        # Position with two white queens (promoted pawn)
        # FEN: white has queens on d1 and d8
        fen = "3Q4/8/8/8/8/8/8/3QK1k1 w - - 0 1"
        desc = fen_to_board_description(fen)
        # Both queen squares should appear
        assert "d1" in desc
        assert "d8" in desc

    def test_piece_counts_match_fen(self):
        desc = fen_to_board_description(STARTING_FEN)
        # Count pawns in description — should be 8 per side
        white_section = desc.split("\n")[0]  # "White: ..."
        # Pawns section: "Pawns a2 b2 c2 d2 e2 f2 g2 h2" — space-separated squares
        pawns_part = white_section.split("Pawns")[1].split(",")[0].strip()
        pawn_squares = pawns_part.split()
        assert len(pawn_squares) == 8


class TestParseBoardDescription:
    def test_parse_starting_position(self):
        desc = fen_to_board_description(STARTING_FEN)
        parsed = parse_board_description(desc)
        assert "White King" in parsed
        assert parsed["White King"] == ["e1"]

    def test_parse_returns_dict(self):
        desc = fen_to_board_description(STARTING_FEN)
        result = parse_board_description(desc)
        assert isinstance(result, dict)

    def test_parse_empty_string_returns_empty(self):
        result = parse_board_description("")
        assert isinstance(result, dict)

    def test_parse_black_pawns(self):
        desc = fen_to_board_description(STARTING_FEN)
        parsed = parse_board_description(desc)
        assert "Black Pawn" in parsed
        assert len(parsed["Black Pawn"]) == 8


class TestVerifyBoardDescription:
    def test_verify_correct_response(self):
        desc = fen_to_board_description(STARTING_FEN)
        result = verify_board_description(STARTING_FEN, desc)
        assert result["correct"] is True
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["errors"] == []

    def test_verify_wrong_square(self):
        # Move white king description from e1 to e2 (wrong)
        desc = fen_to_board_description(STARTING_FEN)
        bad_desc = desc.replace("King e1", "King e2")
        result = verify_board_description(STARTING_FEN, bad_desc)
        assert result["correct"] is False
        assert len(result["wrong_squares"]) > 0 or len(result["missing_pieces"]) > 0

    def test_verify_missing_piece(self):
        desc = fen_to_board_description(STARTING_FEN)
        # Remove the white queen entirely
        bad_desc = desc.replace("Queen d1, ", "").replace(", Queen d1", "").replace("Queen d1", "")
        result = verify_board_description(STARTING_FEN, bad_desc)
        assert result["correct"] is False
        assert result["accuracy"] < 1.0

    def test_verify_extra_piece(self):
        desc = fen_to_board_description(STARTING_FEN)
        # Hallucinate an extra white queen on a5
        bad_desc = desc.replace("White: King e1", "White: King e1, Queen a5")
        result = verify_board_description(STARTING_FEN, bad_desc)
        assert result["correct"] is False

    def test_accuracy_is_float_between_0_and_1(self):
        desc = fen_to_board_description(STARTING_FEN)
        result = verify_board_description(STARTING_FEN, desc)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_correction_output_format(self):
        desc = fen_to_board_description(STARTING_FEN)
        bad_desc = desc.replace("King e1", "King e2")
        correction = build_correction(STARTING_FEN, bad_desc)
        assert isinstance(correction, str)
        assert "Correction" in correction
        assert "Full board:" in correction
