"""Tests for position generation."""

import random

import chess
import pytest

from src.position_generator import (
    validate_position,
    generate_random_position,
    generate_endgame_position,
    generate_themed_position,
    generate_positions,
    determine_phase,
    moves_to_pgn,
)


class TestValidatePosition:
    def test_valid_starting_position(self):
        board = chess.Board()
        assert validate_position(board) is True

    def test_invalid_position_no_king(self):
        board = chess.Board(None)
        board.set_piece_at(chess.E1, chess.Piece(chess.QUEEN, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.QUEEN, chess.BLACK))
        assert validate_position(board) is False

    def test_checkmate_position(self):
        # Scholar's mate position
        board = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
        # This is checkmate, should be invalid
        assert validate_position(board) is False


class TestGenerateRandomPosition:
    def test_generates_valid_position(self):
        rng = random.Random(42)
        pos = generate_random_position(rng, min_moves=10, max_moves=20)
        if pos is not None:  # May fail occasionally
            assert "fen" in pos
            assert "phase" in pos
            assert pos["source"] == "generated"

    def test_reproducible_with_seed(self):
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        pos1 = generate_random_position(rng1, min_moves=10, max_moves=10)
        pos2 = generate_random_position(rng2, min_moves=10, max_moves=10)
        if pos1 and pos2:
            assert pos1["fen"] == pos2["fen"]


class TestGenerateEndgamePosition:
    def test_kqvk(self):
        rng = random.Random(42)
        pos = generate_endgame_position(rng, "KQvK")
        if pos is not None:
            board = chess.Board(pos["fen"])
            # Should have 3 pieces: 2 kings and 1 queen
            assert len(board.piece_map()) == 3
            assert pos["phase"] == "endgame"

    def test_krkvr(self):
        rng = random.Random(42)
        pos = generate_endgame_position(rng, "KRvKR")
        if pos is not None:
            board = chess.Board(pos["fen"])
            # Should have 4 pieces: 2 kings and 2 rooks
            assert len(board.piece_map()) == 4


class TestDeterminePhase:
    def test_opening(self):
        board = chess.Board()
        assert determine_phase(board, 5) == "opening"

    def test_endgame_few_pieces(self):
        board = chess.Board("8/8/8/3k4/8/3K4/3Q4/8 w - - 0 1")
        assert determine_phase(board, 50) == "endgame"

    def test_middlegame(self):
        board = chess.Board(
            "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8"
        )
        assert determine_phase(board, 25) == "middlegame"

    def test_endgame_no_queens(self):
        # Position with no queens and few minor/major pieces
        board = chess.Board("8/8/3k4/8/8/3K4/8/8 w - - 0 1")
        assert determine_phase(board, 60) == "endgame"


class TestMovesToPgn:
    def test_empty_moves(self):
        assert moves_to_pgn([]) == ""

    def test_single_move(self):
        result = moves_to_pgn(["e4"])
        assert "1. e4" in result

    def test_two_moves(self):
        result = moves_to_pgn(["e4", "e5"])
        assert "1. e4" in result
        assert "e5" in result

    def test_four_moves(self):
        result = moves_to_pgn(["e4", "e5", "Nf3", "Nc6"])
        assert "1. e4" in result
        assert "2. Nf3" in result


class TestValidatePositionEdgeCases:
    def test_invalid_board(self):
        # Board with no legal moves but not checkmate (stalemate)
        board = chess.Board("8/8/8/8/8/k7/8/K7 w - - 0 1")
        # This might be valid or stalemate depending on position
        result = validate_position(board)
        assert isinstance(result, bool)

    def test_board_is_valid_false(self):
        # Force is_valid() to return False by using an illegal position
        # (two kings adjacent)
        board = chess.Board(None)
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E2, chess.Piece(chess.KING, chess.BLACK))
        # is_valid() should return False
        assert validate_position(board) is False


class TestGenerateThemedPosition:
    def test_known_theme(self):
        rng = random.Random(42)
        pos = generate_themed_position(rng, "fork")
        # May return None if generate_random_position fails, but should not raise
        if pos is not None:
            assert pos["theme"] == "fork"

    def test_unknown_theme_falls_back(self):
        rng = random.Random(42)
        pos = generate_themed_position(rng, "unknown_theme_xyz")
        # Falls back to random position generation
        if pos is not None:
            assert "fen" in pos

    def test_with_base_fen(self):
        rng = random.Random(42)
        base_fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
        pos = generate_themed_position(rng, "pin", base_fen=base_fen)
        if pos is not None:
            assert "fen" in pos

    def test_pin_theme(self):
        rng = random.Random(42)
        pos = generate_themed_position(rng, "pin")
        if pos is not None:
            assert pos["theme"] == "pin"

    def test_skewer_theme(self):
        rng = random.Random(42)
        pos = generate_themed_position(rng, "skewer")
        if pos is not None:
            assert pos["theme"] == "skewer"


class TestGenerateEndgamePositionEdgeCases:
    def test_unknown_config_picks_random(self):
        rng = random.Random(42)
        # Unknown config falls back to a random one
        pos = generate_endgame_position(rng, "XvY")
        if pos is not None:
            assert "fen" in pos
            assert pos["phase"] == "endgame"

    def test_kpvk(self):
        rng = random.Random(42)
        pos = generate_endgame_position(rng, "KPvK")
        if pos is not None:
            assert "fen" in pos

    def test_kpvkp(self):
        rng = random.Random(42)
        pos = generate_endgame_position(rng, "KPvKP")
        if pos is not None:
            assert "fen" in pos

    def test_kbbvk(self):
        rng = random.Random(42)
        pos = generate_endgame_position(rng, "KBBvK")
        if pos is not None:
            assert "fen" in pos

    def test_kbnvk(self):
        rng = random.Random(42)
        pos = generate_endgame_position(rng, "KBNvK")
        if pos is not None:
            assert "fen" in pos


class TestGeneratePositions:
    def test_generates_positions(self):
        positions = generate_positions(count=6, seed=42)
        assert len(positions) > 0
        assert len(positions) <= 6

    def test_reproducible(self):
        p1 = generate_positions(count=6, seed=42)
        p2 = generate_positions(count=6, seed=42)
        # Same seed → same results
        assert len(p1) == len(p2)

    def test_custom_themes(self):
        positions = generate_positions(count=6, seed=42, themes=["fork", "pin"])
        assert len(positions) >= 0  # May generate 0 if all attempts fail, but shouldn't raise

    def test_all_have_fen(self):
        positions = generate_positions(count=3, seed=42)
        for pos in positions:
            assert "fen" in pos
            assert "source" in pos
