"""Tests for scripts/retry_illegal_moves.py"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import chess
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.retry_illegal_moves import (
    build_retry_prompt,
    get_legal_san_moves,
    is_legal,
    parse_move,
)

# Starting position FEN
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# ---------------------------------------------------------------------------
# build_retry_prompt
# ---------------------------------------------------------------------------

class TestBuildRetryPrompt:
    def test_includes_fen(self):
        prompt = build_retry_prompt(START_FEN, "e5", ["e4", "d4", "Nf3"])
        assert START_FEN in prompt

    def test_includes_legal_moves(self):
        prompt = build_retry_prompt(START_FEN, "e5", ["e4", "d4", "Nf3"])
        assert "e4" in prompt
        assert "d4" in prompt
        assert "Nf3" in prompt

    def test_illegal_move_mentioned(self):
        prompt = build_retry_prompt(START_FEN, "e5", ["e4"])
        assert "e5" in prompt
        assert "not legal" in prompt

    def test_missing_move(self):
        prompt = build_retry_prompt(START_FEN, None, ["e4"])
        assert "did not provide" in prompt

    def test_asks_for_san_only(self):
        prompt = build_retry_prompt(START_FEN, "e5", ["e4"])
        assert "SAN" in prompt or "move" in prompt.lower()


# ---------------------------------------------------------------------------
# parse_move
# ---------------------------------------------------------------------------

class TestParseMove:
    def test_plain_move(self):
        assert parse_move("e4") == "e4"

    def test_strips_punctuation(self):
        assert parse_move("Nf3.") == "Nf3"
        assert parse_move("e4,") == "e4"

    def test_strips_backticks(self):
        assert parse_move("`Nf3`") == "Nf3"

    def test_strips_markdown_bold(self):
        assert parse_move("**e4**") == "e4"

    def test_takes_first_token_only(self):
        assert parse_move("e4 is the best move") == "e4"

    def test_takes_first_line_only(self):
        assert parse_move("Nf3\nsome explanation") == "Nf3"

    def test_strips_move_number_prefix(self):
        assert parse_move("1.e4") == "e4"
        assert parse_move("21.Rxd4") == "Rxd4"

    def test_empty_response(self):
        assert parse_move("") is None
        assert parse_move("   ") is None

    def test_castling(self):
        assert parse_move("O-O") == "O-O"
        assert parse_move("O-O-O") == "O-O-O"


# ---------------------------------------------------------------------------
# get_legal_san_moves
# ---------------------------------------------------------------------------

class TestGetLegalSanMoves:
    def test_starting_position_count(self):
        moves = get_legal_san_moves(START_FEN)
        assert len(moves) == 20  # 16 pawn + 4 knight moves

    def test_starting_position_contains_e4(self):
        moves = get_legal_san_moves(START_FEN)
        assert "e4" in moves

    def test_checkmate_position_no_moves(self):
        # Fool's mate — Black is checkmated
        fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        moves = get_legal_san_moves(fen)
        assert moves == []

    def test_returns_san_not_uci(self):
        moves = get_legal_san_moves(START_FEN)
        # SAN knight move is "Nf3", UCI would be "g1f3"
        assert "Nf3" in moves
        assert "g1f3" not in moves


# ---------------------------------------------------------------------------
# is_legal
# ---------------------------------------------------------------------------

class TestIsLegal:
    def test_legal_move(self):
        assert is_legal(START_FEN, "e4") is True
        assert is_legal(START_FEN, "Nf3") is True

    def test_illegal_move(self):
        assert is_legal(START_FEN, "e5") is False   # pawn can't go there from start
        assert is_legal(START_FEN, "Qd4") is False  # queen blocked

    def test_none_move(self):
        assert is_legal(START_FEN, None) is False

    def test_garbage_move(self):
        assert is_legal(START_FEN, "zzz") is False
        assert is_legal(START_FEN, "") is False

    def test_bad_fen(self):
        assert is_legal("not-a-fen", "e4") is False
