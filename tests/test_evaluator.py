"""Tests for T1/T2/T3 scoring logic."""

import pytest

from src.evaluator import (
    get_direction,
    score_t1,
    score_t2,
    score_t3,
    score_all,
    should_trigger_correction,
    THEME_SYNONYMS,
)


class TestGetDirection:
    def test_white_winning(self):
        assert get_direction(100) == "White"
        assert get_direction(51) == "White"

    def test_black_winning(self):
        assert get_direction(-100) == "Black"
        assert get_direction(-51) == "Black"

    def test_equal(self):
        assert get_direction(0) == "Equal"
        assert get_direction(50) == "Equal"
        assert get_direction(-50) == "Equal"


class TestScoreT1:
    def test_exact_match(self):
        result = score_t1(100, 100)
        assert result["t1_absolute_error"] == 0
        assert result["t1_direction_correct"] is True

    def test_error_calculation(self):
        result = score_t1(150, 100)
        assert result["t1_absolute_error"] == 50

    def test_direction_wrong(self):
        result = score_t1(100, -100)
        assert result["t1_direction_correct"] is False

    def test_clamping(self):
        result = score_t1(5000, 100, eval_range=(-2000, 2000))
        assert result["t1_model_eval"] == 2000
        assert result["t1_absolute_error"] == 1900

    def test_none_eval(self):
        result = score_t1(None, 100)
        assert result["t1_model_eval"] is None
        assert result["t1_absolute_error"] is None


class TestScoreT2:
    def test_legal_move(self):
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
        result = score_t2("Nf6", fen, "Nf6", 50)
        assert result["t2_legal"] is True
        assert result["t2_move"] == "Nf6"

    def test_illegal_move(self):
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
        result = score_t2("Qa5", fen, "Nf6", 50)  # Illegal: queen path d8->c7 blocked by own pawn
        assert result["t2_legal"] is False

    def test_none_move(self):
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
        result = score_t2(None, fen, "Nf6", 50)
        assert result["t2_move"] is None
        assert result["t2_legal"] is False


class TestScoreT3:
    def test_correct_side_and_theme(self):
        result = score_t3(
            explanation="White is better due to the pin on the knight.",
            side_claimed="White",
            stockfish_eval=100,
            theme="pin",
        )
        assert result["t3_p1_side_correct"] == 1
        assert result["t3_p2_theme_correct"] == 1
        assert result["t3_score"] == 2

    def test_wrong_side(self):
        result = score_t3(
            explanation="Black is better due to the pin.",
            side_claimed="Black",
            stockfish_eval=100,  # White is better
            theme="pin",
        )
        assert result["t3_p1_side_correct"] == 0
        assert result["t3_p2_theme_correct"] == 1
        assert result["t3_score"] == 1

    def test_wrong_theme(self):
        result = score_t3(
            explanation="White is better due to better piece activity.",
            side_claimed="White",
            stockfish_eval=100,
            theme="fork",  # No fork mentioned
        )
        assert result["t3_p1_side_correct"] == 1
        assert result["t3_p2_theme_correct"] == 0
        assert result["t3_score"] == 1

    def test_synonym_detection(self):
        result = score_t3(
            explanation="White has a strong double attack threat.",
            side_claimed="White",
            stockfish_eval=100,
            theme="fork",  # "double attack" is a synonym
        )
        assert result["t3_p2_theme_correct"] == 1

    def test_none_explanation(self):
        result = score_t3(None, None, 100, "pin")
        assert result["t3_score"] is None


class TestShouldTriggerCorrection:
    def test_above_threshold(self):
        assert should_trigger_correction(100, threshold=50) is True

    def test_below_threshold(self):
        assert should_trigger_correction(30, threshold=50) is False

    def test_none_cpl(self):
        assert should_trigger_correction(None, threshold=50) is False
