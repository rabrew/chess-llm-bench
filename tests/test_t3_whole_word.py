"""Tests for whole-word T3 theme matching (specs/t3-whole-word-matching.md)."""

import pytest

from src.evaluator import score_t3, theme_mentioned
from src.metrics import rescore_t3_theme

SHOULD_NOT_MATCH = [
    ("mate", "White is better, I estimate about +1.5."),
    ("mate", "This is approximately equal."),
    ("mate", "Ultimately Black holds."),
    ("mate", "White has a material advantage."),
    ("pin", "In my opinion White is better."),
    ("equality", "White will eventually win."),
]

SHOULD_MATCH = [
    ("fork", "The knight forks the king and rook."),
    ("mate", "Black gets mated in two."),
    ("mate", "Checkmate!"),
    ("pin", "Two pins on the e-file decide it."),
    ("pin", "The bishop is pinned."),
    ("kingsideAttack", "White is attacking the king."),
    ("backRankMate", "There is a back-rank mate threat."),
    ("advancedPawn", "The advanced pawn on d7 is decisive."),
    ("mate", "MATE follows."),
]


@pytest.mark.parametrize("theme,text", SHOULD_NOT_MATCH)
def test_no_match_inside_longer_word(theme, text):
    assert theme_mentioned(theme, text) is False


@pytest.mark.parametrize("theme,text", SHOULD_MATCH)
def test_match_whole_word_and_endings(theme, text):
    assert theme_mentioned(theme, text) is True


@pytest.mark.parametrize("theme,text", SHOULD_NOT_MATCH + SHOULD_MATCH)
def test_live_and_rescore_agree(theme, text):
    live = score_t3(text, "White", 300, theme)["t3_p2_theme_correct"]
    assert live == rescore_t3_theme(theme, text)


def test_none_inputs():
    assert theme_mentioned(None, "a pin") is False
    assert theme_mentioned("pin", None) is False
    assert rescore_t3_theme(None, "a pin") is None
    assert rescore_t3_theme("pin", None) is None
