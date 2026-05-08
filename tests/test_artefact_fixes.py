"""Tests for the artefact fixes in src/evaluator.py and src/metrics.py.

Covers:
  - CPL clamp prevents mate-encoding inflation
  - CPL is unchanged in normal eval range
  - camelCase Lichess theme labels match natural-English explanations
  - Existing snake_case theme synonyms still work
  - T1 abs_error_excl_mate drops mate-truth rows
  - WP-loss is bounded [0, 1000]
"""

import math

import chess
import pandas as pd
import pytest

from src.evaluator import (
    EVAL_CLAMP_CP,
    THEME_SYNONYMS,
    _camel_to_words,
    score_t2,
    score_t3,
)
from src.metrics import (
    MATE_SCORE_THRESHOLD_CP,
    compute_clamped_cpl,
    compute_wp_loss,
    rescore_t3_theme,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubEngine:
    """Minimal Stockfish stand-in returning a configured eval_after."""

    def __init__(self, eval_after: int) -> None:
        self.eval_after = eval_after

    def evaluate_after_move(self, fen: str, move_san: str) -> int:
        return self.eval_after


# Position with White to move (standard starting move-1).
WHITE_FEN = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
BLACK_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


# ---------------------------------------------------------------------------
# CPL clamp behaviour
# ---------------------------------------------------------------------------


def test_cpl_clamp_caps_mate_inflation_white_to_move():
    """When eval_after is mate-encoded, raw CPL would explode; clamped CPL ≤ 2*EVAL_CLAMP_CP."""
    # Stockfish saw +50 (small advantage), model walks into mate-against-white
    # encoded as -9990. Raw CPL would be 50 - (-9990) = 10,040.
    engine = _StubEngine(eval_after=-9990)
    result = score_t2(
        model_move="Nf3",
        fen=WHITE_FEN,
        stockfish_best_move="Nf3",  # any legal move; we just need legality
        stockfish_eval=50,
        engine=engine,
    )
    assert result["t2_legal"] is True
    # 50 clamped → 50; -9990 clamped → -1000. CPL = 50 - (-1000) = 1050.
    assert result["t2_cpl"] == 1050
    assert result["t2_cpl"] <= 2 * EVAL_CLAMP_CP


def test_cpl_clamp_caps_mate_inflation_black_to_move():
    """Same artefact, black-to-move side. CPL must still be bounded."""
    engine = _StubEngine(eval_after=9990)  # mate for white = bad for black-to-move
    result = score_t2(
        model_move="e5",
        fen=BLACK_FEN,
        stockfish_best_move="e5",
        stockfish_eval=-50,  # slight black advantage
        engine=engine,
    )
    assert result["t2_legal"] is True
    # -50 clamped → -50; +9990 clamped → +1000.
    # Black-to-move CPL = eval_after - sf = 1000 - (-50) = 1050.
    assert result["t2_cpl"] == 1050


def test_cpl_unchanged_in_normal_range():
    """Within the clamp window, CPL is identical to the unclamped formula."""
    engine = _StubEngine(eval_after=-300)
    result = score_t2(
        model_move="Nf3",
        fen=WHITE_FEN,
        stockfish_best_move="Nf3",
        stockfish_eval=200,
        engine=engine,
    )
    assert result["t2_cpl"] == 500  # 200 - (-300)


def test_cpl_compute_helper_white_to_move():
    """Direct test of the standalone helper."""
    assert compute_clamped_cpl(200, -300, white_to_move=True) == 500
    assert compute_clamped_cpl(50, -9990, white_to_move=True) == 1050  # clamped


def test_cpl_compute_helper_black_to_move():
    assert compute_clamped_cpl(-200, 300, white_to_move=False) == 500
    assert compute_clamped_cpl(-50, 9990, white_to_move=False) == 1050


def test_cpl_negative_floor_to_zero():
    """Depth/horizon noise: model's move can evaluate marginally better; CPL = 0."""
    assert compute_clamped_cpl(100, 110, white_to_move=True) == 0


# ---------------------------------------------------------------------------
# Theme matcher: camelCase + existing synonyms
# ---------------------------------------------------------------------------


def test_camel_to_words():
    assert _camel_to_words("advancedPawn") == "advanced pawn"
    assert _camel_to_words("backRankMate") == "back rank mate"
    assert _camel_to_words("kingsideAttack") == "kingside attack"
    assert _camel_to_words("mate") == "mate"
    assert _camel_to_words("") == ""


def test_camel_to_words_handles_consecutive_caps():
    """xRayAttack should become 'x ray attack', not 'xray attack'."""
    assert _camel_to_words("xRayAttack") == "x ray attack"


def test_theme_match_camelcase_label_advanced_pawn():
    """A model talking about an advanced pawn matches the advancedPawn label."""
    result = score_t3(
        explanation="The advanced pawn on the seventh rank cannot be stopped.",
        side_claimed="White",
        stockfish_eval=400,
        theme="advancedPawn",
    )
    assert result["t3_p2_theme_correct"] == 1


def test_theme_match_camelcase_label_kingside_attack():
    result = score_t3(
        explanation="White has a strong kingside attack and Black's king is exposed.",
        side_claimed="White",
        stockfish_eval=300,
        theme="kingsideAttack",
    )
    assert result["t3_p2_theme_correct"] == 1


def test_theme_match_camelcase_label_back_rank_mate():
    result = score_t3(
        explanation="Black has a back rank weakness allowing mate.",
        side_claimed="White",
        stockfish_eval=600,
        theme="backRankMate",
    )
    assert result["t3_p2_theme_correct"] == 1


def test_theme_match_existing_synonyms_fork():
    """Existing snake_case synonyms still work (fork → 'double attack')."""
    result = score_t3(
        explanation="The knight forks the queen and rook in a double attack.",
        side_claimed="White",
        stockfish_eval=400,
        theme="fork",
    )
    assert result["t3_p2_theme_correct"] == 1


def test_theme_no_match_returns_zero():
    result = score_t3(
        explanation="The position is unclear and Black has good piece activity.",
        side_claimed="Black",
        stockfish_eval=-50,
        theme="advancedPawn",
    )
    assert result["t3_p2_theme_correct"] == 0


def test_rescore_helper_matches_score_t3():
    """The standalone rescore helper agrees with score_t3 on the theme component."""
    explanation = "The kingside attack is decisive."
    assert rescore_t3_theme("kingsideAttack", explanation) == 1
    assert rescore_t3_theme("advancedPawn", explanation) == 0


def test_rescore_helper_handles_none_inputs():
    assert rescore_t3_theme(None, "anything") is None
    assert rescore_t3_theme("fork", None) is None


# ---------------------------------------------------------------------------
# Win-probability loss
# ---------------------------------------------------------------------------


def test_wp_loss_zero_when_no_change():
    assert compute_wp_loss(200, 200, white_to_move=True) == pytest.approx(0)


def test_wp_loss_bounded_above_at_1000():
    """Worst case (full -1000 → +1000 swing) is below 1000 milli-WP."""
    val = compute_wp_loss(1000, -1000, white_to_move=True)
    assert 0 <= val <= 1000


def test_wp_loss_saturates_outside_clamp():
    """A swing from +9000 to -9000 saturates at the clamp window."""
    inside = compute_wp_loss(1000, -1000, white_to_move=True)
    outside = compute_wp_loss(9000, -9000, white_to_move=True)
    assert inside == pytest.approx(outside)


def test_wp_loss_non_negative_on_horizon_noise():
    assert compute_wp_loss(100, 110, white_to_move=True) == 0


# ---------------------------------------------------------------------------
# T1 absolute-error excl. mate
# ---------------------------------------------------------------------------


def test_t1_abs_error_excl_mate_drops_mate_rows():
    """Mate-truth rows become NaN; non-mate rows are unchanged."""
    from src.metrics import load_results_df, MATE_SCORE_THRESHOLD_CP

    # We'll simulate the derived-column logic directly to avoid file I/O.
    df = pd.DataFrame({
        "t1_stockfish_eval": [50, 9999, -200, -12000, 0],
        "t1_absolute_error": [10, 9990, 100, 12000, 5],
        "t1_model_eval":    [60, 0,    -100,  0,     5],
    })

    mate_truth = df["t1_stockfish_eval"].abs() >= MATE_SCORE_THRESHOLD_CP
    df["t1_abs_error_excl_mate"] = df["t1_absolute_error"].where(~mate_truth)

    # Non-mate rows preserved
    assert df.loc[0, "t1_abs_error_excl_mate"] == 10
    assert df.loc[2, "t1_abs_error_excl_mate"] == 100
    assert df.loc[4, "t1_abs_error_excl_mate"] == 5
    # Mate rows NaN'd
    assert pd.isna(df.loc[1, "t1_abs_error_excl_mate"])
    assert pd.isna(df.loc[3, "t1_abs_error_excl_mate"])

    # Mean excludes mate rows
    assert df["t1_abs_error_excl_mate"].mean() == pytest.approx((10 + 100 + 5) / 3)


def test_mate_threshold_value():
    """Sanity check on the constant used for filtering mate-encoded scores."""
    # mate_in=100 plies → 10000 - 1000 = 9000 (boundary)
    # mate_in=1   plies → 10000 - 10  = 9990 (well above)
    # Anything ≥ 9000 is treated as mate.
    assert MATE_SCORE_THRESHOLD_CP == 9000


# ---------------------------------------------------------------------------
# Smoke test: full load_results_df derives expected columns
# ---------------------------------------------------------------------------


def test_load_results_df_adds_derived_columns(tmp_path):
    """End-to-end: write a tiny JSONL + data file, load, check derived columns."""
    import json
    from src.metrics import load_results_df

    # Build a stub data/ directory with one tier
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "easy.json").write_text(json.dumps([
        {"id": 1, "fen": WHITE_FEN, "stockfish_eval": 50,
         "stockfish_best_move": "Nf3", "difficulty": "easy",
         "theme": "advancedPawn", "phase": "middlegame", "source": "x"}
    ]))

    # Build a tiny evaluations.jsonl
    rows = [
        # Non-mate row
        {"job_id": "j1", "model": "m", "position_id": 1, "fen": WHITE_FEN,
         "difficulty": "easy", "theme": "advancedPawn", "phase": "middlegame",
         "prompt_format": "fen_only", "model_family": "m", "model_size_b": 1,
         "t1_model_eval": 100, "t1_stockfish_eval": 50,
         "t1_absolute_error": 50, "t1_direction_correct": True,
         "t2_move": "Nf3", "t2_legal": True, "t2_cpl": 200,
         "t3_explanation": "Advanced pawn is decisive.",
         "t3_side_claimed": "White", "t3_p1_side_correct": 1,
         "t3_p2_theme_correct": 0, "t3_score": 1,
         "inference_ms": 100, "source": "x"},
        # Mate-truth row
        {"job_id": "j2", "model": "m", "position_id": 1, "fen": WHITE_FEN,
         "difficulty": "easy", "theme": "advancedPawn", "phase": "middlegame",
         "prompt_format": "fen_only", "model_family": "m", "model_size_b": 1,
         "t1_model_eval": 200, "t1_stockfish_eval": 9990,
         "t1_absolute_error": 9790, "t1_direction_correct": True,
         "t2_move": "Nf3", "t2_legal": True, "t2_cpl": 9000,
         "t3_explanation": "Position is unclear.",
         "t3_side_claimed": "White", "t3_p1_side_correct": 1,
         "t3_p2_theme_correct": 0, "t3_score": 1,
         "inference_ms": 100, "source": "x"},
    ]
    jsonl = tmp_path / "evals.jsonl"
    with open(jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Reset module-level FEN cache so the test data_dir is read fresh
    import src.metrics as M
    M._FEN_LOOKUP_CACHE = None

    df = load_results_df(str(jsonl), data_dir=str(data_dir))

    # Derived columns present
    for col in (
        "t1_relative_error",
        "t1_abs_error_excl_mate",
        "t2_cpl_clamped",
        "t2_wp_loss",
        "t3_p2_theme_correct_v2",
        "t3_score_v2",
    ):
        assert col in df.columns, f"missing column: {col}"

    # Non-mate row has finite abs_error_excl_mate; mate row is NaN
    assert df.loc[0, "t1_abs_error_excl_mate"] == 50
    assert pd.isna(df.loc[1, "t1_abs_error_excl_mate"])

    # Clamped CPL on the mate-eval row is bounded
    assert df.loc[1, "t2_cpl_clamped"] <= 2 * EVAL_CLAMP_CP

    # The new theme matcher catches "Advanced pawn"
    assert df.loc[0, "t3_p2_theme_correct_v2"] == 1
    assert df.loc[1, "t3_p2_theme_correct_v2"] == 0
