"""Scoring logic for T1, T2, and T3 benchmark tasks."""

import logging
from typing import Any

import chess

from .utils import clamp

logger = logging.getLogger("chess_llm_bench")


# Centipawn clamp applied to both endpoints before computing CPL.
#
# Without this clamp, mate-encoded scores (engine_wrapper encodes mate-in-N as
# ±10000 - mate_in*10) leak into the CPL difference, producing values up to
# 20,000 cp from a single move when the model walks into a forced mate.
# Lichess's analysis page uses ±1000 cp for the same reason — anything beyond
# "down a queen" already saturates the move-quality signal, so the exact
# magnitude carries no extra information about how bad the move was.
EVAL_CLAMP_CP = 1000


# Theme synonyms for T3 scoring.
#
# Keyed by the *exact Lichess theme label* used in the puzzle dataset (mostly
# camelCase) so puzzle rows match up directly. Each entry lists natural-English
# phrases a model might write when discussing the theme. Snake-case keys are
# preserved as aliases for backwards compatibility with code that constructs
# them programmatically.
THEME_SYNONYMS = {
    # Tactics
    "fork": ["fork", "double attack", "knight fork", "family fork"],
    "pin": ["pin", "pinned", "pinning", "absolute pin", "relative pin"],
    "skewer": ["skewer", "skewered"],
    "discovery": ["discovered", "discovery", "discovered attack", "discovered check"],
    "discoveredAttack": ["discovered attack", "discovery", "discovered check", "discovered"],
    "doubleCheck": ["double check", "double attack on the king"],
    "deflection": ["deflection", "deflect", "decoy"],
    "decoy": ["decoy", "decoying", "lure", "luring"],
    "sacrifice": ["sacrifice", "sac", "sacrificing"],
    "queenSacrifice": ["queen sacrifice", "sacrifice the queen", "sacrificing the queen"],
    "exposedKing": ["exposed king", "king is exposed", "weak king", "open king"],
    "hangingPiece": ["hanging", "undefended", "loose piece", "hanging piece"],
    "trappedPiece": ["trapped", "trap", "trapped piece"],
    "overloading": ["overloaded", "overworked", "overloading"],
    "interference": ["interference", "interfering"],
    "clearance": ["clearance", "clearing", "clearance sacrifice"],
    "undermining": ["undermining", "removing the defender", "remove the defender"],
    "attraction": ["attraction", "attract", "attracting"],
    "attackingF2F7": ["attack on f2", "attack on f7", "weak square f2", "weak square f7"],
    "xRayAttack": ["x-ray", "x ray", "xray"],
    "zugzwang": ["zugzwang"],
    "intermezzo": ["zwischenzug", "intermezzo", "in-between move", "intermediate move"],
    "zwischenzug": ["zwischenzug", "intermezzo", "in-between move", "intermediate"],
    "skewer_": ["skewer"],
    "quietMove": ["quiet move", "waiting move", "prophylactic"],
    # Mate themes
    "mate": ["mate", "checkmate", "mating"],
    "mateIn1": ["mate in one", "mate in 1", "checkmate", "mate"],
    "mateIn2": ["mate in two", "mate in 2", "mating sequence"],
    "mateIn3": ["mate in three", "mate in 3", "mating sequence"],
    "mateIn4": ["mate in four", "mate in 4", "mating sequence"],
    "mateIn5": ["mate in five", "mate in 5", "mating sequence"],
    "smotheredMate": ["smothered mate", "smothered"],
    "backRankMate": ["back rank", "back-rank mate", "back rank mate", "weak back rank"],
    "anastasiaMate": ["anastasia", "anastasia's mate"],
    "arabianMate": ["arabian mate"],
    "bodenMate": ["boden", "boden's mate"],
    "doubleBishopMate": ["double bishop mate", "two bishops mate"],
    "killBoxMate": ["kill box"],
    "hookMate": ["hook mate"],
    "vukovicMate": ["vukovic"],
    "dovetailMate": ["dovetail"],
    # Position evaluation / themes
    "advantage": ["advantage", "better", "winning", "ahead", "favourable", "favorable"],
    "crushing": [
        "crushing", "winning", "decisive", "overwhelming", "dominant", "dominating",
        "completely winning", "totally winning", "much better",
    ],
    "equality": ["equal", "equality", "balanced", "level", "even"],
    "advancedPawn": [
        "advanced pawn", "passed pawn", "far advanced", "promoted", "promotion",
        "pawn on the seventh", "passer",
    ],
    "passedPawn": ["passed pawn", "passer", "outside passer", "connected passers"],
    "passed_pawn": ["passed pawn", "passer", "outside passer", "connected passers"],
    "kingsideAttack": [
        "kingside attack", "attacking the king", "king-side attack", "attack on the king",
        "attack on the kingside", "kingside",
    ],
    "queensideAttack": ["queenside attack", "queenside", "attack on the queenside"],
    "kingAttack": ["attack on the king", "king attack", "mating attack"],
    "fianchetto": ["fianchetto", "fianchettoed"],
    "long": ["long sequence", "long combination"],
    "short": ["short sequence", "short combination"],
    # Phases
    "opening": ["opening", "development"],
    "middlegame": ["middlegame", "middle game"],
    "endgame": ["endgame", "ending", "endgame technique"],
    "rookEndgame": ["rook endgame", "rook ending"],
    "bishopEndgame": ["bishop endgame", "bishop ending"],
    "knightEndgame": ["knight endgame", "knight ending"],
    "queenEndgame": ["queen endgame", "queen ending"],
    "pawnEndgame": ["pawn endgame", "pawn ending", "king and pawn endgame"],
    "queenRookEndgame": ["queen and rook endgame", "queen rook endgame"],
    # Catch-alls / generic
    "tactics": ["tactics", "tactical", "tactic"],
    "trapped": ["trapped", "trap"],
    "back_rank": ["back rank", "backrank", "back-rank mate"],
    "game_position": ["position", "positional"],
    "random_play": ["position", "positional"],
}


def _camel_to_words(label: str) -> str:
    """Convert a camelCase Lichess theme label to space-separated lowercase words.

    >>> _camel_to_words("advancedPawn")
    'advanced pawn'
    >>> _camel_to_words("backRankMate")
    'back rank mate'
    >>> _camel_to_words("mate")
    'mate'
    """
    if not label:
        return ""
    out = []
    for ch in label:
        if ch.isupper() and out:
            out.append(" ")
        out.append(ch.lower())
    return "".join(out)


def get_direction(eval_cp: int, threshold: int = 50) -> str:
    """Determine which side is better based on centipawn evaluation.

    Args:
        eval_cp: Centipawn evaluation from White's perspective
        threshold: Threshold for considering position equal

    Returns:
        "White", "Black", or "Equal"
    """
    if eval_cp > threshold:
        return "White"
    elif eval_cp < -threshold:
        return "Black"
    else:
        return "Equal"


def score_t1(
    model_eval: int | None,
    stockfish_eval: int,
    eval_range: tuple[int, int] = (-2000, 2000),
) -> dict[str, Any]:
    """Score Task 1: Centipawn Evaluation.

    Args:
        model_eval: Model's centipawn evaluation (may be None if parsing failed)
        stockfish_eval: Stockfish's ground truth evaluation
        eval_range: Range to clamp model evaluation

    Returns:
        Dictionary with T1 scoring results
    """
    if model_eval is None:
        return {
            "t1_model_eval": None,
            "t1_stockfish_eval": stockfish_eval,
            "t1_absolute_error": None,
            "t1_direction_correct": None,
        }

    # Clamp model evaluation to range
    clamped_eval = int(clamp(model_eval, eval_range[0], eval_range[1]))

    absolute_error = abs(clamped_eval - stockfish_eval)
    direction_correct = get_direction(clamped_eval) == get_direction(stockfish_eval)

    return {
        "t1_model_eval": clamped_eval,
        "t1_stockfish_eval": stockfish_eval,
        "t1_absolute_error": absolute_error,
        "t1_direction_correct": direction_correct,
    }


def score_t2(
    model_move: str | None,
    fen: str,
    stockfish_best_move: str,
    stockfish_eval: int,
    engine=None,
) -> dict[str, Any]:
    """Score Task 2: Best Move.

    Args:
        model_move: Model's move in SAN notation (may be None if parsing failed)
        fen: Position FEN
        stockfish_best_move: Stockfish's best move
        stockfish_eval: Stockfish evaluation before the move
        engine: Optional Stockfish engine for CPL calculation

    Returns:
        Dictionary with T2 scoring results
    """
    if model_move is None:
        return {
            "t2_move": None,
            "t2_best_move": stockfish_best_move,
            "t2_legal": False,
            "t2_cpl": None,
        }

    # Check legality
    try:
        board = chess.Board(fen)
        move = board.parse_san(model_move)
        is_legal = move in board.legal_moves
    except Exception:
        is_legal = False

    if not is_legal:
        return {
            "t2_move": model_move,
            "t2_best_move": stockfish_best_move,
            "t2_legal": False,
            "t2_cpl": None,
        }

    # Check if it's the best move
    is_best = model_move == stockfish_best_move

    # Calculate CPL if engine is available.
    #
    # Both endpoints are clamped to ±EVAL_CLAMP_CP before subtraction so that
    # mate-encoded scores (which can reach ±16,000 cp in this dataset) don't
    # leak into the difference. See module docstring on EVAL_CLAMP_CP.
    cpl = None
    if engine is not None:
        try:
            eval_after = engine.evaluate_after_move(fen, model_move)
            board = chess.Board(fen)
            sf_clamped = max(-EVAL_CLAMP_CP, min(EVAL_CLAMP_CP, stockfish_eval))
            ea_clamped = max(-EVAL_CLAMP_CP, min(EVAL_CLAMP_CP, eval_after))
            if board.turn == chess.WHITE:
                cpl = sf_clamped - ea_clamped
            else:
                cpl = ea_clamped - sf_clamped
            # CPL should be non-negative (best move has CPL 0). Negative values
            # arise from depth/horizon noise when the model's move evaluates
            # marginally better than Stockfish's at the same depth.
            cpl = max(0, cpl)
        except Exception as e:
            logger.warning(f"CPL calculation failed: {e}")
            cpl = None
    elif is_best:
        cpl = 0

    return {
        "t2_move": model_move,
        "t2_best_move": stockfish_best_move,
        "t2_legal": True,
        "t2_cpl": cpl,
    }


def score_t3(
    explanation: str | None,
    side_claimed: str | None,
    stockfish_eval: int,
    theme: str,
) -> dict[str, Any]:
    """Score Task 3: Positional Explanation (Option A).

    Two binary criteria:
    - Point 1: Correct side identification
    - Point 2: Theme mention

    Args:
        explanation: Model's explanation text
        side_claimed: Side the model claims is better
        stockfish_eval: Stockfish evaluation for ground truth
        theme: Expected theme tag

    Returns:
        Dictionary with T3 scoring results
    """
    if explanation is None:
        return {
            "t3_explanation": None,
            "t3_side_claimed": side_claimed,
            "t3_p1_side_correct": None,
            "t3_p2_theme_correct": None,
            "t3_score": None,
        }

    # Point 1: Side identification
    ground_truth_side = get_direction(stockfish_eval)
    p1 = 1 if side_claimed == ground_truth_side else 0

    # Point 2: Theme identification.
    #
    # Lichess theme labels are mostly camelCase (advancedPawn, kingsideAttack,
    # backRankMate). The synonym dict is keyed by exact label. Final fallback
    # converts camelCase to space-separated words so unmapped labels still
    # have a sensible matcher.
    p2 = 0
    explanation_lower = explanation.lower()

    candidates: list[str] = []
    if isinstance(theme, str):
        candidates.extend(THEME_SYNONYMS.get(theme, []))
        # Normalised forms of the label itself
        candidates.append(theme.lower())
        candidates.append(_camel_to_words(theme))
        candidates.append(theme.lower().replace("_", " "))

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if candidate in explanation_lower:
            p2 = 1
            break

    return {
        "t3_explanation": explanation,
        "t3_side_claimed": side_claimed,
        "t3_p1_side_correct": p1,
        "t3_p2_theme_correct": p2,
        "t3_score": p1 + p2,
    }


def score_all(
    parsed_response: dict[str, Any],
    position: dict[str, Any],
    engine=None,
    eval_range: tuple[int, int] = (-2000, 2000),
) -> dict[str, Any]:
    """Score all three tasks for a position.

    Args:
        parsed_response: Parsed LLM response with eval, move, explanation
        position: Position dictionary with fen, stockfish_eval, stockfish_best_move, theme
        engine: Optional Stockfish engine for CPL calculation
        eval_range: Range to clamp model evaluation

    Returns:
        Combined scoring results for T1, T2, T3
    """
    results = {}

    # T1 scoring
    t1_results = score_t1(
        model_eval=parsed_response.get("eval"),
        stockfish_eval=position.get("stockfish_eval", 0),
        eval_range=eval_range,
    )
    results.update(t1_results)

    # T2 scoring
    t2_results = score_t2(
        model_move=parsed_response.get("move"),
        fen=position["fen"],
        stockfish_best_move=position.get("stockfish_best_move", ""),
        stockfish_eval=position.get("stockfish_eval", 0),
        engine=engine,
    )
    results.update(t2_results)

    # T3 scoring
    t3_results = score_t3(
        explanation=parsed_response.get("explanation"),
        side_claimed=parsed_response.get("side_claimed"),
        stockfish_eval=position.get("stockfish_eval", 0),
        theme=position.get("theme", ""),
    )
    results.update(t3_results)

    return results


def should_trigger_correction(
    t2_cpl: int | None,
    threshold: int = 50,
) -> bool:
    """Determine if a correction loop should be triggered.

    Args:
        t2_cpl: Centipawn loss from T2 scoring
        threshold: CPL threshold for triggering correction

    Returns:
        True if correction should be triggered
    """
    if t2_cpl is None:
        return False
    return t2_cpl > threshold
