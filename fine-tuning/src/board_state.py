"""Board state utility for chess FEN parsing, verification, and correction.

Used by:
  - fine-tuning/scripts/generate_board_dataset.py  (training data generation)
  - fine-tuning/scripts/finetune_model.py          (held-out eval during training)
  - fine-tuning/scripts/compare_finetuning_results.py (T0 board accuracy metric)
"""

import re
from typing import Any

import chess

# Canonical piece type names used in descriptions
PIECE_TYPE_NAMES = {
    chess.KING:   "King",
    chess.QUEEN:  "Queen",
    chess.ROOK:   "Rook",
    chess.BISHOP: "Bishop",
    chess.KNIGHT: "Knight",
    chess.PAWN:   "Pawn",
}

# Plural forms used in the description string (for multi-piece types)
PIECE_TYPE_PLURAL = {
    chess.KING:   "King",
    chess.QUEEN:  "Queen",
    chess.ROOK:   "Rooks",
    chess.BISHOP: "Bishops",
    chess.KNIGHT: "Knights",
    chess.PAWN:   "Pawns",
}

# Ordered for canonical output
PIECE_ORDER = [
    chess.KING,
    chess.QUEEN,
    chess.ROOK,
    chess.BISHOP,
    chess.KNIGHT,
    chess.PAWN,
]


def _squares_for(board: chess.Board, color: chess.Color, piece_type: int) -> list[str]:
    """Return sorted list of square names for a given color/piece_type."""
    squares = board.pieces(piece_type, color)
    names = [chess.square_name(sq) for sq in sorted(squares)]
    return names


def _color_line(board: chess.Board, color: chess.Color) -> str:
    """Build the 'White: ...' or 'Black: ...' description line."""
    color_name = "White" if color == chess.WHITE else "Black"
    parts = []
    for piece_type in PIECE_ORDER:
        squares = _squares_for(board, color, piece_type)
        if not squares:
            continue
        label = PIECE_TYPE_PLURAL[piece_type] if len(squares) > 1 else PIECE_TYPE_NAMES[piece_type]
        parts.append(f"{label} {' '.join(squares)}")
    return f"{color_name}: {', '.join(parts)}"


def fen_to_board_description(fen: str) -> str:
    """Ground-truth board description from FEN using python-chess.

    Returns:
        Multi-line string:
          'White: King e1, Queen d1, Rooks a1 h1, ...'
          'Black: King e8, ...'
          'Side to move: White'
    """
    board = chess.Board(fen)
    white_line = _color_line(board, chess.WHITE)
    black_line = _color_line(board, chess.BLACK)
    side = "White" if board.turn == chess.WHITE else "Black"
    return f"{white_line}\n{black_line}\nSide to move: {side}"


# ---------------------------------------------------------------------------
# Parser — best-effort conversion of model output back to piece dict
# ---------------------------------------------------------------------------

# All valid square names
_VALID_SQUARES = {chess.square_name(sq) for sq in chess.SQUARES}

# Maps singular/plural labels back to canonical singular form
_LABEL_MAP: dict[str, str] = {}
for _pt, _name in PIECE_TYPE_NAMES.items():
    _LABEL_MAP[_name.lower()] = _name
    _LABEL_MAP[PIECE_TYPE_PLURAL[_pt].lower()] = _name


def parse_board_description(text: str) -> dict[str, list[str]]:
    """Parse a model's board description back into {piece_label: [squares]}.

    Best-effort — handles partial or slightly malformed model output.

    Returns:
        Dict mapping e.g. 'White King' -> ['e1'], 'White Pawn' -> ['a2', ...]
    """
    result: dict[str, list[str]] = {}
    if not text:
        return result

    current_color: str | None = None

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Detect color prefix
        if line.lower().startswith("white:"):
            current_color = "White"
            remainder = line[6:].strip()
        elif line.lower().startswith("black:"):
            current_color = "Black"
            remainder = line[6:].strip()
        elif line.lower().startswith("side to move"):
            continue
        else:
            remainder = line

        if current_color is None:
            continue

        # Split on comma — each segment is "<PieceType(s)> <sq1> <sq2> ..."
        for segment in remainder.split(","):
            segment = segment.strip()
            if not segment:
                continue

            tokens = segment.split()
            if not tokens:
                continue

            # Find piece label — one or two leading words that match known types
            piece_label: str | None = None
            sq_start = 0

            # Try two-word match first (e.g. "White King" in full text shouldn't happen here,
            # but handle gracefully)
            for n_words in (1, 2):
                candidate = " ".join(tokens[:n_words]).lower().rstrip("s")
                # Also try with trailing 's' stripped
                for try_label in (candidate, candidate + "s"):
                    mapped = _LABEL_MAP.get(try_label)
                    if mapped:
                        piece_label = mapped
                        sq_start = n_words
                        break
                if piece_label:
                    break

            if piece_label is None:
                continue

            full_key = f"{current_color} {piece_label}"
            squares = [t.lower() for t in tokens[sq_start:] if t.lower() in _VALID_SQUARES]
            if squares:
                result.setdefault(full_key, []).extend(squares)

    # Deduplicate while preserving order
    for key in result:
        seen: set[str] = set()
        deduped = []
        for sq in result[key]:
            if sq not in seen:
                seen.add(sq)
                deduped.append(sq)
        result[key] = deduped

    return result


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

def _ground_truth_pieces(fen: str) -> dict[str, list[str]]:
    """Build ground-truth {piece_label: [squares]} from FEN."""
    board = chess.Board(fen)
    result: dict[str, list[str]] = {}
    for color in (chess.WHITE, chess.BLACK):
        color_name = "White" if color == chess.WHITE else "Black"
        for piece_type in PIECE_ORDER:
            squares = _squares_for(board, color, piece_type)
            if squares:
                result[f"{color_name} {PIECE_TYPE_NAMES[piece_type]}"] = squares
    return result


def verify_board_description(fen: str, model_text: str) -> dict[str, Any]:
    """Compare model's board description against python-chess ground truth.

    Returns:
        {
            'correct': bool,
            'errors': [str],
            'missing_pieces': [str],
            'extra_pieces': [str],
            'wrong_squares': [str],
            'accuracy': float,   # recall: correct pieces / real pieces
            'recall': float,
            'precision': float,  # correct pieces / pieces listed (F-8)
            'total_pieces': int,
            'missing_count': int,
            'extra_count': int,
            'side_to_move_correct': bool | None,
        }
    """
    ground_truth = _ground_truth_pieces(fen)
    model_pieces = parse_board_description(model_text)

    errors: list[str] = []
    missing: list[str] = []
    extra: list[str] = []
    wrong: list[str] = []

    total_pieces = sum(len(v) for v in ground_truth.values())
    correct_pieces = 0

    all_keys = set(ground_truth) | set(model_pieces)

    for key in all_keys:
        gt_squares = set(ground_truth.get(key, []))
        model_squares = set(model_pieces.get(key, []))

        if not gt_squares and model_squares:
            # Hallucinated piece type
            for sq in sorted(model_squares):
                extra.append(f"{key} on {sq} (not in position)")
                errors.append(f"Extra {key} on {sq}")
            continue

        if gt_squares and not model_squares:
            # Entire piece type missing
            for sq in sorted(gt_squares):
                missing.append(f"{key} on {sq}")
                errors.append(f"Missing {key} on {sq}")
            continue

        # Both present — compare squares
        for sq in sorted(gt_squares):
            if sq in model_squares:
                correct_pieces += 1
            else:
                # Could be wrong square or missing
                missing.append(f"{key} on {sq}")
                errors.append(f"{key}: expected {sq}, not found in model output")

        for sq in sorted(model_squares):
            if sq not in gt_squares:
                extra.append(f"{key} on {sq} (should not be there)")
                wrong.append(f"{key} mentioned on {sq}, not in ground truth")
                errors.append(f"Extra {key} on {sq}")

    accuracy = correct_pieces / total_pieces if total_pieces > 0 else 0.0
    listed_pieces = correct_pieces + len(extra)
    precision = correct_pieces / listed_pieces if listed_pieces > 0 else 0.0

    return {
        "correct": len(errors) == 0,
        "errors": errors,
        "missing_pieces": missing,
        "extra_pieces": extra,
        "wrong_squares": wrong,
        "accuracy": accuracy,
        "recall": accuracy,
        "precision": precision,
        "total_pieces": total_pieces,
        "missing_count": len(missing),
        "extra_count": len(extra),
        "side_to_move_correct": _side_to_move_correct(fen, model_text),
    }


_SIDE_TO_MOVE_RE = re.compile(r"side to move\s*:\s*(white|black)", re.IGNORECASE)


def _side_to_move_correct(fen: str, model_text: str) -> bool | None:
    """True/False if the text states the side to move, None if it doesn't."""
    match = _SIDE_TO_MOVE_RE.search(model_text or "")
    if match is None:
        return None
    stated_white = match.group(1).lower() == "white"
    return stated_white == (chess.Board(fen).turn == chess.WHITE)


def build_correction(fen: str, model_text: str) -> str:
    """Produce a correction string for a wrong board description.

    Format: 'Correction — <list of fixes>. Full board: <ground truth>'
    """
    result = verify_board_description(fen, model_text)
    ground_truth_desc = fen_to_board_description(fen)

    if result["correct"]:
        return f"Correct. Full board: {ground_truth_desc}"

    fixes = result["errors"][:5]  # cap to keep string manageable
    fix_str = "; ".join(fixes)
    if len(result["errors"]) > 5:
        fix_str += f" (and {len(result['errors']) - 5} more errors)"

    return f"Correction — {fix_str}. Full board: {ground_truth_desc}"
