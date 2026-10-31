"""A chess position in plain text, with no FEN (specs/plain-board-prompts.md).

The piece lists use the same wording as the fine-tuning targets in
fine-tuning/src/board_state.py; tests/test_plain_board.py checks they stay identical.
"""

import chess

PIECE_TYPE_NAMES = {
    chess.KING: "King",
    chess.QUEEN: "Queen",
    chess.ROOK: "Rook",
    chess.BISHOP: "Bishop",
    chess.KNIGHT: "Knight",
    chess.PAWN: "Pawn",
}
PIECE_TYPE_PLURAL = {
    chess.KING: "King",
    chess.QUEEN: "Queen",  # as in board_state.py: "Queen a1 b1" after a promotion
    chess.ROOK: "Rooks",
    chess.BISHOP: "Bishops",
    chess.KNIGHT: "Knights",
    chess.PAWN: "Pawns",
}
PIECE_ORDER = (chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN)

DIAGRAM_HEADER = "Board (White at the bottom, rank 8 at the top):"
DIAGRAM_FILES = "    a b c d e f g h"
DIAGRAM_KEY = (
    "Upper case = White, lower case = Black. "
    "K king, Q queen, R rook, B bishop, N knight, P pawn, . empty."
)
EMPTY_SQUARE = "."
NONE = "none"
CASTLING_RIGHTS = (
    (chess.WHITE, chess.BB_H1, "White king side"),
    (chess.WHITE, chess.BB_A1, "White queen side"),
    (chess.BLACK, chess.BB_H8, "Black king side"),
    (chess.BLACK, chess.BB_A8, "Black queen side"),
)


def side_name(color: chess.Color) -> str:
    return "White" if color == chess.WHITE else "Black"


def board_diagram(board: chess.Board) -> str:
    """Rank 8 at the top, one character per square."""
    ranks = []
    for rank in range(7, -1, -1):
        cells = [board.piece_at(chess.square(file, rank)) for file in range(8)]
        ranks.append(f"{rank + 1} | " + " ".join(p.symbol() if p else EMPTY_SQUARE for p in cells))
    return "\n".join([DIAGRAM_HEADER, *ranks, DIAGRAM_FILES, DIAGRAM_KEY])


def piece_line(board: chess.Board, color: chess.Color) -> str:
    """'White pieces: King e1, Queen d1, Rooks a1 h1, ...'"""
    parts = []
    for piece_type in PIECE_ORDER:
        squares = [chess.square_name(sq) for sq in sorted(board.pieces(piece_type, color))]
        if squares:
            label = PIECE_TYPE_PLURAL[piece_type] if len(squares) > 1 else PIECE_TYPE_NAMES[piece_type]
            parts.append(f"{label} {' '.join(squares)}")
    return f"{side_name(color)} pieces: {', '.join(parts)}"


def castling_text(board: chess.Board) -> str:
    rights = [name for _, rook_square, name in CASTLING_RIGHTS if board.castling_rights & rook_square]
    return ", ".join(rights) if rights else NONE


def en_passant_text(board: chess.Board) -> str:
    return chess.square_name(board.ep_square) if board.ep_square is not None else NONE


def fen_to_board_text(fen: str) -> str:
    """Diagram, piece lists, side to move, castling and en passant; never the FEN itself."""
    board = chess.Board(fen)
    return "\n".join([
        board_diagram(board),
        "",
        piece_line(board, chess.WHITE),
        piece_line(board, chess.BLACK),
        f"Side to move: {side_name(board.turn)}",
        f"Castling still possible: {castling_text(board)}",
        f"En passant: {en_passant_text(board)}",
    ])


def sorted_legal_moves(fen: str) -> list[str]:
    board = chess.Board(fen)
    return sorted(board.san(move) for move in board.legal_moves)
