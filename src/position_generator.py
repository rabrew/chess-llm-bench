"""Generate novel chess positions for benchmarking."""

import logging
import random
from typing import Any

import chess

logger = logging.getLogger("chess_llm_bench")


def validate_position(board: chess.Board) -> bool:
    """Validate that a position is legal and suitable for evaluation.

    Args:
        board: Chess board to validate

    Returns:
        True if position is valid for benchmarking
    """
    # Must be a legal position
    if not board.is_valid():
        return False

    # Must not be game over
    if board.is_game_over():
        return False

    # Must have both kings
    if not board.king(chess.WHITE) or not board.king(chess.BLACK):
        return False

    # Must have at least one legal move
    if not list(board.legal_moves):
        return False

    return True


def generate_random_position(
    rng: random.Random,
    min_moves: int = 10,
    max_moves: int = 60,
) -> dict[str, Any] | None:
    """Generate a position by playing random legal moves from start.

    Args:
        rng: Random number generator instance
        min_moves: Minimum number of half-moves to play
        max_moves: Maximum number of half-moves to play

    Returns:
        Position dictionary or None if generation failed
    """
    board = chess.Board()
    move_history = []
    target_moves = rng.randint(min_moves, max_moves)

    for _ in range(target_moves):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        move = rng.choice(legal_moves)
        move_history.append(board.san(move))
        board.push(move)

        if board.is_game_over():
            break

    if not validate_position(board):
        return None

    # Determine game phase based on material and move count
    phase = determine_phase(board, len(move_history))

    return {
        "fen": board.fen(),
        "pgn_moves": moves_to_pgn(move_history),
        "phase": phase,
        "source": "generated",
        "theme": "random_play",
    }


def generate_themed_position(
    rng: random.Random,
    theme: str,
    base_fen: str | None = None,
) -> dict[str, Any] | None:
    """Generate a position with a specific tactical theme.

    Args:
        rng: Random number generator instance
        theme: Tactical theme (fork, pin, skewer, etc.)
        base_fen: Optional starting FEN for the theme

    Returns:
        Position dictionary or None if generation failed
    """
    # Theme-specific base positions (known positions with the theme)
    theme_bases = {
        "fork": [
            # Knight fork positions
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        ],
        "pin": [
            # Bishop pin positions
            "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
            "rnbqk1nr/pppp1ppp/4p3/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
        ],
        "skewer": [
            "8/8/8/3k4/8/3B4/3K4/8 w - - 0 1",
        ],
        "passed_pawn": [
            "8/5P2/8/8/8/8/8/4K2k w - - 0 1",
            "8/8/8/8/8/5p2/8/4K2k b - - 0 1",
        ],
        "discovery": [
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        ],
    }

    if base_fen:
        board = chess.Board(base_fen)
    elif theme in theme_bases:
        base_fen = rng.choice(theme_bases[theme])
        board = chess.Board(base_fen)
    else:
        # Fall back to random position
        return generate_random_position(rng)

    # Apply some random perturbations (limited moves)
    num_perturbations = rng.randint(0, 3)
    move_history = []

    for _ in range(num_perturbations):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        move = rng.choice(legal_moves)
        move_history.append(board.san(move))
        board.push(move)

        if board.is_game_over():
            break

    if not validate_position(board):
        return None

    phase = determine_phase(board, len(move_history))

    return {
        "fen": board.fen(),
        "pgn_moves": moves_to_pgn(move_history) if move_history else "",
        "phase": phase,
        "source": "generated",
        "theme": theme,
    }


def generate_endgame_position(
    rng: random.Random,
    piece_config: str = "KQvK",
) -> dict[str, Any] | None:
    """Generate an endgame position with specific piece configuration.

    Args:
        rng: Random number generator instance
        piece_config: Piece configuration string (e.g., "KQvK", "KRvK", "KPvK")

    Returns:
        Position dictionary or None if generation failed
    """
    # Parse piece configuration
    configs = {
        "KQvK": {"white": [chess.QUEEN], "black": []},
        "KRvK": {"white": [chess.ROOK], "black": []},
        "KBBvK": {"white": [chess.BISHOP, chess.BISHOP], "black": []},
        "KBNvK": {"white": [chess.BISHOP, chess.KNIGHT], "black": []},
        "KPvK": {"white": [chess.PAWN], "black": []},
        "KRvKR": {"white": [chess.ROOK], "black": [chess.ROOK]},
        "KPvKP": {"white": [chess.PAWN], "black": [chess.PAWN]},
    }

    if piece_config not in configs:
        piece_config = rng.choice(list(configs.keys()))

    config = configs[piece_config]
    board = chess.Board(None)  # Empty board

    # Place kings (not adjacent)
    white_king_sq = rng.randint(0, 63)
    board.set_piece_at(white_king_sq, chess.Piece(chess.KING, chess.WHITE))

    # Find valid square for black king (not adjacent to white king)
    valid_squares = []
    for sq in range(64):
        if sq == white_king_sq:
            continue
        if chess.square_distance(sq, white_king_sq) > 1:
            valid_squares.append(sq)

    if not valid_squares:
        return None

    black_king_sq = rng.choice(valid_squares)
    board.set_piece_at(black_king_sq, chess.Piece(chess.KING, chess.BLACK))

    # Place white pieces
    occupied = {white_king_sq, black_king_sq}
    for piece_type in config["white"]:
        placed = False
        for _ in range(100):  # Max attempts
            sq = rng.randint(0, 63)
            if sq in occupied:
                continue
            # Pawns can't be on rank 1 or 8
            if piece_type == chess.PAWN:
                rank = chess.square_rank(sq)
                if rank == 0 or rank == 7:
                    continue
            board.set_piece_at(sq, chess.Piece(piece_type, chess.WHITE))
            occupied.add(sq)
            placed = True
            break
        if not placed:
            logger.warning(f"Could not place white {chess.piece_name(piece_type)} after 100 attempts — position may be incomplete")

    # Place black pieces
    for piece_type in config["black"]:
        placed = False
        for _ in range(100):
            sq = rng.randint(0, 63)
            if sq in occupied:
                continue
            if piece_type == chess.PAWN:
                rank = chess.square_rank(sq)
                if rank == 0 or rank == 7:
                    continue
            board.set_piece_at(sq, chess.Piece(piece_type, chess.BLACK))
            occupied.add(sq)
            placed = True
            break
        if not placed:
            logger.warning(f"Could not place black {chess.piece_name(piece_type)} after 100 attempts — position may be incomplete")

    # Set turn randomly
    board.turn = rng.choice([chess.WHITE, chess.BLACK])

    if not validate_position(board):
        return None

    return {
        "fen": board.fen(),
        "pgn_moves": "",
        "phase": "endgame",
        "source": "generated",
        "theme": f"endgame_{piece_config}",
    }


def determine_phase(board: chess.Board, move_count: int) -> str:
    """Determine the game phase based on material and move count.

    Args:
        board: Chess board
        move_count: Number of half-moves played

    Returns:
        Phase string: "opening", "middlegame", or "endgame"
    """
    # Count material
    piece_count = len(board.piece_map())
    queen_count = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(
        board.pieces(chess.QUEEN, chess.BLACK)
    )
    minor_major_count = (
        len(board.pieces(chess.ROOK, chess.WHITE))
        + len(board.pieces(chess.ROOK, chess.BLACK))
        + len(board.pieces(chess.BISHOP, chess.WHITE))
        + len(board.pieces(chess.BISHOP, chess.BLACK))
        + len(board.pieces(chess.KNIGHT, chess.WHITE))
        + len(board.pieces(chess.KNIGHT, chess.BLACK))
    )

    # Opening: first ~20 half-moves
    if move_count <= 20 and piece_count >= 28:
        return "opening"

    # Endgame: few pieces left
    if piece_count <= 10 or (queen_count == 0 and minor_major_count <= 4):
        return "endgame"

    return "middlegame"


def moves_to_pgn(moves: list[str]) -> str:
    """Convert a list of SAN moves to PGN format.

    Args:
        moves: List of moves in SAN notation

    Returns:
        PGN-formatted move string
    """
    pgn_parts = []
    for i, move in enumerate(moves):
        if i % 2 == 0:
            pgn_parts.append(f"{i // 2 + 1}. {move}")
        else:
            pgn_parts.append(move)
    return " ".join(pgn_parts)


def generate_positions(
    count: int,
    seed: int = 42,
    themes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate multiple positions for the dataset.

    Args:
        count: Number of positions to generate
        seed: Random seed for reproducibility
        themes: Optional list of themes to include

    Returns:
        List of position dictionaries
    """
    rng = random.Random(seed)
    positions = []

    if themes is None:
        themes = ["fork", "pin", "skewer", "passed_pawn", "discovery"]

    endgame_configs = ["KQvK", "KRvK", "KPvK", "KRvKR", "KPvKP"]

    # Generate a mix of position types
    target_random = count // 3
    target_themed = count // 3
    target_endgame = count - target_random - target_themed

    # Random positions
    attempts = 0
    while len(positions) < target_random and attempts < count * 10:
        pos = generate_random_position(rng)
        if pos:
            positions.append(pos)
        attempts += 1

    # Themed positions
    attempts = 0
    while len(positions) < target_random + target_themed and attempts < count * 10:
        theme = rng.choice(themes)
        pos = generate_themed_position(rng, theme)
        if pos:
            positions.append(pos)
        attempts += 1

    # Endgame positions
    attempts = 0
    while len(positions) < count and attempts < count * 10:
        config = rng.choice(endgame_configs)
        pos = generate_endgame_position(rng, config)
        if pos:
            positions.append(pos)
        attempts += 1

    logger.info(f"Generated {len(positions)} positions")
    return positions
