"""Stockfish chess engine wrapper for position evaluation."""

import subprocess
import logging
import os
import select
import time
from collections import deque
from typing import Any

import chess

logger = logging.getLogger("chess_llm_bench")

DEFAULT_PROTOCOL_TIMEOUT = 120.0
PROTOCOL_READ_BYTES = 65536
MAX_PROTOCOL_LINES = 10000


def _read_uci_line(stream, pending: bytes, timeout: float) -> tuple[str, bytes]:
    """Read a complete line within a deadline, retaining already-buffered bytes."""
    try:
        descriptor = stream.fileno()
    except (AttributeError, OSError):
        descriptor = None
    if not isinstance(descriptor, int):  # In-memory streams and test doubles.
        line = stream.readline()
        if line == "":
            raise EOFError("Engine closed its output before the expected response")
        return line.strip(), pending
    deadline = time.monotonic() + timeout
    while b"\n" not in pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([descriptor], [], [], remaining)[0]:
            raise TimeoutError("Timed out waiting for an engine response")
        chunk = os.read(descriptor, PROTOCOL_READ_BYTES)
        if not chunk:
            raise EOFError("Engine closed its output before the expected response")
        pending += chunk
        if len(pending) > PROTOCOL_READ_BYTES and b"\n" not in pending:
            raise RuntimeError("Engine returned an oversized unterminated line")
    line, pending = pending.split(b"\n", 1)
    return line.decode("utf-8", errors="replace").strip(), pending


def _wait_for_response(engine, target: str) -> list[str]:
    deadline = time.monotonic() + engine.timeout
    lines: deque[str] = deque(maxlen=MAX_PROTOCOL_LINES)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for engine token {target!r}")
        if engine.process is None or engine.process.stdout is None:
            raise EOFError("Engine output is unavailable")
        line = engine._read_line(timeout=remaining)
        if line:
            lines.append(line)
        if target in line:
            return list(lines)


class StockfishEngine:
    """Wrapper for Stockfish UCI engine subprocess."""

    def __init__(
        self,
        path: str = "/usr/games/stockfish",
        depth: int = 22,
        threads: int = 1,
        timeout: float = DEFAULT_PROTOCOL_TIMEOUT,
    ):
        """Initialize Stockfish engine.

        Args:
            path: Path to Stockfish executable
            depth: Search depth for evaluation
            threads: Number of threads for Stockfish to use
        """
        self.path = path
        self.depth = depth
        self.threads = threads
        if timeout <= 0:
            raise ValueError("Engine timeout must be positive")
        self.timeout = timeout
        self.process: subprocess.Popen | None = None
        self._start()

    def _start(self) -> None:
        """Start the Stockfish subprocess."""
        self._read_buffer = b""
        try:
            self.process = subprocess.Popen(
                [self.path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self._send("uci")
            self._wait_for("uciok")
            self._send(f"setoption name Threads value {self.threads}")
            self._send("isready")
            self._wait_for("readyok")
            logger.info(f"Stockfish started: {self.path}")
        except FileNotFoundError:
            raise RuntimeError(f"Stockfish not found at {self.path}")
        except Exception as e:
            self.close()
            raise RuntimeError(f"Failed to start Stockfish: {e}")

    def _send(self, command: str) -> None:
        """Send a command to Stockfish."""
        if self.process and self.process.stdin:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()

    def _read_line(self, timeout: float | None = None) -> str:
        """Read a line from Stockfish output."""
        if self.process and self.process.stdout:
            line, self._read_buffer = _read_uci_line(
                self.process.stdout, getattr(self, "_read_buffer", b""),
                self.timeout if timeout is None else timeout,
            )
            return line
        return ""

    def _wait_for(self, target: str) -> list[str]:
        """Read lines until target string is found.

        Returns:
            List of lines read (including the target line)
        """
        return _wait_for_response(self, target)

    def _restart(self) -> None:
        """Restart the Stockfish subprocess."""
        logger.warning("Restarting Stockfish engine...")
        self.close()
        self._start()

    def evaluate(self, fen: str) -> dict[str, Any]:
        """Evaluate a position and return centipawn score and best move.

        Args:
            fen: FEN string of the position to evaluate

        Returns:
            Dictionary with 'eval' (centipawns from White's perspective),
            'best_move' (SAN notation), and 'mate' (moves to mate, if applicable)
        """
        try:
            return self._evaluate_internal(fen)
        except Exception as e:
            logger.warning(f"Stockfish evaluation failed, restarting: {e}")
            self._restart()
            return self._evaluate_internal(fen)

    def _evaluate_internal(self, fen: str) -> dict[str, Any]:
        """Internal evaluation implementation."""
        self._send("ucinewgame")
        self._send(f"position fen {fen}")
        self._send(f"go depth {self.depth}")

        lines = self._wait_for("bestmove")

        # Parse the evaluation from info lines
        eval_cp = 0
        mate_in = None
        best_move_uci = None

        for line in lines:
            if line.startswith("info") and "score" in line:
                parts = line.split()
                try:
                    score_idx = parts.index("score")
                    if parts[score_idx + 1] == "cp":
                        eval_cp = int(parts[score_idx + 2])
                    elif parts[score_idx + 1] == "mate":
                        mate_in = int(parts[score_idx + 2])
                        # Convert mate to large centipawn value
                        if mate_in > 0:
                            eval_cp = 10000 - mate_in * 10
                        else:
                            eval_cp = -10000 - mate_in * 10
                except (ValueError, IndexError):
                    pass

            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    best_move_uci = parts[1]

        # Convert best move from UCI to SAN
        best_move_san = None
        if best_move_uci and best_move_uci != "(none)":
            try:
                board = chess.Board(fen)
                move = chess.Move.from_uci(best_move_uci)
                best_move_san = board.san(move)
            except Exception:
                best_move_san = best_move_uci  # Fallback to UCI notation

        # Adjust evaluation for side to move
        # Stockfish reports score from the perspective of the side to move
        # We want it from White's perspective
        board = chess.Board(fen)
        if board.turn == chess.BLACK:
            eval_cp = -eval_cp
            if mate_in is not None:
                mate_in = -mate_in

        return {
            "eval": eval_cp,
            "best_move": best_move_san,
            "mate": mate_in,
        }

    def evaluate_after_move(self, fen: str, move_san: str) -> int:
        """Evaluate position after applying a move.

        Args:
            fen: Starting FEN
            move_san: Move in SAN notation

        Returns:
            Centipawn evaluation from White's perspective
        """
        board = chess.Board(fen)
        move = board.parse_san(move_san)
        board.push(move)
        result = self.evaluate(board.fen())
        return result["eval"]

    def is_legal_move(self, fen: str, move_san: str) -> bool:
        """Check if a move is legal in the given position.

        Args:
            fen: FEN string
            move_san: Move in SAN notation

        Returns:
            True if the move is legal, False otherwise
        """
        try:
            board = chess.Board(fen)
            move = board.parse_san(move_san)
            return move in board.legal_moves
        except Exception:
            return False

    def close(self) -> None:
        """Shut down the Stockfish subprocess."""
        if self.process:
            try:
                self._send("quit")
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            finally:
                self.process = None
                logger.info("Stockfish closed")

    def __enter__(self) -> "StockfishEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class Lc0Engine:
    """Wrapper for Lc0 UCI engine subprocess (GPU-accelerated)."""

    def __init__(
        self,
        path: str | None = None,
        weights: str | None = None,
        nodes: int = 1000,
        backend: str = "cuda-auto",
        timeout: float = DEFAULT_PROTOCOL_TIMEOUT,
    ):
        """Initialize Lc0 engine.

        Args:
            path: Path to Lc0 executable. Required — pass explicitly or via config.
            weights: Path to neural network weights. Required.
            nodes: Number of nodes to search per position
            backend: Backend to use (cuda-auto, cuda, cudnn, etc.)
        """
        if path is None or weights is None:
            raise ValueError(
                "Lc0Engine requires both `path` and `weights`; supply them via "
                "config/config.yaml (lc0.binary / lc0.weights) instead of "
                "relying on hardcoded defaults."
            )
        self.path = path
        self.weights = weights
        self.nodes = nodes
        self.backend = backend
        if timeout <= 0:
            raise ValueError("Engine timeout must be positive")
        self.timeout = timeout
        self.process: subprocess.Popen | None = None
        self._start()

    def _start(self) -> None:
        """Start the Lc0 subprocess."""
        self._read_buffer = b""
        try:
            self.process = subprocess.Popen(
                [self.path, f"--weights={self.weights}", f"--backend={self.backend}"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self._send("uci")
            self._wait_for("uciok")
            self._send("isready")
            self._wait_for("readyok")
            logger.info(f"Lc0 started: {self.path} (backend={self.backend})")
        except FileNotFoundError:
            raise RuntimeError(f"Lc0 not found at {self.path}")
        except Exception as e:
            self.close()
            raise RuntimeError(f"Failed to start Lc0: {e}")

    def _send(self, command: str) -> None:
        """Send a command to Lc0."""
        if self.process and self.process.stdin:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()

    def _read_line(self, timeout: float | None = None) -> str:
        """Read a line from Lc0 output."""
        if self.process and self.process.stdout:
            line, self._read_buffer = _read_uci_line(
                self.process.stdout, getattr(self, "_read_buffer", b""),
                self.timeout if timeout is None else timeout,
            )
            return line
        return ""

    def _wait_for(self, target: str) -> list[str]:
        """Read lines until target string is found."""
        return _wait_for_response(self, target)

    def _restart(self) -> None:
        """Restart the Lc0 subprocess."""
        logger.warning("Restarting Lc0 engine...")
        self.close()
        self._start()

    def evaluate(self, fen: str) -> dict[str, Any]:
        """Evaluate a position and return centipawn score and best move."""
        try:
            return self._evaluate_internal(fen)
        except Exception as e:
            logger.warning(f"Lc0 evaluation failed, restarting: {e}")
            self._restart()
            return self._evaluate_internal(fen)

    def _evaluate_internal(self, fen: str) -> dict[str, Any]:
        """Internal evaluation implementation."""
        self._send("ucinewgame")
        self._send(f"position fen {fen}")
        self._send(f"go nodes {self.nodes}")

        lines = self._wait_for("bestmove")

        # Parse the evaluation from info lines
        eval_cp = 0
        mate_in = None
        best_move_uci = None

        for line in lines:
            if line.startswith("info") and "score" in line:
                parts = line.split()
                try:
                    score_idx = parts.index("score")
                    if parts[score_idx + 1] == "cp":
                        eval_cp = int(parts[score_idx + 2])
                    elif parts[score_idx + 1] == "mate":
                        mate_in = int(parts[score_idx + 2])
                        if mate_in > 0:
                            eval_cp = 10000 - mate_in * 10
                        else:
                            eval_cp = -10000 - mate_in * 10
                except (ValueError, IndexError):
                    pass

            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    best_move_uci = parts[1]

        # Convert best move from UCI to SAN
        best_move_san = None
        if best_move_uci and best_move_uci != "(none)":
            try:
                board = chess.Board(fen)
                move = chess.Move.from_uci(best_move_uci)
                best_move_san = board.san(move)
            except Exception:
                best_move_san = best_move_uci

        # Adjust evaluation for side to move
        board = chess.Board(fen)
        if board.turn == chess.BLACK:
            eval_cp = -eval_cp
            if mate_in is not None:
                mate_in = -mate_in

        return {
            "eval": eval_cp,
            "best_move": best_move_san,
            "mate": mate_in,
        }

    def close(self) -> None:
        """Shut down the Lc0 subprocess."""
        if self.process:
            try:
                self._send("quit")
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            finally:
                self.process = None
                logger.info("Lc0 closed")

    def __enter__(self) -> "Lc0Engine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
