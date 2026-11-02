"""Stockfish chess engine wrapper for position evaluation."""

import subprocess
import logging
import math
import queue
import threading
import time
from typing import Any

import chess

logger = logging.getLogger("chess_llm_bench")


PROTOCOL_TIMEOUT_SECONDS = 60.0
MAX_PROTOCOL_LINES = 100_000
OUTPUT_QUEUE_LINES = 4096
READER_POLL_SECONDS = 0.05
READER_JOIN_SECONDS = 1.0


def _validate_timeout(timeout: float) -> None:
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("protocol_timeout must be a positive finite number")


def _enqueue_output(output_lines, item, stop_event) -> bool:
    """Bound insertion waits so a stopped consumer cannot strand its reader."""
    while not stop_event.is_set():
        try:
            output_lines.put(item, timeout=READER_POLL_SECONDS)
            return True
        except queue.Full:
            continue
    return False


def _collect_output(stream, output_lines, stop_event) -> None:
    """Read buffered output off-thread so consumers can enforce a deadline."""
    try:
        while not stop_event.is_set():
            raw_line = stream.readline()
            item = None if raw_line == "" else raw_line.rstrip("\r\n")
            if not _enqueue_output(output_lines, item, stop_event) or item is None:
                return
    except Exception as error:
        _enqueue_output(output_lines, error, stop_event)


def _start_reader(engine) -> None:
    engine._output_lines = queue.Queue(maxsize=OUTPUT_QUEUE_LINES)
    engine._reader_stop = threading.Event()
    engine._reader_thread = threading.Thread(
        target=_collect_output,
        args=(engine.process.stdout, engine._output_lines, engine._reader_stop),
        daemon=True, name="chess-uci-output",
    )
    engine._reader_thread.start()


def _stop_reader(engine) -> None:
    stop_event = getattr(engine, "_reader_stop", None)
    if stop_event is not None:
        stop_event.set()


def _join_reader(engine) -> None:
    reader = getattr(engine, "_reader_thread", None)
    if reader is not None:
        reader.join(timeout=READER_JOIN_SECONDS)
        if reader.is_alive():
            logger.warning("Engine reader has not exited after subprocess shutdown")


def _read_output(engine, timeout=None) -> str:
    if not engine.process or not engine.process.stdout:
        return ""
    try:
        line = engine._output_lines.get(timeout=timeout)
    except queue.Empty as error:
        raise TimeoutError("Engine output deadline exceeded") from error
    if line is None:
        raise RuntimeError("Engine stdout reached EOF before protocol target")
    if isinstance(line, Exception):
        raise RuntimeError("Engine output reader failed") from line
    return line


def _wait_for_output(engine, target: str) -> list[str]:
    deadline = time.monotonic() + engine.protocol_timeout
    lines = []
    while len(lines) < MAX_PROTOCOL_LINES:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Engine deadline exceeded waiting for {target}")
        try:
            line = engine._read_line(timeout=remaining)
        except TimeoutError as error:
            raise TimeoutError(f"Engine deadline exceeded waiting for {target}") from error
        if not line:
            continue
        lines.append(line)
        if line.split()[0] == target:
            return lines
    raise RuntimeError(f"Engine exceeded protocol line limit waiting for {target}")


class StockfishEngine:
    """Wrapper for Stockfish UCI engine subprocess."""

    def __init__(
        self,
        path: str = "/usr/games/stockfish",
        depth: int = 22,
        threads: int = 1,
        protocol_timeout: float = PROTOCOL_TIMEOUT_SECONDS,
    ):
        """Initialize Stockfish engine.

        Args:
            path: Path to Stockfish executable
            depth: Search depth for evaluation
            threads: Number of threads for Stockfish to use
        """
        _validate_timeout(protocol_timeout)
        self.protocol_timeout = protocol_timeout
        self.path = path
        self.depth = depth
        self.threads = threads
        self.process: subprocess.Popen | None = None
        self._start()

    def _start(self) -> None:
        """Start the Stockfish subprocess."""
        try:
            self.process = subprocess.Popen(
                [self.path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            _start_reader(self)
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

    def _read_line(self, timeout=None) -> str:
        """Read one line, distinguishing EOF from blank lines."""
        return _read_output(self, timeout)

    def _wait_for(self, target: str) -> list[str]:
        """Read through a UCI token under a finite deadline."""
        return _wait_for_output(self, target)

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
        _stop_reader(self)
        if self.process:
            try:
                self._send("quit")
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            finally:
                self.process = None
                _join_reader(self)
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
        protocol_timeout: float = PROTOCOL_TIMEOUT_SECONDS,
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
        _validate_timeout(protocol_timeout)
        self.protocol_timeout = protocol_timeout
        self.path = path
        self.weights = weights
        self.nodes = nodes
        self.backend = backend
        self.process: subprocess.Popen | None = None
        self._start()

    def _start(self) -> None:
        """Start the Lc0 subprocess."""
        try:
            self.process = subprocess.Popen(
                [self.path, f"--weights={self.weights}", f"--backend={self.backend}"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            _start_reader(self)
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

    def _read_line(self, timeout=None) -> str:
        """Read one line, distinguishing EOF from blank lines."""
        return _read_output(self, timeout)

    def _wait_for(self, target: str) -> list[str]:
        """Read through a UCI token under a finite deadline."""
        return _wait_for_output(self, target)

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
        _stop_reader(self)
        if self.process:
            try:
                self._send("quit")
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            finally:
                self.process = None
                _join_reader(self)
                logger.info("Lc0 closed")

    def __enter__(self) -> "Lc0Engine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
