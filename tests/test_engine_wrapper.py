"""Tests for src/engine_wrapper.py (mocked subprocess)."""

import subprocess
from io import StringIO
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.engine_wrapper import StockfishEngine, Lc0Engine


def _make_mock_process(stdout_lines: list[str]):
    """Create a mock subprocess.Popen with given stdout lines."""
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stdout = MagicMock()
    proc.stdout.readline = MagicMock(side_effect=[line + "\n" for line in stdout_lines] + [""])
    proc.wait = MagicMock()
    proc.kill = MagicMock()
    return proc


STOCKFISH_INIT_LINES = ["uciok", "readyok"]
STOCKFISH_EVAL_LINES = [
    "info depth 22 score cp 45 pv e4",
    "bestmove e2e4 ponder d7d5",
]

LC0_INIT_LINES = ["uciok", "readyok"]
LC0_EVAL_LINES = [
    "info depth 1 score cp 30 pv e2e4",
    "bestmove e2e4",
]


class TestStockfishEngine:
    @patch("subprocess.Popen")
    def test_init_success(self, mock_popen):
        mock_popen.return_value = _make_mock_process(STOCKFISH_INIT_LINES)
        engine = StockfishEngine(path="/fake/stockfish")
        assert engine.depth == 22

    @patch("subprocess.Popen")
    def test_init_file_not_found(self, mock_popen):
        mock_popen.side_effect = FileNotFoundError
        with pytest.raises(RuntimeError, match="not found"):
            StockfishEngine(path="/nonexistent/stockfish")

    @patch("subprocess.Popen")
    def test_init_other_exception(self, mock_popen):
        mock_popen.side_effect = OSError("permission denied")
        with pytest.raises(RuntimeError, match="Failed to start"):
            StockfishEngine(path="/fake/stockfish")

    @patch("subprocess.Popen")
    def test_evaluate_returns_dict(self, mock_popen):
        all_lines = STOCKFISH_INIT_LINES + STOCKFISH_EVAL_LINES
        mock_popen.return_value = _make_mock_process(all_lines)
        engine = StockfishEngine(path="/fake/stockfish")
        result = engine.evaluate("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
        assert "eval" in result
        assert "best_move" in result
        assert "mate" in result

    @patch("subprocess.Popen")
    def test_evaluate_adjusts_for_black_to_move(self, mock_popen):
        # FEN with black to move — eval should be negated
        all_lines = STOCKFISH_INIT_LINES + [
            "info depth 1 score cp 45 pv e7e5",
            "bestmove e7e5",
        ]
        mock_popen.return_value = _make_mock_process(all_lines)
        engine = StockfishEngine(path="/fake/stockfish")
        # Black to move position
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        result = engine.evaluate(fen)
        assert result["eval"] == -45  # negated for black perspective

    @patch("subprocess.Popen")
    def test_evaluate_mate_score(self, mock_popen):
        all_lines = STOCKFISH_INIT_LINES + [
            "info depth 1 score mate 2 pv e4",
            "bestmove e2e4",
        ]
        mock_popen.return_value = _make_mock_process(all_lines)
        engine = StockfishEngine(path="/fake/stockfish")
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        result = engine.evaluate(fen)
        assert result["mate"] == 2

    @patch("subprocess.Popen")
    def test_evaluate_negative_mate(self, mock_popen):
        all_lines = STOCKFISH_INIT_LINES + [
            "info depth 1 score mate -1 pv e4",
            "bestmove e2e4",
        ]
        mock_popen.return_value = _make_mock_process(all_lines)
        engine = StockfishEngine(path="/fake/stockfish")
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        result = engine.evaluate(fen)
        assert result["mate"] is not None

    @patch("subprocess.Popen")
    def test_evaluate_none_bestmove(self, mock_popen):
        all_lines = STOCKFISH_INIT_LINES + [
            "info depth 1 score cp 0",
            "bestmove (none)",
        ]
        mock_popen.return_value = _make_mock_process(all_lines)
        engine = StockfishEngine(path="/fake/stockfish")
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        result = engine.evaluate(fen)
        assert result["best_move"] is None

    @patch("subprocess.Popen")
    def test_evaluate_retries_on_exception(self, mock_popen):
        """On first evaluate internal failure, engine restarts and retries."""
        proc1 = _make_mock_process(STOCKFISH_INIT_LINES)
        proc1.stdin.write.side_effect = [None, None, None, BrokenPipeError, None]
        proc2_lines = STOCKFISH_INIT_LINES + STOCKFISH_EVAL_LINES
        proc2 = _make_mock_process(proc2_lines)

        mock_popen.side_effect = [proc1, proc2]
        engine = StockfishEngine(path="/fake/stockfish")
        # Force failure on internal eval, restart, succeed
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        with patch.object(engine, "_evaluate_internal", side_effect=[Exception("fail"),
                          {"eval": 45, "best_move": "e4", "mate": None}]):
            result = engine.evaluate(fen)
            assert result["eval"] == 45

    @patch("subprocess.Popen")
    def test_is_legal_move_true(self, mock_popen):
        mock_popen.return_value = _make_mock_process(STOCKFISH_INIT_LINES)
        engine = StockfishEngine(path="/fake/stockfish")
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        assert engine.is_legal_move(fen, "e4") is True

    @patch("subprocess.Popen")
    def test_is_legal_move_false(self, mock_popen):
        mock_popen.return_value = _make_mock_process(STOCKFISH_INIT_LINES)
        engine = StockfishEngine(path="/fake/stockfish")
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        assert engine.is_legal_move(fen, "e5") is False

    @patch("subprocess.Popen")
    def test_is_legal_move_invalid(self, mock_popen):
        mock_popen.return_value = _make_mock_process(STOCKFISH_INIT_LINES)
        engine = StockfishEngine(path="/fake/stockfish")
        assert engine.is_legal_move("invalid-fen", "e4") is False

    @patch("subprocess.Popen")
    def test_evaluate_after_move(self, mock_popen):
        lines = STOCKFISH_INIT_LINES + STOCKFISH_EVAL_LINES
        mock_popen.return_value = _make_mock_process(lines)
        engine = StockfishEngine(path="/fake/stockfish")
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        with patch.object(engine, "evaluate", return_value={"eval": 10, "best_move": "e5", "mate": None}):
            result = engine.evaluate_after_move(fen, "e4")
            assert result == 10

    @patch("subprocess.Popen")
    def test_close(self, mock_popen):
        mock_popen.return_value = _make_mock_process(STOCKFISH_INIT_LINES)
        engine = StockfishEngine(path="/fake/stockfish")
        engine.close()
        assert engine.process is None

    @patch("subprocess.Popen")
    def test_close_when_no_process(self, mock_popen):
        mock_popen.return_value = _make_mock_process(STOCKFISH_INIT_LINES)
        engine = StockfishEngine(path="/fake/stockfish")
        engine.process = None
        engine.close()  # Should not raise

    @patch("subprocess.Popen")
    def test_close_handles_exception(self, mock_popen):
        proc = _make_mock_process(STOCKFISH_INIT_LINES)
        proc.wait.side_effect = Exception("timeout")
        mock_popen.return_value = proc
        engine = StockfishEngine(path="/fake/stockfish")
        engine.close()  # Should not raise, should kill
        proc.kill.assert_called_once()

    @patch("subprocess.Popen")
    def test_context_manager(self, mock_popen):
        mock_popen.return_value = _make_mock_process(STOCKFISH_INIT_LINES)
        with StockfishEngine(path="/fake/stockfish") as engine:
            assert engine is not None
        assert engine.process is None

    @patch("subprocess.Popen")
    def test_send_no_process(self, mock_popen):
        mock_popen.return_value = _make_mock_process(STOCKFISH_INIT_LINES)
        engine = StockfishEngine(path="/fake/stockfish")
        engine.process = None
        engine._send("quit")  # Should not raise

    @patch("subprocess.Popen")
    def test_read_line_no_process(self, mock_popen):
        mock_popen.return_value = _make_mock_process(STOCKFISH_INIT_LINES)
        engine = StockfishEngine(path="/fake/stockfish")
        engine.process = None
        result = engine._read_line()
        assert result == ""

    @patch("subprocess.Popen")
    def test_restart(self, mock_popen):
        proc1 = _make_mock_process(STOCKFISH_INIT_LINES)
        proc2 = _make_mock_process(STOCKFISH_INIT_LINES)
        mock_popen.side_effect = [proc1, proc2]
        engine = StockfishEngine(path="/fake/stockfish")
        engine._restart()
        assert mock_popen.call_count == 2


class TestLc0Engine:
    @patch("subprocess.Popen")
    def test_init_success(self, mock_popen):
        mock_popen.return_value = _make_mock_process(LC0_INIT_LINES)
        engine = Lc0Engine(path="/fake/lc0", weights="/fake/weights.pb")
        assert engine.nodes == 1000

    @patch("subprocess.Popen")
    def test_init_file_not_found(self, mock_popen):
        mock_popen.side_effect = FileNotFoundError
        with pytest.raises(RuntimeError, match="not found"):
            Lc0Engine(path="/nonexistent/lc0", weights="/fake/weights.pb")

    @patch("subprocess.Popen")
    def test_init_other_exception(self, mock_popen):
        mock_popen.side_effect = OSError("fail")
        with pytest.raises(RuntimeError, match="Failed to start"):
            Lc0Engine(path="/fake/lc0", weights="/fake/weights.pb")

    @patch("subprocess.Popen")
    def test_evaluate_returns_dict(self, mock_popen):
        all_lines = LC0_INIT_LINES + LC0_EVAL_LINES
        mock_popen.return_value = _make_mock_process(all_lines)
        engine = Lc0Engine(path="/fake/lc0", weights="/fake/w.pb")
        with patch.object(engine, "_evaluate_internal", return_value={"eval": 30, "best_move": "e4", "mate": None}):
            result = engine.evaluate("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        assert "eval" in result

    @patch("subprocess.Popen")
    def test_evaluate_retries_on_exception(self, mock_popen):
        mock_popen.return_value = _make_mock_process(LC0_INIT_LINES)
        engine = Lc0Engine(path="/fake/lc0", weights="/fake/w.pb")
        with patch.object(engine, "_evaluate_internal",
                          side_effect=[Exception("fail"),
                                       {"eval": 30, "best_move": "e4", "mate": None}]):
            result = engine.evaluate("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            assert result["eval"] == 30

    @patch("subprocess.Popen")
    def test_close(self, mock_popen):
        mock_popen.return_value = _make_mock_process(LC0_INIT_LINES)
        engine = Lc0Engine(path="/fake/lc0", weights="/fake/w.pb")
        engine.close()
        assert engine.process is None

    @patch("subprocess.Popen")
    def test_close_when_no_process(self, mock_popen):
        mock_popen.return_value = _make_mock_process(LC0_INIT_LINES)
        engine = Lc0Engine(path="/fake/lc0", weights="/fake/w.pb")
        engine.process = None
        engine.close()  # Should not raise

    @patch("subprocess.Popen")
    def test_close_handles_exception(self, mock_popen):
        proc = _make_mock_process(LC0_INIT_LINES)
        proc.wait.side_effect = Exception("timeout")
        mock_popen.return_value = proc
        engine = Lc0Engine(path="/fake/lc0", weights="/fake/w.pb")
        engine.close()
        proc.kill.assert_called_once()

    @patch("subprocess.Popen")
    def test_context_manager(self, mock_popen):
        mock_popen.return_value = _make_mock_process(LC0_INIT_LINES)
        with Lc0Engine(path="/fake/lc0", weights="/fake/w.pb") as engine:
            assert engine is not None
        assert engine.process is None

    @patch("subprocess.Popen")
    def test_send_no_process(self, mock_popen):
        mock_popen.return_value = _make_mock_process(LC0_INIT_LINES)
        engine = Lc0Engine(path="/fake/lc0", weights="/fake/w.pb")
        engine.process = None
        engine._send("quit")  # Should not raise

    @patch("subprocess.Popen")
    def test_read_line_no_process(self, mock_popen):
        mock_popen.return_value = _make_mock_process(LC0_INIT_LINES)
        engine = Lc0Engine(path="/fake/lc0", weights="/fake/w.pb")
        engine.process = None
        result = engine._read_line()
        assert result == ""

    @patch("subprocess.Popen")
    def test_evaluate_internal_black_to_move(self, mock_popen):
        all_lines = LC0_INIT_LINES + [
            "info depth 1 score cp 30 pv e7e5",
            "bestmove e7e5",
        ]
        mock_popen.return_value = _make_mock_process(all_lines)
        engine = Lc0Engine(path="/fake/lc0", weights="/fake/w.pb")
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        result = engine._evaluate_internal(fen)
        assert result["eval"] == -30  # negated for black perspective

    @patch("subprocess.Popen")
    def test_evaluate_internal_mate_negative(self, mock_popen):
        all_lines = LC0_INIT_LINES + [
            "info depth 1 score mate -2 pv e7e5",
            "bestmove e7e5",
        ]
        mock_popen.return_value = _make_mock_process(all_lines)
        engine = Lc0Engine(path="/fake/lc0", weights="/fake/w.pb")
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        result = engine._evaluate_internal(fen)
        assert result["mate"] is not None

    @patch("subprocess.Popen")
    def test_evaluate_internal_none_bestmove(self, mock_popen):
        all_lines = LC0_INIT_LINES + [
            "info depth 1 score cp 0",
            "bestmove (none)",
        ]
        mock_popen.return_value = _make_mock_process(all_lines)
        engine = Lc0Engine(path="/fake/lc0", weights="/fake/w.pb")
        result = engine._evaluate_internal("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        assert result["best_move"] is None

    @patch("subprocess.Popen")
    def test_restart(self, mock_popen):
        proc1 = _make_mock_process(LC0_INIT_LINES)
        proc2 = _make_mock_process(LC0_INIT_LINES)
        mock_popen.side_effect = [proc1, proc2]
        engine = Lc0Engine(path="/fake/lc0", weights="/fake/w.pb")
        engine._restart()
        assert mock_popen.call_count == 2
