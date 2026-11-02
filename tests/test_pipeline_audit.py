"""CPU-only regressions for audited pipeline defects."""
import io
import json
import math
import queue
from unittest.mock import MagicMock, patch
import chess
import pandas as pd
import pytest
from src.engine_wrapper import StockfishEngine, Lc0Engine
from src.dataset_builder import LichessPuzzleFetcher, build_dataset, DIFFICULTY_TIERS
from src.position_generator import validate_position
from scripts import precompute_stockfish as stockfish
from scripts import precompute_lc0_batch_v2 as lc0

@pytest.mark.parametrize("engine_class,kwargs", [
    (StockfishEngine, {"path": "/fake/stockfish"}),
    (Lc0Engine, {"path": "/fake/lc0", "weights": "/fake/weights"}),
])
def test_engine_eof_fails_bounded(engine_class, kwargs):
    process = MagicMock()
    process.stdout = io.StringIO("\n")
    with patch("src.engine_wrapper.subprocess.Popen", return_value=process):
        with pytest.raises(RuntimeError, match="EOF"):
            engine_class(**kwargs, protocol_timeout=0.05)
    process.wait.assert_called()

@pytest.mark.parametrize("engine_class", [StockfishEngine, Lc0Engine])
def test_live_silent_engine_times_out(engine_class):
    engine = engine_class.__new__(engine_class)
    engine.process = MagicMock()
    engine.protocol_timeout = 0.01
    engine._output_lines = queue.Queue()
    with pytest.raises(TimeoutError, match="bestmove"):
        engine._wait_for("bestmove")

@pytest.mark.parametrize("timeout", [0, -1, math.inf, math.nan])
def test_invalid_protocol_timeout(timeout):
    with pytest.raises(ValueError, match="protocol_timeout"):
        StockfishEngine(protocol_timeout=timeout)

@pytest.mark.parametrize("fen", [
    "7k/8/8/8/8/8/P7/K7 w - - 0 1",
    "7K/8/8/8/8/8/p7/k7 b - - 0 1",
])
def test_a1_king_is_valid(fen):
    assert validate_position(chess.Board(fen))

def _puzzles_per_tier(count):
    return {tier: [
        {"fen": f"unique-{tier}-{index}", "difficulty": tier, "source": "lichess_puzzles"}
        for index in range(count)
    ] for tier in DIFFICULTY_TIERS}

def test_default_build_requests_full_puzzle_tiers(tmp_path):
    config = {"benchmark": {"max_positions_per_tier": 3}, "dataset": {"lichess_csv_path": "fake.csv"}}
    with patch.object(LichessPuzzleFetcher, "fetch_from_csv", return_value=_puzzles_per_tier(3)) as fetch, \
         patch("src.dataset_builder.generate_positions") as generated:
        built = build_dataset(config, str(tmp_path))
    fetch.assert_called_once_with(count_per_tier=3, seed=42)
    generated.assert_not_called()
    assert {tier: len(rows) for tier, rows in built.items()} == dict.fromkeys(DIFFICULTY_TIERS, 3)
    assert all(row["source"] == "lichess_puzzles" for rows in built.values() for row in rows)

def test_short_tier_fails_before_writing(tmp_path):
    config = {"benchmark": {"max_positions_per_tier": 3}, "dataset": {"lichess_csv_path": "fake.csv"}}
    with patch.object(LichessPuzzleFetcher, "fetch_from_csv", return_value=_puzzles_per_tier(2)):
        with pytest.raises(ValueError, match="tier"):
            build_dataset(config, str(tmp_path))
    assert not list(tmp_path.glob("*.json"))

def test_parallel_csv_results_keep_sample_order(tmp_path):
    csv_path = tmp_path / "puzzles.csv"
    pd.DataFrame([{"FEN": f"fen-{index}", "Rating": 500, "Themes": "fork"} for index in range(6)]).to_csv(csv_path, index=False)
    expected = pd.read_csv(csv_path).sample(n=6, random_state=42)["FEN"].tolist()[:2]
    executor = MagicMock()
    executor.__enter__.return_value = executor
    executor.map.side_effect = lambda function, rows, **kwargs: iter(
        {"fen": row["FEN"], "difficulty": "easy"} for row in rows
    )
    # Reversed future completion would pick a different cohort in the old implementation.
    futures = []
    def submit(function, row):
        future = MagicMock()
        future.result.return_value = {"fen": row["FEN"], "difficulty": "easy"}
        futures.append(future)
        return future
    executor.submit.side_effect = submit
    with patch("src.dataset_builder.ProcessPoolExecutor", return_value=executor), \
         patch("src.dataset_builder.as_completed", side_effect=lambda mapping: reversed(list(mapping)), create=True):
        selected = LichessPuzzleFetcher(csv_path=str(csv_path)).fetch_from_csv(count_per_tier=2)
    assert [row["fen"] for row in selected["easy"]] == expected

def test_stockfish_completion_requires_source_budget_and_legal_move():
    position = {"fen": chess.STARTING_FEN, "stockfish_eval": 30, "stockfish_best_move": "e4",
                "reference_provenance": {"engine": "stockfish", "depth": 15}}
    assert not stockfish.needs_stockfish_evaluation(position, 15)
    for changes in [
        {"stockfish_best_move": None}, {"stockfish_eval": float("nan")},
        {"stockfish_best_move": "Qa5"}, {"reference_provenance": {"engine": "lc0-onnx", "depth": 15}},
        {"reference_provenance": {"engine": "stockfish", "depth": 22}}, {"reference_provenance": None},
    ]:
        assert stockfish.needs_stockfish_evaluation({**position, **changes}, 15)

def test_lc0_result_records_actual_source():
    record = {}
    lc0.store_lc0_evaluation(record, 123, "model.onnx")
    assert record["lc0_eval"] == 123
    assert record["stockfish_eval"] == 123
    assert record["reference_provenance"]["engine"] == "lc0-onnx"
    assert record["reference_provenance"]["search_nodes"] == 0
    assert stockfish.needs_stockfish_evaluation({**record, "fen": chess.STARTING_FEN}, 15)

@pytest.mark.parametrize("engine_class", [StockfishEngine, Lc0Engine])
def test_blank_lines_do_not_count_as_eof(engine_class):
    engine = engine_class.__new__(engine_class)
    engine.process = MagicMock()
    engine.protocol_timeout = 0.05
    engine._output_lines = queue.Queue()
    for line in ["", "info depth 1", "bestmove e2e4"]:
        engine._output_lines.put(line)
    assert engine._wait_for("bestmove") == ["info depth 1", "bestmove e2e4"]

def test_lc0_does_not_certify_legacy_or_other_models():
    assert lc0.needs_lc0_evaluation({"stockfish_eval": 12}, "net.onnx")
    position = {}
    lc0.store_lc0_evaluation(position, 50, "net.onnx")
    assert not lc0.needs_lc0_evaluation(position, "net.onnx")
    assert lc0.needs_lc0_evaluation(position, "other.onnx")
    assert lc0.needs_lc0_evaluation({**position, "lc0_eval": None}, "net.onnx")

def test_preprocessors_preserve_other_source_measurement():
    position = {"stockfish_eval": 24, "stockfish_best_move": "e4",
                "reference_provenance": {"engine": "stockfish", "depth": 15}}
    lc0.store_lc0_evaluation(position, 48, "net.onnx")
    assert position["stockfish_reference_eval"] == 24
    assert position["stockfish_reference_best_move"] == "e4"
    stockfish.store_stockfish_evaluation(position, 25, "d4", 15, "/fake/sf")
    assert position["lc0_eval"] == 48
    assert position["stockfish_eval"] == 25
    assert position["reference_provenance"]["engine"] == "stockfish"

def test_unknown_dataset_design_rejected(tmp_path):
    with pytest.raises(ValueError, match="design"):
        build_dataset({"dataset": {"design": "typo"}}, str(tmp_path))
    assert not list(tmp_path.iterdir())

@pytest.mark.parametrize("engine_class", [StockfishEngine, Lc0Engine])
def test_close_cancels_flooded_output_reader(engine_class):
    import threading
    import time
    from src.engine_wrapper import _start_reader
    engine = engine_class.__new__(engine_class)
    engine.process = MagicMock()
    engine.process.stdout = io.StringIO("info flood\n" * 5000)
    existing_threads = set(threading.enumerate())
    _start_reader(engine)
    reader = next(thread for thread in threading.enumerate() if thread not in existing_threads)
    try:
        deadline = time.monotonic() + 1
        while not engine._output_lines.full() and time.monotonic() < deadline:
            threading.Event().wait(0.001)
        assert engine._output_lines.full()
        engine.close()
        reader.join(timeout=0.2)
        assert not reader.is_alive()
    finally:
        # Even the pre-fix regression must not leave its own daemon behind.
        while reader.is_alive():
            try:
                engine._output_lines.get(timeout=0.01)
            except queue.Empty:
                break
        reader.join(timeout=0.2)
