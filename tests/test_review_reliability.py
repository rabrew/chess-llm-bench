"""Regression cases from the static audit; no engines or model services."""

import io
import json
import os
from unittest.mock import patch

import chess
import pandas as pd
import pytest

from src.data_loader import DataLoader
from src.engine_wrapper import Lc0Engine, StockfishEngine
from src.position_generator import validate_position
from src.job_queue import JobQueue
from src.worker import Worker


@pytest.mark.parametrize("position, expected", [
    ({}, True), ({"stockfish_eval": 0, "stockfish_best_move": None}, True),
    ({"stockfish_eval": None, "stockfish_best_move": "e4"}, True),
    ({"stockfish_eval": 0, "stockfish_best_move": ""}, True),
    ({"stockfish_eval": 0, "stockfish_best_move": "e4"}, False),
])
def test_precompute_does_not_skip_null_reference_values(position, expected):
    from scripts.precompute_stockfish import needs_evaluation

    assert needs_evaluation(position) is expected


@pytest.mark.parametrize("engine_type", [StockfishEngine, Lc0Engine])
def test_protocol_eof_raises_instead_of_waiting_forever(engine_type):
    engine = engine_type.__new__(engine_type)
    engine.process = type("Process", (), {"stdout": io.StringIO("")})()
    engine.timeout = 0.1
    with pytest.raises(EOFError):
        engine._wait_for("bestmove")


@pytest.mark.parametrize("engine_type", [StockfishEngine, Lc0Engine])
def test_protocol_wait_times_out_without_a_line(engine_type):
    read_fd, write_fd = os.pipe()
    try:
        with os.fdopen(read_fd, "r") as stream:
            engine = engine_type.__new__(engine_type)
            engine.process = type("Process", (), {"stdout": stream})()
            engine.timeout = 0.01
            with pytest.raises(TimeoutError):
                engine._wait_for("bestmove")
    finally:
        os.close(write_fd)


def test_protocol_handles_blank_and_buffered_lines():
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"\ninfo depth 1 score cp 5\nbestmove e2e4\n")
        with os.fdopen(read_fd, "r") as stream:
            engine = StockfishEngine.__new__(StockfishEngine)
            engine.process = type("Process", (), {"stdout": stream})()
            engine.timeout = 0.1
            assert engine._wait_for("bestmove")[-1] == "bestmove e2e4"
    finally:
        os.close(write_fd)


def test_clear_cache_invalidates_id_lookup(tmp_path):
    path = tmp_path / "easy.json"
    path.write_text(json.dumps([{"id": 1, "fen": "old"}]))
    loader = DataLoader(str(tmp_path))
    assert loader.get_by_id(1)["fen"] == "old"
    path.write_text(json.dumps([{"id": 1, "fen": "new"}]))
    loader.clear_cache()
    assert loader.get_by_id(1)["fen"] == "new"


def test_worker_dry_run_claims_and_completes_only_snapshot(tmp_path):
    database = tmp_path / "jobs.db"
    queue = JobQueue(str(database))
    queue.insert_job({"job_id": "job", "position_id": 1, "model": "qwen2.5:7b"})
    queue.close()
    config = {"paths": {"jobs_db": str(database), "data_dir": str(tmp_path),
                         "results_file": str(tmp_path / "output" / "results.jsonl")}}
    with patch("src.worker.OllamaClient"):
        worker = Worker("dry", config, dry_run=True)
        claimed = worker.job_queue.claim_job("dry")
        worker.job_queue.complete_job(claimed["job_id"])
        worker.job_queue.close()
    queue = JobQueue(str(database))
    assert queue.get_progress()["pending"] == 1
    assert queue.get_progress()["done"] == 0
    queue.close()
    assert not (tmp_path / "output").exists()


def test_dry_run_recovers_stale_jobs_only_in_memory(tmp_path):
    database = tmp_path / "jobs.db"
    queue = JobQueue(str(database))
    queue.insert_job({"job_id": "job", "position_id": 1, "model": "qwen2.5:7b"})
    queue.claim_job("old")
    queue._connect().execute("UPDATE jobs SET claimed_at='2000-01-01 00:00:00'")
    queue._connect().commit()
    queue.close()
    config = {"paths": {"jobs_db": str(database), "results_file": str(tmp_path / "out.jsonl")}}
    with patch("src.worker.OllamaClient"):
        worker = Worker("dry", config, dry_run=True)
        assert worker.job_queue.claim_job("dry")["job_id"] == "job"
        worker.job_queue.close()
    queue = JobQueue(str(database))
    assert queue.get_job("job")["worker_id"] == "old"
    assert queue.get_job("job")["status"] == "in_progress"
    queue.close()


def test_snapshot_of_missing_queue_does_not_create_database(tmp_path):
    path = tmp_path / "missing" / "jobs.db"
    with pytest.raises(FileNotFoundError):
        JobQueue.snapshot(str(path))
    assert not path.parent.exists()


def test_enrichment_refuses_loaded_models_without_loading_engine(monkeypatch):
    from scripts import enrich_cpl

    response = type("Response", (), {
        "status_code": 200, "raise_for_status": lambda self: None,
        "json": lambda self: {"models": [{"name": "qwen2.5:7b"}]},
    })()
    monkeypatch.setattr(enrich_cpl.requests, "get", lambda *args, **kwargs: response)
    with pytest.raises(RuntimeError, match="loaded"):
        enrich_cpl.ensure_ollama_unloaded("http://localhost:11434")


def test_enrichment_refuses_unknown_service_state(monkeypatch):
    from scripts import enrich_cpl
    import requests

    def refused(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(enrich_cpl.requests, "get", refused)
    with pytest.raises(RuntimeError, match="verify"):
        enrich_cpl.ensure_ollama_unloaded("http://localhost:11434")


def test_enrichment_accepts_empty_loaded_model_list(monkeypatch):
    from scripts import enrich_cpl

    response = type("Response", (), {
        "raise_for_status": lambda self: None, "json": lambda self: {"models": []},
    })()
    monkeypatch.setattr(enrich_cpl.requests, "get", lambda *args, **kwargs: response)
    enrich_cpl.ensure_ollama_unloaded("http://localhost:11434")


def test_enrichment_main_checks_gpu_before_constructing_engine(monkeypatch):
    from scripts import enrich_cpl

    monkeypatch.setattr("sys.argv", ["enrich_cpl.py"])
    monkeypatch.setattr(enrich_cpl, "load_config", lambda path: {
        "lc0": {"binary": "fake-engine", "weights": "fake-weights"},
    })
    with patch.object(enrich_cpl, "ensure_ollama_unloaded", side_effect=RuntimeError("loaded")), \
            patch.object(enrich_cpl, "Lc0Engine") as engine:
        with pytest.raises(RuntimeError, match="loaded"):
            enrich_cpl.main()
        engine.assert_not_called()


@pytest.mark.parametrize("fen", [
    "7k/7p/8/8/8/8/P7/K7 w - - 0 1",
    "7K/7P/8/8/8/8/p7/k7 b - - 0 1",
])
def test_king_on_square_zero_is_present(fen):
    assert validate_position(chess.Board(fen))


@pytest.mark.parametrize("indexes", [[0, 1], [5, 8], [0, 0]])
def test_heatmap_colours_follow_ranked_labels(tmp_path, indexes):
    from scripts import generate_plots as plots

    frame = pd.DataFrame({
        "model": ["weak", "strong"], "family": ["LLaMA 3"] * 2, "size_b": [3, 8],
        "t1_headline_mean": [0.0, 0.5], "t2_legal_headline_mean": [0.5, 1.0],
        "t2_cpl_headline_mean": [900, 100], "t3_score_headline_mean": [0.1, 1.0],
    }, index=indexes)
    original_close = plots.plt.close
    with patch.object(plots.plt, "close"):
        try:
            plots.plot_summary_heatmap(frame, tmp_path)
            axes = plots.plt.gcf().axes[0]
            assert axes.get_yticklabels()[0].get_text().startswith("strong")
            assert list(axes.images[0].get_array()[0]) == [1.0] * 4
            assert list(axes.images[0].get_array()[1]) == [0.0] * 4
        finally:
            original_close("all")
