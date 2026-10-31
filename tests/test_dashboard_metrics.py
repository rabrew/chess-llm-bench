"""Dashboard uses the shared corrected scoring definitions without research writes."""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))

import server  # noqa: E402
from src.metrics import load_results_df  # noqa: E402

FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MODEL = "llama3.2:3b"


def _record(**changes):
    return {"model": MODEL, "difficulty": "easy", "prompt_format": "fen_only", "fen": FEN,
            "position_id": 1, "t1_stockfish_eval": 1500, "t1_model_eval": 1200,
            "t1_absolute_error": 300, "t2_legal": True, "t2_move": "e4",
            "t2_eval_before": 500, "t2_cpl": 1000, "t3_score": 2,
            "t3_p1_side_correct": 1, "t3_explanation": "White — in my opinion this is quiet.",
            "theme": "pin", **changes}


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    path = tmp_path / "results.jsonl"
    database = tmp_path / "jobs.db"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE jobs (model TEXT, status TEXT)")
        conn.execute("INSERT INTO jobs VALUES (?, 'in_progress')", (MODEL,))
    monkeypatch.setattr(server, "EVALUATIONS_PATH", str(path))
    monkeypatch.setattr(server, "DB_PATH", str(database))
    server.app.config["TESTING"] = True
    return path, server.app.test_client()


def _write(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_dashboard_corrected_metrics_match_shared_implementation(dashboard, tmp_path):
    path, client = dashboard
    records = [_record(), _record(position_id=2, t1_stockfish_eval=9999,
                                 t1_model_eval=1000, t1_absolute_error=8999)]
    _write(path, records)
    shared = load_results_df(str(path), data_dir=str(tmp_path / "no-data"))
    data = client.get("/api/metrics").get_json()
    row = data["by_model"][0]
    assert row["t2_cpl_mean"] == pytest.approx(shared["t2_cpl_clamped"].mean())
    assert row["t3_score_mean"] == pytest.approx(shared["t3_score_v2"].mean())
    assert row["t1_mae_mean"] == pytest.approx(shared["t1_abs_error_excl_mate"].mean()) == 300
    assert data["metric_definitions"]["t3_score_mean"]["range"] == [0, 2]


def test_dashboard_excludes_nonapplicable_tasks_and_counts_missing_moves(dashboard):
    path, client = dashboard
    _write(path, [_record(), _record(prompt_format="eval_only", t2_legal=False, t3_score=0),
                  _record(prompt_format="explanation_only", t2_legal=False, t1_absolute_error=999),
                  _record(position_id=2, t2_legal=None, t2_move=None, t2_cpl=None)])
    row = client.get("/api/metrics").get_json()["by_model"][0]
    assert row["t2_legal_mean"] == pytest.approx(0.5)
    assert row["t1_mae_mean"] == 300


def test_dashboard_legacy_cpl_and_t3_are_unavailable_with_explicit_warning(dashboard):
    path, client = dashboard
    _write(path, [_record(t2_eval_before=None, t3_p1_side_correct=None, t3_explanation=None)])
    data = client.get("/api/metrics").get_json()
    row = data["by_model"][0]
    assert row["t2_cpl_mean"] is None and row["t3_score_mean"] is None
    assert data["metric_quality"]["excluded_cpl_missing_reference"] == 1
    assert data["metric_quality"]["excluded_t3_missing_corrected_score"] == 1
    assert data["warnings"]


def test_current_explanation_percentage_has_maximum_one_hundred(dashboard):
    path, client = dashboard
    _write(path, [_record(t3_explanation="White — the pin wins.")])
    data = client.get("/api/current").get_json()
    assert data["tier_data"]["easy"]["t3"] == 100


def test_dashboard_reuses_unchanged_snapshot_and_refreshes_changed_records(dashboard, monkeypatch):
    path, client = dashboard
    _write(path, [_record()])
    client.get("/api/metrics")
    calls = []
    original = server._metric_values
    monkeypatch.setattr(server, "_metric_values", lambda record, quality: (calls.append(record), original(record, quality))[1])
    client.get("/api/metrics")
    assert calls == []
    with path.open("a") as output:
        output.write(json.dumps(_record(position_id=2)) + "\n")
    client.get("/api/metrics")
    assert len(calls) == 2
    assert client.get("/api/current").get_json()["total_done"] == 2


def test_snapshot_rebuilds_after_same_inode_larger_rewrite(dashboard):
    path, client = dashboard
    _write(path, [_record()])
    client.get("/api/metrics")
    inode = path.stat().st_ino
    _write(path, [_record(model="qwen2.5:7b", position_id=i) for i in (10, 11)])
    assert path.stat().st_ino == inode
    data = client.get("/api/metrics").get_json()
    assert [row["model"] for row in data["by_model"]] == ["qwen2.5:7b"]
    assert server._metric_snapshot()["counts"]["qwen2.5:7b"]["easy"] == 2


def test_snapshot_waits_for_partial_line_completion(dashboard):
    path, client = dashboard
    _write(path, [_record()])
    client.get("/api/metrics")
    with path.open("a") as output:
        output.write(json.dumps(_record(position_id=2)))
    assert client.get("/api/current").get_json()["total_done"] == 1
    with path.open("a") as output:
        output.write("\n")
    assert client.get("/api/current").get_json()["total_done"] == 2


def test_snapshot_rebuilds_after_shrink(dashboard):
    path, client = dashboard
    _write(path, [_record(position_id=i) for i in (1, 2)])
    assert client.get("/api/current").get_json()["total_done"] == 2
    _write(path, [_record(t2_legal=False)])
    assert client.get("/api/current").get_json()["total_done"] == 1
    assert client.get("/api/metrics").get_json()["by_model"][0]["t2_legal_mean"] == 0


def test_snapshot_rebuilds_after_atomic_replacement(dashboard):
    path, client = dashboard
    _write(path, [_record()])
    client.get("/api/metrics")
    replacement = path.with_name("replacement.jsonl")
    _write(replacement, [_record(t2_legal=False)])
    replacement.replace(path)
    assert client.get("/api/metrics").get_json()["by_model"][0]["t2_legal_mean"] == 0
