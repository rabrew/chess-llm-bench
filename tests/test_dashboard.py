"""Tests for the dashboard Flask server."""
import json
import os
import sys
import sqlite3

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dashboard"))


@pytest.fixture
def sample_db(tmp_path):
    db_path = tmp_path / "jobs.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            position_id INTEGER
        )
    """)
    rows = [
        ("j1", "llama3.2:3b", "done", 100),
        ("j2", "llama3.2:3b", "done", 200),
        ("j3", "llama3.2:3b", "pending", 300),
        ("j4", "qwen2.5:7b", "in_progress", 150),
        ("j5", "qwen2.5:7b", "pending", 400),
        ("j6", "mistral:7b", "pending", 500),
    ]
    conn.executemany("INSERT INTO jobs VALUES (?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return str(db_path)


@pytest.fixture
def sample_jsonl(tmp_path):
    """Create a minimal evaluations.jsonl for testing."""
    records = [
        # llama3.2:3b easy — illegal move
        {"model": "llama3.2:3b", "difficulty": "easy", "t1_direction_correct": False,
         "t2_legal": False, "t2_cpl": None, "t3_score": 1.0},
        {"model": "llama3.2:3b", "difficulty": "easy", "t1_direction_correct": True,
         "t2_legal": False, "t2_cpl": None, "t3_score": 0.0},
        # qwen2.5:7b easy — 1 legal out of 4 (25%)
        {"model": "qwen2.5:7b", "difficulty": "easy", "t1_direction_correct": True,
         "t2_legal": True,  "t2_cpl": 452.0, "t3_score": 0.5},
        {"model": "qwen2.5:7b", "difficulty": "easy", "t1_direction_correct": False,
         "t2_legal": False, "t2_cpl": None, "t3_score": 0.3},
        {"model": "qwen2.5:7b", "difficulty": "easy", "t1_direction_correct": False,
         "t2_legal": False, "t2_cpl": None, "t3_score": 0.4},
        {"model": "qwen2.5:7b", "difficulty": "easy", "t1_direction_correct": True,
         "t2_legal": False, "t2_cpl": None, "t3_score": 0.2},
        # qwen2.5:7b hard
        {"model": "qwen2.5:7b", "difficulty": "hard", "t1_direction_correct": False,
         "t2_legal": True,  "t2_cpl": 6476.0, "t3_score": 0.4},
        {"model": "qwen2.5:7b", "difficulty": "hard", "t1_direction_correct": False,
         "t2_legal": False, "t2_cpl": None,   "t3_score": 0.4},
        {"model": "qwen2.5:7b", "difficulty": "hard", "t1_direction_correct": True,
         "t2_legal": False, "t2_cpl": None,   "t3_score": 0.5},
        {"model": "qwen2.5:7b", "difficulty": "hard", "t1_direction_correct": False,
         "t2_legal": False, "t2_cpl": None,   "t3_score": 0.3},
        {"model": "qwen2.5:7b", "difficulty": "hard", "t1_direction_correct": False,
         "t2_legal": False, "t2_cpl": None,   "t3_score": 0.3},
        {"model": "qwen2.5:7b", "difficulty": "hard", "t1_direction_correct": False,
         "t2_legal": False, "t2_cpl": None,   "t3_score": 0.3},
        {"model": "qwen2.5:7b", "difficulty": "hard", "t1_direction_correct": False,
         "t2_legal": False, "t2_cpl": None,   "t3_score": 0.4},
        {"model": "qwen2.5:7b", "difficulty": "hard", "t1_direction_correct": True,
         "t2_legal": False, "t2_cpl": None,   "t3_score": 0.4},
    ]
    path = tmp_path / "evaluations.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return str(path)


@pytest.fixture
def app(sample_db, sample_jsonl):
    import server
    server.DB_PATH = sample_db
    server.EVALUATIONS_PATH = sample_jsonl
    server.app.config["TESTING"] = True
    return server.app.test_client()


# --- /api/progress ---

def test_progress_total(app):
    resp = app.get("/api/progress")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["total"] == 6
    assert data["done"] == 2
    assert data["overall_pct"] == pytest.approx(33, abs=1)


def test_progress_per_model(app):
    resp = app.get("/api/progress")
    data = json.loads(resp.data)
    models = {m["model"]: m for m in data["models"]}
    assert models["llama3.2:3b"]["done"] == 2
    assert models["llama3.2:3b"]["total"] == 3
    assert models["qwen2.5:7b"]["in_progress"] == 1


def test_progress_currently_running(app):
    resp = app.get("/api/progress")
    data = json.loads(resp.data)
    models = {m["model"]: m for m in data["models"]}
    assert models["qwen2.5:7b"]["running"] is True
    assert models["llama3.2:3b"]["running"] is False


def test_progress_not_started_model(app):
    resp = app.get("/api/progress")
    data = json.loads(resp.data)
    models = {m["model"]: m for m in data["models"]}
    assert models["mistral:7b"]["done"] == 0


# --- /api/metrics ---

def test_metrics_returns_200(app):
    resp = app.get("/api/metrics")
    assert resp.status_code == 200


def test_metrics_by_model_keys(app):
    resp = app.get("/api/metrics")
    data = json.loads(resp.data)
    assert "by_model" in data
    row = next(r for r in data["by_model"] if r["model"] == "llama3.2:3b")
    assert "t2_legal_mean" in row
    assert "t1_direction_correct_mean" in row
    assert "t3_score_mean" in row


def test_metrics_nan_becomes_null(app):
    resp = app.get("/api/metrics")
    raw = resp.data.decode()
    assert "NaN" not in raw
    assert '"nan"' not in raw


def test_metrics_legal_rate(app):
    resp = app.get("/api/metrics")
    data = json.loads(resp.data)
    row = next(r for r in data["by_model"] if r["model"] == "llama3.2:3b")
    assert row["t2_legal_mean"] == pytest.approx(0.0, abs=0.01)
    row2 = next(r for r in data["by_model"] if r["model"] == "qwen2.5:7b")
    # 2 legal out of 12 total (1 easy + 1 hard)
    assert row2["t2_legal_mean"] == pytest.approx(2/12, abs=0.01)


def test_metrics_by_difficulty(app):
    resp = app.get("/api/metrics")
    data = json.loads(resp.data)
    assert "by_difficulty" in data
    row = next(
        r for r in data["by_difficulty"]
        if r["model"] == "qwen2.5:7b" and r["difficulty"] == "hard"
    )
    assert row["t2_legal"] == pytest.approx(1/8, abs=0.01)


def test_metrics_hallucination(app):
    resp = app.get("/api/metrics")
    data = json.loads(resp.data)
    assert "hallucination" in data
    row = next(
        r for r in data["hallucination"]
        if r["model"] == "qwen2.5:7b" and r["difficulty"] == "easy"
    )
    # 3 illegal out of 4 = 75%
    assert row["hallucination_rate"] == pytest.approx(0.75, abs=0.01)


# --- static index ---

def test_index_served(app):
    resp = app.get("/")
    assert resp.status_code == 200
    assert b"<!DOCTYPE html>" in resp.data or b"<html" in resp.data
