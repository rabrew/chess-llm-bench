"""Tests for the dashboard Flask server."""
import json
import math
import os
import sys
import tempfile
import sqlite3

import pytest

# Add dashboard dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dashboard"))


@pytest.fixture
def sample_db(tmp_path):
    """Create a minimal jobs.db for testing."""
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
def sample_metrics_dir(tmp_path):
    """Create minimal CSV metric files for testing."""
    by_model = tmp_path / "by_model.csv"
    by_model.write_text(
        "model,t1_absolute_error_mean,t1_direction_correct_mean,"
        "t2_legal_mean,t2_cpl_mean,t3_score_mean,job_id_count\n"
        "llama3.2:3b,2568.0,0.28,0.11,431.0,0.58,19\n"
        "qwen2.5:7b,1307.0,0.32,0.11,2423.0,0.43,336\n"
    )

    by_diff = tmp_path / "by_difficulty.csv"
    by_diff.write_text(
        "model,difficulty,t1_absolute_error,t2_cpl,t2_legal,t3_score,job_id\n"
        "llama3.2:3b,easy,363.5,,0.0,1.0,2\n"
        "qwen2.5:7b,easy,1441.0,452.5,0.04,0.38,52\n"
        "qwen2.5:7b,hard,714.8,6476.6,0.13,0.41,64\n"
    )

    halluc = tmp_path / "hallucination_rate.csv"
    halluc.write_text(
        "model,difficulty,hallucination_rate,job_id\n"
        "llama3.2:3b,easy,1.0,2\n"
        "qwen2.5:7b,easy,0.96,52\n"
        "qwen2.5:7b,hard,0.88,64\n"
    )

    return str(tmp_path)


@pytest.fixture
def app(sample_db, sample_metrics_dir):
    import server
    server.DB_PATH = sample_db
    server.METRICS_DIR = sample_metrics_dir
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
    """Models with only pending jobs should appear with 0 done."""
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
    """NaN values in CSVs must be serialised as null, not the string 'nan'."""
    resp = app.get("/api/metrics")
    raw = resp.data.decode()
    assert "NaN" not in raw
    assert '"nan"' not in raw


def test_metrics_by_difficulty(app):
    resp = app.get("/api/metrics")
    data = json.loads(resp.data)
    assert "by_difficulty" in data
    row = next(
        r for r in data["by_difficulty"]
        if r["model"] == "qwen2.5:7b" and r["difficulty"] == "hard"
    )
    assert row["t2_legal"] == pytest.approx(0.13, abs=0.01)


def test_metrics_hallucination(app):
    resp = app.get("/api/metrics")
    data = json.loads(resp.data)
    assert "hallucination" in data
    row = next(
        r for r in data["hallucination"]
        if r["model"] == "qwen2.5:7b" and r["difficulty"] == "easy"
    )
    assert row["hallucination_rate"] == pytest.approx(0.96, abs=0.01)


# --- static index ---

def test_index_served(app):
    resp = app.get("/")
    assert resp.status_code == 200
    assert b"<!DOCTYPE html>" in resp.data or b"<html" in resp.data
