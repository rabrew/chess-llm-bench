"""run_all.sh tier tracking and output paths (specs/run-all-tier-tracking.md)."""

import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import pending_jobs  # noqa: E402
import run_workers  # noqa: E402
from src.job_queue import JobQueue  # noqa: E402

RUN_ALL = (REPO / "bin" / "run_all.sh").read_text()
MODEL = "llama3.2:3b-board-ft"


def _job(job_id, model=MODEL):
    return {"job_id": job_id, "job_type": "standard", "position_id": 1, "model": model,
            "prompt_format": "fen_only", "trial": 1}


@pytest.fixture
def jobs_db(tmp_path):
    path = str(tmp_path / "jobs.db")
    queue = JobQueue(path)
    for i in range(5):
        queue.insert_job(_job(f"j{i}"))
    queue.insert_job(_job("other", model="gemma3:4b-board-ft"))
    queue.claim_job("w1", model=MODEL)       # j0 -> in_progress
    queue.complete_job("j1")
    queue.fail_job("j2", "boom")
    queue.close()
    return path


# ---------- count_pending ----------

def test_counts_pending_and_in_progress_for_model_only(jobs_db):
    # j0 in_progress, j3 + j4 pending; j1 done, j2 failed, "other" is another model
    assert pending_jobs.count_pending(jobs_db, MODEL) == 3


def test_missing_db_raises_and_creates_nothing(tmp_path):
    missing = tmp_path / "nope.db"
    with pytest.raises(FileNotFoundError):
        pending_jobs.count_pending(str(missing), MODEL)
    assert not missing.exists()


def test_non_database_file_raises(tmp_path):
    junk = tmp_path / "junk.db"
    junk.write_text("not a database")
    with pytest.raises(sqlite3.Error):
        pending_jobs.count_pending(str(junk), MODEL)


def _cli(*args):
    return subprocess.run([sys.executable, str(REPO / "scripts" / "pending_jobs.py"), *args],
                          capture_output=True, text=True)


def test_cli_prints_count(jobs_db):
    result = _cli("--db", jobs_db, "--model", MODEL)
    assert result.returncode == 0
    assert result.stdout.strip() == "3"


def test_cli_exits_1_on_missing_db(tmp_path):
    result = _cli("--db", str(tmp_path / "nope.db"), "--model", MODEL)
    assert result.returncode == 1
    assert "nope.db" in result.stderr


# ---------- run_all.sh ----------

def _worker_loop() -> str:
    start = RUN_ALL.index('until tier_done "$model" "$tier"; do')
    return RUN_ALL[start:RUN_ALL.index("done", RUN_ALL.index("SKIPPED_TIERS", start))]


def test_tier_done_checked_before_pending_and_skip_marking():
    loop = _worker_loop()
    workers = loop.index("run_workers.py")
    completion_check = loop.index('if tier_done "$model" "$tier"; then', workers)
    assert completion_check < loop.index("pending_jobs.py", workers)
    assert completion_check < loop.index('>> "$SKIPPED_TIERS"', workers)


def test_unreadable_db_stops_the_run():
    loop = _worker_loop()
    assert re.search(r"if ! PENDING=\$\(python3? scripts/pending_jobs\.py", loop)
    assert "exit 1" in loop
    assert "except Exception" not in RUN_ALL
    assert "|| echo 0)" not in loop


def test_summary_paths_come_from_config():
    assert "results/plots/" not in RUN_ALL
    assert "results/metrics/" not in RUN_ALL
    assert "cfg_path plots_dir" in RUN_ALL and "cfg_path metrics_dir" in RUN_ALL


# ---------- run_workers.py ----------

def test_run_workers_has_no_utcnow():
    assert "utcnow" not in (REPO / "scripts" / "run_workers.py").read_text()


def test_run_log_timestamp_keeps_z_format(tmp_path):
    run_workers.write_run_log({}, "20261023_000000", str(tmp_path))
    stamp = json.loads((tmp_path / "run_20261023_000000.json").read_text())["timestamp"]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z", stamp)


# ---------- tier target from the config (added 2026-10-31) ----------

import tier_target  # noqa: E402

SIX_FORMATS = ["fen_only", "pgn+fen", "cot", "eval_only", "move_only", "explanation_only"]


def test_tier_target_full_benchmark_is_unchanged():
    assert tier_target.results_needed_per_tier({"max_positions_per_tier": 1000, "prompt_formats": SIX_FORMATS}) == 5900


def test_tier_target_uses_sample_per_tier_first():
    bench = {"max_positions_per_tier": 1000, "sample_per_tier": 125,
             "prompt_formats": ["fen_only", "board_text", "board_text+legal"]}
    assert tier_target.results_needed_per_tier(bench) == 368


@pytest.mark.parametrize("bench", [
    {"max_positions_per_tier": 1000},
    {"max_positions_per_tier": 1000, "prompt_formats": []},
    {"prompt_formats": SIX_FORMATS},
])
def test_tier_target_bad_config_raises(bench):
    with pytest.raises(ValueError):
        tier_target.results_needed_per_tier(bench)


def test_tier_target_cli_exits_1_on_bad_config(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text("benchmark:\n  max_positions_per_tier: 10\n")
    result = subprocess.run([sys.executable, str(REPO / "scripts" / "tier_target.py"), "--config", str(config)],
                            capture_output=True, text=True)
    assert result.returncode == 1 and "prompt_formats" in result.stderr


def test_run_all_takes_tier_target_and_skip_list_from_the_config():
    assert not re.search(r"JOBS_PER_TIER=\d", RUN_ALL)
    assert "scripts/tier_target.py" in RUN_ALL
    assert "jobs/db/skipped_tiers.txt" not in RUN_ALL
    assert 'SKIPPED_TIERS="${JOBS_DB%.db}_skipped_tiers.txt"' in RUN_ALL
