from unittest.mock import patch
import sqlite3
import pytest
from src.job_queue import JobQueue
from src.worker import Worker
from tests.test_worker import SAMPLE_JOB, SAMPLE_POSITION, LLM_LEGAL_RESPONSE

def test_queue_snapshot_leaves_original_unchanged(tmp_path):
    path = tmp_path / "queue.db"
    source = JobQueue(str(path))
    source.insert_job({**SAMPLE_JOB, "hash": "fixture"})
    source.close()
    snapshot = JobQueue.readonly_snapshot(str(path))
    claimed = snapshot.claim_job("dry")
    snapshot.complete_job(claimed["job_id"])
    snapshot.close()
    original = JobQueue(str(path))
    assert original.get_job(SAMPLE_JOB["job_id"])["status"] == "pending"
    original.close()

def test_missing_snapshot_does_not_create_database(tmp_path):
    path = tmp_path / "missing.db"
    with pytest.raises(sqlite3.OperationalError):
        JobQueue.readonly_snapshot(str(path))
    assert not path.exists()

def test_dry_worker_uses_snapshot_and_never_writer(tmp_path):
    config = {"paths": {"jobs_db": str(tmp_path / "jobs.db"), "results_file": str(tmp_path / "absent/results.jsonl")}, "correction_loop": {"enabled": True}}
    with patch("src.worker.JobQueue.readonly_snapshot") as snapshot, patch("src.worker.ResultWriter") as writer, patch("src.worker.DataLoader") as loader, patch("src.worker.OllamaClient") as client:
        snapshot.return_value.get_done_job_ids.return_value = []
        loader.return_value.get_by_id.return_value = SAMPLE_POSITION
        client.return_value.chat.return_value = dict(LLM_LEGAL_RESPONSE)
        worker = Worker("dry", config, dry_run=True)
        with patch.object(worker.correction_manager, "trigger_correction") as correction:
            worker.process_job(SAMPLE_JOB)
            correction.assert_not_called()
        snapshot.assert_called_once_with(config["paths"]["jobs_db"])
        writer.assert_not_called()
    assert not (tmp_path / "absent").exists()

def test_cli_dry_run_does_not_write_run_log(tmp_path, monkeypatch):
    import sys
    from scripts import run_workers
    config = {"paths": {"jobs_db": str(tmp_path / "queue.db"), "logs_dir": str(tmp_path / "logs")}}
    with patch.object(run_workers, "load_config", return_value=config), patch.object(run_workers.JobQueue, "readonly_snapshot") as snapshot, patch.object(run_workers, "OllamaClient"), patch.object(run_workers, "run_worker", return_value=1), patch.object(run_workers, "write_run_log") as write_log:
        snapshot.return_value.get_progress.return_value = {"pending": 1}
        monkeypatch.setattr(sys, "argv", ["run_workers.py", "--dry-run", "--max-jobs", "1"])
        run_workers.main()
        write_log.assert_not_called()
    assert not (tmp_path / "logs").exists()
