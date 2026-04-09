"""Tests for SQLite job queue."""

import os
import tempfile

import pytest

from src.job_queue import JobQueue


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def job_queue(temp_db):
    """Create a job queue with temporary database."""
    return JobQueue(temp_db)


class TestJobQueue:
    def test_insert_job(self, job_queue, sample_job):
        result = job_queue.insert_job(sample_job)
        assert result is True
        assert job_queue.count_total() == 1

    def test_duplicate_hash_rejected(self, job_queue, sample_job):
        job_queue.insert_job(sample_job)
        result = job_queue.insert_job(sample_job)  # Same hash
        assert result is False
        assert job_queue.count_total() == 1

    def test_claim_job(self, job_queue, sample_job):
        job_queue.insert_job(sample_job)
        claimed = job_queue.claim_job("worker_1")
        assert claimed is not None
        assert claimed["job_id"] == sample_job["job_id"]
        assert claimed["status"] == "in_progress"

    def test_claim_empty_queue(self, job_queue):
        claimed = job_queue.claim_job("worker_1")
        assert claimed is None

    def test_complete_job(self, job_queue, sample_job):
        job_queue.insert_job(sample_job)
        job_queue.claim_job("worker_1")
        job_queue.complete_job(sample_job["job_id"])

        counts = job_queue.count_by_status()
        assert counts.get("done", 0) == 1
        assert counts.get("in_progress", 0) == 0

    def test_fail_job(self, job_queue, sample_job):
        job_queue.insert_job(sample_job)
        job_queue.claim_job("worker_1")
        job_queue.fail_job(sample_job["job_id"], "Test error")

        job = job_queue.get_job(sample_job["job_id"])
        assert job["status"] == "failed"
        assert job["error_message"] == "Test error"

    def test_reset_job(self, job_queue, sample_job):
        job_queue.insert_job(sample_job)
        job_queue.claim_job("worker_1")
        job_queue.reset_job(sample_job["job_id"])

        job = job_queue.get_job(sample_job["job_id"])
        assert job["status"] == "pending"
        assert job["worker_id"] is None

    def test_progress(self, job_queue, sample_job):
        job_queue.insert_job(sample_job)
        progress = job_queue.get_progress()
        assert progress["total"] == 1
        assert progress["pending"] == 1
        assert progress["percent_complete"] == 0

    def test_insert_jobs_batch(self, job_queue, sample_job):
        job2 = {**sample_job, "job_id": "job_2", "hash": "hash_2"}
        inserted = job_queue.insert_jobs([sample_job, job2])
        assert inserted == 2
        assert job_queue.count_total() == 2

    def test_insert_jobs_skips_duplicates(self, job_queue, sample_job):
        inserted = job_queue.insert_jobs([sample_job, sample_job])
        assert inserted == 1

    def test_claim_job_with_model_filter(self, job_queue, sample_job):
        job_other = {**sample_job, "job_id": "job_other", "model": "other:7b", "hash": "hash_other"}
        job_queue.insert_job(sample_job)
        job_queue.insert_job(job_other)
        # Claim only for the specific model
        claimed = job_queue.claim_job("worker_1", model=sample_job["model"])
        assert claimed is not None
        assert claimed["model"] == sample_job["model"]

    def test_claim_job_model_filter_no_match(self, job_queue, sample_job):
        job_queue.insert_job(sample_job)
        claimed = job_queue.claim_job("worker_1", model="nonexistent:7b")
        assert claimed is None

    def test_reset_stale_jobs(self, job_queue, sample_job):
        job_queue.insert_job(sample_job)
        job_queue.claim_job("worker_1")
        # Reset with 0-minute timeout to catch all in_progress jobs
        reset = job_queue.reset_stale_jobs(timeout_minutes=0)
        assert reset >= 1
        job = job_queue.get_job(sample_job["job_id"])
        assert job["status"] == "pending"

    def test_get_job_exists(self, job_queue, sample_job):
        job_queue.insert_job(sample_job)
        job = job_queue.get_job(sample_job["job_id"])
        assert job is not None
        assert job["job_id"] == sample_job["job_id"]

    def test_get_job_not_found(self, job_queue):
        job = job_queue.get_job("nonexistent_id")
        assert job is None

    def test_get_jobs_by_status(self, job_queue, sample_job):
        job_queue.insert_job(sample_job)
        pending = job_queue.get_jobs_by_status("pending")
        assert len(pending) == 1
        assert pending[0]["job_id"] == sample_job["job_id"]

    def test_get_jobs_by_status_empty(self, job_queue):
        done = job_queue.get_jobs_by_status("done")
        assert done == []

    def test_has_hash_true(self, job_queue, sample_job):
        job_queue.insert_job(sample_job)
        assert job_queue.has_hash(sample_job["hash"]) is True

    def test_has_hash_false(self, job_queue):
        assert job_queue.has_hash("nonexistent_hash") is False

    def test_clear_all(self, job_queue, sample_job):
        job_queue.insert_job(sample_job)
        job_queue.clear_all()
        assert job_queue.count_total() == 0

    def test_get_progress_empty(self, job_queue):
        progress = job_queue.get_progress()
        assert progress["total"] == 0
        assert progress["percent_complete"] == 0

    def test_get_progress_with_done(self, job_queue, sample_job):
        job_queue.insert_job(sample_job)
        job_queue.claim_job("w1")
        job_queue.complete_job(sample_job["job_id"])
        progress = job_queue.get_progress()
        assert progress["done"] == 1
        assert progress["percent_complete"] == 100.0
