"""SQLite-based job queue for benchmark execution."""

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("chess_llm_bench")


class JobQueue:
    """SQLite-based job queue with atomic operations."""

    def __init__(self, db_path: str = "jobs/jobs.db"):
        """Initialize job queue.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """Return the persistent connection, creating it if needed."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._connect()
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL;")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                position_id INTEGER NOT NULL,
                model TEXT NOT NULL,
                prompt_format TEXT NOT NULL,
                trial INTEGER DEFAULT 1,
                status TEXT DEFAULT 'pending',
                worker_id TEXT,
                claimed_at TEXT,
                completed_at TEXT,
                paired_control_job_id TEXT,
                parent_job_id TEXT,
                hash TEXT UNIQUE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                error_message TEXT
            )
        """)

        # Create indexes for common queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_status ON jobs(status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_model ON jobs(model)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_position_id ON jobs(position_id)"
        )

        conn.commit()

    def insert_job(self, job: dict[str, Any]) -> bool:
        """Insert a new job into the queue.

        Args:
            job: Job dictionary with required fields

        Returns:
            True if inserted, False if duplicate (hash collision)
        """
        try:
            conn = self._connect()
            conn.execute("""
                INSERT INTO jobs (
                    job_id, job_type, position_id,
                    model, prompt_format, trial, status,
                    paired_control_job_id, parent_job_id, hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job["job_id"],
                job.get("job_type", "standard"),
                job["position_id"],
                job["model"],
                job.get("prompt_format", "pgn+fen"),
                job.get("trial", 1),
                "pending",
                job.get("paired_control_job_id"),
                job.get("parent_job_id"),
                job.get("hash"),
            ))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Duplicate hash
            return False

    def insert_jobs(self, jobs: list[dict[str, Any]]) -> int:
        """Insert multiple jobs in a single connection, skipping duplicates.

        Args:
            jobs: List of job dictionaries

        Returns:
            Number of jobs inserted
        """
        inserted = 0
        conn = self._connect()
        for job in jobs:
            try:
                conn.execute("""
                    INSERT INTO jobs (
                        job_id, job_type, position_id,
                        model, prompt_format, trial, status,
                        paired_control_job_id, parent_job_id, hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    job["job_id"],
                    job.get("job_type", "standard"),
                    job["position_id"],
                    job["model"],
                    job.get("prompt_format", "pgn+fen"),
                    job.get("trial", 1),
                    "pending",
                    job.get("paired_control_job_id"),
                    job.get("parent_job_id"),
                    job.get("hash"),
                ))
                inserted += 1
            except sqlite3.IntegrityError:
                pass
        conn.commit()
        return inserted

    def claim_job(self, worker_id: str, model: str | None = None) -> dict[str, Any] | None:
        """Atomically claim the next pending job.

        Args:
            worker_id: Unique identifier for the claiming worker
            model: If set, only claim jobs for this model

        Returns:
            Job dictionary or None if no pending jobs
        """
        conn = self._connect()
        # Atomic claim using UPDATE...RETURNING (SQLite 3.35+)
        if model:
            cursor = conn.execute("""
                UPDATE jobs
                SET status = 'in_progress',
                    worker_id = ?,
                    claimed_at = datetime('now')
                WHERE job_id = (
                    SELECT job_id FROM jobs
                    WHERE status = 'pending' AND model = ?
                    ORDER BY ROWID
                    LIMIT 1
                )
                RETURNING *
            """, (worker_id, model))
        else:
            cursor = conn.execute("""
                UPDATE jobs
                SET status = 'in_progress',
                    worker_id = ?,
                    claimed_at = datetime('now')
                WHERE job_id = (
                    SELECT job_id FROM jobs
                    WHERE status = 'pending'
                    ORDER BY ROWID
                    LIMIT 1
                )
                RETURNING *
            """, (worker_id,))

        row = cursor.fetchone()
        conn.commit()

        if row:
            return dict(row)
        return None

    def complete_job(self, job_id: str) -> None:
        """Mark a job as completed.

        Args:
            job_id: Job identifier
        """
        conn = self._connect()
        conn.execute("""
            UPDATE jobs
            SET status = 'done',
                completed_at = ?
            WHERE job_id = ?
        """, (datetime.utcnow().isoformat(), job_id))
        conn.commit()

    def fail_job(self, job_id: str, error_message: str = "") -> None:
        """Mark a job as failed.

        Args:
            job_id: Job identifier
            error_message: Error description
        """
        conn = self._connect()
        conn.execute("""
            UPDATE jobs
            SET status = 'failed',
                completed_at = ?,
                error_message = ?
            WHERE job_id = ?
        """, (datetime.utcnow().isoformat(), error_message, job_id))
        conn.commit()

    def reset_job(self, job_id: str) -> None:
        """Reset a job to pending status.

        Args:
            job_id: Job identifier
        """
        conn = self._connect()
        conn.execute("""
            UPDATE jobs
            SET status = 'pending',
                worker_id = NULL,
                claimed_at = NULL,
                completed_at = NULL,
                error_message = NULL
            WHERE job_id = ?
        """, (job_id,))
        conn.commit()

    def reset_stale_jobs(self, timeout_minutes: int = 30) -> int:
        """Reset jobs that have been in_progress for too long.

        Args:
            timeout_minutes: Time after which to consider a job stale

        Returns:
            Number of jobs reset
        """
        conn = self._connect()
        cursor = conn.execute("""
            UPDATE jobs
            SET status = 'pending',
                worker_id = NULL,
                claimed_at = NULL
            WHERE status = 'in_progress'
            AND claimed_at <= datetime('now', ?)
        """, (f"-{timeout_minutes} minutes",))
        conn.commit()
        return cursor.rowcount

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Get a job by ID.

        Args:
            job_id: Job identifier

        Returns:
            Job dictionary or None
        """
        conn = self._connect()
        cursor = conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?",
            (job_id,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    def get_jobs_by_status(self, status: str) -> list[dict[str, Any]]:
        """Get all jobs with a specific status.

        Args:
            status: Job status (pending, in_progress, done, failed)

        Returns:
            List of job dictionaries
        """
        conn = self._connect()
        cursor = conn.execute(
            "SELECT * FROM jobs WHERE status = ?",
            (status,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def count_by_status(self) -> dict[str, int]:
        """Get count of jobs by status.

        Returns:
            Dictionary mapping status to count
        """
        conn = self._connect()
        cursor = conn.execute("""
            SELECT status, COUNT(*) as count
            FROM jobs
            GROUP BY status
        """)
        return {row["status"]: row["count"] for row in cursor.fetchall()}

    def count_total(self) -> int:
        """Get total number of jobs.

        Returns:
            Total job count
        """
        conn = self._connect()
        cursor = conn.execute("SELECT COUNT(*) as count FROM jobs")
        return cursor.fetchone()["count"]

    def has_hash(self, hash_value: str) -> bool:
        """Check if a job with the given hash exists.

        Args:
            hash_value: SHA256 hash

        Returns:
            True if exists
        """
        conn = self._connect()
        cursor = conn.execute(
            "SELECT 1 FROM jobs WHERE hash = ?",
            (hash_value,)
        )
        return cursor.fetchone() is not None

    def clear_all(self) -> None:
        """Delete all jobs from the queue."""
        conn = self._connect()
        conn.execute("DELETE FROM jobs")
        conn.commit()
        logger.warning("All jobs cleared from queue")

    def get_progress(self) -> dict[str, Any]:
        """Get detailed progress information.

        Returns:
            Progress statistics
        """
        counts = self.count_by_status()
        total = self.count_total()

        done = counts.get("done", 0)
        failed = counts.get("failed", 0)
        in_progress = counts.get("in_progress", 0)
        pending = counts.get("pending", 0)

        return {
            "total": total,
            "done": done,
            "failed": failed,
            "in_progress": in_progress,
            "pending": pending,
            "completed": done + failed,
            "percent_complete": (done + failed) / total * 100 if total > 0 else 0,
        }
