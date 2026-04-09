"""File-locked JSONL result writer for concurrent-safe output."""

import json
import logging
import re
from pathlib import Path
from typing import Any

_JOB_ID_RE = re.compile(r'"job_id"\s*:\s*"([^"]+)"')

from filelock import FileLock

from .utils import get_timestamp, parse_model_info

logger = logging.getLogger("chess_llm_bench")


class ResultWriter:
    """Thread-safe JSONL result writer using file locks."""

    def __init__(self, results_file: str = "results/evaluations.jsonl"):
        """Initialize result writer.

        Args:
            results_file: Path to the JSONL results file
        """
        self.results_file = Path(results_file)
        self.lock_file = Path(f"{results_file}.lock")

        # Ensure directory exists
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(self.lock_file)

    def write_result(self, result: dict[str, Any]) -> None:
        """Write a single result record to the JSONL file.

        Args:
            result: Result dictionary to write
        """
        with self.lock:
            with open(self.results_file, "a") as f:
                f.write(json.dumps(result) + "\n")

    def write_results(self, results: list[dict[str, Any]]) -> None:
        """Write multiple result records to the JSONL file.

        Args:
            results: List of result dictionaries
        """
        with self.lock:
            with open(self.results_file, "a") as f:
                for result in results:
                    f.write(json.dumps(result) + "\n")


def build_result_record(
    job: dict[str, Any],
    parsed_response: dict[str, Any],
    scores: dict[str, Any],
    inference_ms: int,
) -> dict[str, Any]:
    """Build a complete result record for storage.

    Args:
        job: Job dictionary
        parsed_response: Parsed LLM response
        scores: T1/T2/T3 scoring results
        inference_ms: Inference time in milliseconds

    Returns:
        Complete result record
    """
    # Parse model info
    model_info = parse_model_info(job["model"])

    record = {
        # Job identification
        "job_id": job["job_id"],
        "job_type": job.get("job_type", "standard"),
        "attempt": job.get("trial", 1),
        "parent_job_id": job.get("parent_job_id"),

        # Position info
        "position_id": job["position_id"],
        "fen": job.get("fen"),

        # Model info
        "model": job["model"],
        "model_family": model_info["family"],
        "model_size_b": model_info["size_b"],
        "prompt_format": job.get("prompt_format", "pgn+fen"),

        # Position metadata
        "difficulty": job.get("difficulty"),
        "phase": job.get("phase"),
        "source": job.get("source"),
        "theme": job.get("theme"),

        # Timing
        "inference_ms": inference_ms,
        "timestamp": get_timestamp(),
    }

    # Add T1 scores
    record["t1_model_eval"] = scores.get("t1_model_eval")
    record["t1_stockfish_eval"] = scores.get("t1_stockfish_eval")
    record["t1_absolute_error"] = scores.get("t1_absolute_error")
    record["t1_direction_correct"] = scores.get("t1_direction_correct")

    # Add T2 scores
    record["t2_move"] = scores.get("t2_move")
    record["t2_best_move"] = scores.get("t2_best_move")
    record["t2_legal"] = scores.get("t2_legal")
    record["t2_cpl"] = scores.get("t2_cpl")

    # Add T3 scores
    record["t3_explanation"] = scores.get("t3_explanation")
    record["t3_side_claimed"] = scores.get("t3_side_claimed")
    record["t3_p1_side_correct"] = scores.get("t3_p1_side_correct")
    record["t3_p2_theme_correct"] = scores.get("t3_p2_theme_correct")
    record["t3_score"] = scores.get("t3_score")

    # Add parse errors if any
    if parsed_response.get("parse_errors"):
        record["parse_errors"] = parsed_response["parse_errors"]

    return record


def load_results(results_file: str = "results/evaluations.jsonl") -> list[dict[str, Any]]:
    """Load all results from the JSONL file.

    Args:
        results_file: Path to the JSONL results file

    Returns:
        List of result dictionaries
    """
    results_path = Path(results_file)
    if not results_path.exists():
        return []

    results = []
    with open(results_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse result line: {line[:100]}...")

    return results


def get_completed_job_ids(
    results_file: str = "results/evaluations.jsonl",
) -> set[str]:
    """Get set of job IDs that have already been completed.

    Uses regex extraction instead of full JSON parsing to avoid loading
    all result records into memory (saves ~300MB per worker process).

    Args:
        results_file: Path to the JSONL results file

    Returns:
        Set of completed job IDs
    """
    job_ids: set[str] = set()
    results_path = Path(results_file)
    if not results_path.exists():
        return job_ids
    with open(results_path, "r") as f:
        for line in f:
            m = _JOB_ID_RE.search(line)
            if m:
                job_ids.add(m.group(1))
    return job_ids
