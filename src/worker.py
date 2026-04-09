"""Worker process for benchmark job execution."""

import logging
import os
import re
import time
from typing import Any

# Shared position cache: populated in the parent process before forking so all
# workers inherit it via COW without each independently loading gigabyte-scale
# JSON files.  Keyed by position_id (int).
_SHARED_POSITIONS: dict[int, dict] = {}

import chess

from .data_loader import DataLoader
from .evaluator import score_all, should_trigger_correction
from .feedback_loop import CorrectionLoopManager
from .job_queue import JobQueue
from .llm_client import (
    OllamaClient,
    build_prompt,
    build_move_prompt,
    build_eval_prompt,
    build_explanation_prompt,
    MOVE_SYSTEM_PROMPT,
    EVAL_SYSTEM_PROMPT,
    EXPLANATION_SYSTEM_PROMPT,
    parse_response,
    parse_eval_response,
    parse_explanation_response,
    extract_move_from_text,
)
from .result_writer import ResultWriter, build_result_record, get_completed_job_ids
from .utils import load_config

logger = logging.getLogger("chess_llm_bench")


class Worker:
    """Benchmark worker that processes jobs from the queue."""

    def __init__(
        self,
        worker_id: str,
        config: dict[str, Any],
        dry_run: bool = False,
        model: str | None = None,
    ):
        """Initialize worker.

        Args:
            worker_id: Unique identifier for this worker
            config: Configuration dictionary
            dry_run: If True, don't write results
            model: If set, only process jobs for this model
        """
        self.worker_id = worker_id
        self.config = config
        self.dry_run = dry_run
        self.model = model

        # Initialize components — resolve paths to absolute to survive multiprocessing cwd changes
        paths = config.get("paths", {})
        self.job_queue = JobQueue(os.path.abspath(paths.get("jobs_db", "jobs/jobs.db")))
        self.result_writer = ResultWriter(
            os.path.abspath(paths.get("results_file", "results/evaluations.jsonl"))
        )
        self.data_loader = DataLoader(os.path.abspath(paths.get("data_dir", "data")))

        # Ollama client
        ollama_config = config.get("ollama", {})
        self.llm_client = OllamaClient(
            base_url=ollama_config.get("base_url", "http://localhost:11434"),
            timeout=ollama_config.get("timeout", 180),
            max_retries=ollama_config.get("max_retries", 3),
        )

        # Stockfish not used — evals are pre-computed in the dataset
        self.engine = None

        # Correction loop manager
        self.correction_manager = CorrectionLoopManager(
            self.data_loader,
            self.job_queue,
            config,
        )

        # Evaluation settings
        eval_config = config.get("evaluation", {})
        eval_range = eval_config.get("centipawn_eval_range", {})
        self.eval_range = (
            eval_range.get("min", -2000),
            eval_range.get("max", 2000),
        )
        self.cpl_threshold = eval_config.get("cpl_threshold", 50)

        # Track completed jobs (for duplicate detection)
        self.completed_jobs = get_completed_job_ids(
            paths.get("results_file", "results/evaluations.jsonl")
        )

    def process_job(self, job: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single benchmark job.

        Args:
            job: Job dictionary from the queue

        Returns:
            Result record or None if failed
        """
        job_id = job["job_id"]

        # Check if already completed
        if job_id in self.completed_jobs:
            logger.debug(f"Job {job_id} already completed, skipping")
            self.job_queue.complete_job(job_id)
            return None

        # Enrich job with position data from the JSON dataset
        pos = (
            _SHARED_POSITIONS.get(job["position_id"])
            if _SHARED_POSITIONS
            else self.data_loader.get_by_id(job["position_id"])
        )
        if pos is None:
            error_msg = f"Position {job['position_id']} not found in dataset"
            logger.error(f"Job {job_id}: {error_msg}")
            self.job_queue.fail_job(job_id, error_msg)
            return None
        job = {
            **job,
            "fen": pos["fen"],
            "pgn_moves": pos.get("pgn_moves", ""),
            "difficulty": pos.get("difficulty"),
            "phase": pos.get("phase"),
            "source": pos.get("source"),
            "theme": pos.get("theme", ""),
            "stockfish_eval": pos.get("stockfish_eval", 0),
            "stockfish_best_move": pos.get("stockfish_best_move", ""),
        }

        logger.info(f"Processing job {job_id} ({job['model']})")

        # Build position dict for scoring
        position = {
            "id": job["position_id"],
            "fen": job["fen"],
            "stockfish_eval": job["stockfish_eval"],
            "stockfish_best_move": job["stockfish_best_move"],
            "theme": job["theme"],
        }

        prompt_format = job.get("prompt_format", "pgn+fen")
        fen = job["fen"]
        pgn_moves = job.get("pgn_moves")

        def _is_legal(fen: str, san: str | None) -> bool:
            if not san:
                return False
            try:
                board = chess.Board(fen)
                return board.parse_san(san) in board.legal_moves
            except Exception:
                return False

        if prompt_format == "eval_only":
            prompt = build_eval_prompt(fen, pgn_moves)
            llm_result = self.llm_client.chat(job["model"], prompt, system_prompt=EVAL_SYSTEM_PROMPT)
            if not llm_result["success"]:
                self.job_queue.fail_job(job_id, llm_result.get("error", "Unknown error"))
                return None
            parsed = parse_eval_response(llm_result["response"])

        elif prompt_format == "move_only":
            prompt = build_move_prompt(fen)
            llm_result = self.llm_client.chat(job["model"], prompt, system_prompt=MOVE_SYSTEM_PROMPT, temperature=0)
            if not llm_result["success"]:
                self.job_queue.fail_job(job_id, llm_result.get("error", "Unknown error"))
                return None
            move_text = extract_move_from_text(fen, llm_result["response"])
            parsed = {
                "eval": None,
                "move": move_text,
                "explanation": None,
                "side_claimed": None,
                "parse_errors": [] if move_text else ["No legal move found in response"],
            }

        elif prompt_format == "explanation_only":
            prompt = build_explanation_prompt(fen, pgn_moves)
            llm_result = self.llm_client.chat(job["model"], prompt, system_prompt=EXPLANATION_SYSTEM_PROMPT)
            if not llm_result["success"]:
                self.job_queue.fail_job(job_id, llm_result.get("error", "Unknown error"))
                return None
            parsed = parse_explanation_response(llm_result["response"])

        else:
            # Combined prompt (fen_only / pgn+fen / cot)
            prompt = build_prompt(fen=fen, pgn_moves=pgn_moves, prompt_format=prompt_format)
            llm_result = self.llm_client.chat(job["model"], prompt)
            if not llm_result["success"]:
                self.job_queue.fail_job(job_id, llm_result.get("error", "Unknown error"))
                return None
            parsed = parse_response(llm_result["response"])

            # If combined prompt returned illegal/missing move, retry with isolated call
            if not _is_legal(fen, parsed.get("move")):
                # First try to rescue a legal move from the original response
                rescued = extract_move_from_text(fen, llm_result["response"])
                if rescued:
                    parsed["move"] = rescued
                else:
                    move_prompt = build_move_prompt(fen)
                    move_result = self.llm_client.chat(job["model"], move_prompt, system_prompt=MOVE_SYSTEM_PROMPT, temperature=0)
                    if move_result["success"] and move_result["response"].strip():
                        extracted = extract_move_from_text(fen, move_result["response"])
                        if extracted:
                            parsed["move"] = extracted
                    llm_result["inference_ms"] += move_result.get("inference_ms", 0)

            if (
                parsed["eval"] is None
                and parsed["move"] is None
                and parsed["explanation"] is None
            ):
                error_msg = "All three fields missing from response"
                logger.error(f"Job {job_id}: {error_msg}")
                self.job_queue.fail_job(job_id, error_msg)
                return None

        # Score all tasks
        scores = score_all(
            parsed_response=parsed,
            position=position,
            engine=self.engine,
            eval_range=self.eval_range,
        )

        # Build result record
        result = build_result_record(
            job=job,
            parsed_response=parsed,
            scores=scores,
            inference_ms=llm_result["inference_ms"],
        )

        # Write result
        if not self.dry_run:
            self.result_writer.write_result(result)
            self.completed_jobs.add(job_id)

        # Mark job complete
        self.job_queue.complete_job(job_id)

        # Check for correction loop trigger
        correction_enabled = self.config.get("correction_loop", {}).get(
            "enabled", True
        )
        if (
            correction_enabled
            and job.get("job_type") == "standard"
            and should_trigger_correction(scores.get("t2_cpl"), self.cpl_threshold)
        ):
            self.correction_manager.trigger_correction(job, result)

        logger.info(
            f"Completed job {job_id}: "
            f"T1_err={scores.get('t1_absolute_error')}, "
            f"T2_legal={scores.get('t2_legal')}, "
            f"T3_score={scores.get('t3_score')}"
        )

        return result

    def run(self, max_jobs: int | None = None) -> int:
        """Run the worker loop.

        Args:
            max_jobs: Maximum number of jobs to process (None for unlimited)

        Returns:
            Number of jobs processed
        """
        # Check Ollama availability
        if not self.llm_client.is_available():
            logger.error("Ollama is not running. Aborting.")
            raise RuntimeError("Ollama not available")

        jobs_processed = 0

        while True:
            # Check job limit
            if max_jobs is not None and jobs_processed >= max_jobs:
                logger.info(f"Reached job limit ({max_jobs})")
                break

            # Claim next job
            job = self.job_queue.claim_job(self.worker_id, model=self.model)

            if job is None:
                logger.info("No more pending jobs")
                break

            # Process job
            try:
                result = self.process_job(job)
                if result is not None:
                    jobs_processed += 1
            except Exception as e:
                logger.exception(f"Error processing job {job['job_id']}: {e}")
                self.job_queue.fail_job(job["job_id"], str(e))

        logger.info(f"Worker {self.worker_id} processed {jobs_processed} jobs")
        return jobs_processed


def run_worker(
    worker_id: str,
    config_path: str = "config/config.yaml",
    max_jobs: int | None = None,
    dry_run: bool = False,
    model: str | None = None,
) -> int:
    """Run a single worker process.

    Args:
        worker_id: Unique worker identifier
        config_path: Path to configuration file
        max_jobs: Maximum jobs to process
        dry_run: If True, don't write results
        model: If set, only process jobs for this model

    Returns:
        Number of jobs processed
    """
    config = load_config(config_path)
    worker = Worker(worker_id, config, dry_run=dry_run, model=model)
    return worker.run(max_jobs=max_jobs)
