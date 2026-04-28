"""Correction loop experiment for secondary analysis."""

import hashlib
import logging
from typing import Any

from .data_loader import DataLoader
from .job_generator import generate_correction_jobs
from .job_queue import JobQueue
from .llm_client import build_prompt

logger = logging.getLogger("chess_llm_bench")


def build_correction_prompt(
    original_fen: str,
    model_move: str,
    best_move: str,
    follow_up_position: dict[str, Any],
    prompt_format: str = "pgn+fen",
) -> str:
    """Build the correction prompt for a follow-up position.

    This prompt includes feedback about the incorrect move and then
    presents a new position of the same type.

    Args:
        original_fen: FEN of the original position
        model_move: The model's (incorrect) move
        best_move: The correct best move
        follow_up_position: The new position to evaluate
        prompt_format: Prompt format for the follow-up

    Returns:
        Correction prompt string
    """
    feedback = f"""Your move was not the best.

Position: {original_fen}
Your move: {model_move}
Best move: {best_move}

Explain why {best_move} is stronger. Then answer the same three questions
for the following new position of the same type.

---

"""

    # Standard prompt for the follow-up position
    standard_prompt = build_prompt(
        fen=follow_up_position["fen"],
        pgn_moves=follow_up_position.get("pgn_moves"),
        prompt_format=prompt_format,
    )

    return feedback + standard_prompt


def select_follow_up_position(
    original_position: dict[str, Any],
    data_loader: DataLoader,
    exclude_ids: set[int],
    seed: int,
) -> dict[str, Any] | None:
    """Select a follow-up position for correction/control testing.

    Selection criteria (in order of preference):
    1. Same theme and difficulty
    2. Same difficulty only
    3. Any available position

    Args:
        original_position: The original position that triggered correction
        data_loader: DataLoader instance
        exclude_ids: Position IDs to exclude (already used)
        seed: Random seed for reproducibility

    Returns:
        Follow-up position or None if none available
    """
    # Add original position ID to exclude set
    exclude_ids = exclude_ids.copy()
    exclude_ids.add(original_position.get("id", -1))

    # Try to find position with same theme and difficulty
    follow_up = data_loader.get_similar(
        original_position,
        exclude_ids=exclude_ids,
        seed=seed,
    )

    return follow_up


def calculate_learning_delta(
    cpl_attempt_1: int | None,
    cpl_attempt_2: int | None,
) -> int | None:
    """Calculate the learning delta between two attempts.

    Learning Delta = CPL(attempt 1) - CPL(attempt 2)
    Positive value means improvement.

    Args:
        cpl_attempt_1: CPL from first attempt
        cpl_attempt_2: CPL from second attempt

    Returns:
        Learning delta or None if calculation not possible
    """
    if cpl_attempt_1 is None or cpl_attempt_2 is None:
        return None
    return cpl_attempt_1 - cpl_attempt_2


def calculate_net_feedback_effect(
    correction_delta: int | None,
    control_delta: int | None,
) -> int | None:
    """Calculate the net effect of feedback.

    Net Effect = Learning Delta (correction) - Learning Delta (control)
    Positive value means feedback helped.

    Args:
        correction_delta: Learning delta from correction condition
        control_delta: Learning delta from control condition

    Returns:
        Net feedback effect or None if calculation not possible
    """
    if correction_delta is None or control_delta is None:
        return None
    return correction_delta - control_delta


class CorrectionLoopManager:
    """Manage correction loop experiment execution."""

    def __init__(
        self,
        data_loader: DataLoader,
        job_queue: JobQueue,
        config: dict[str, Any],
    ):
        """Initialize correction loop manager.

        Args:
            data_loader: DataLoader instance
            job_queue: JobQueue instance
            config: Configuration dictionary
        """
        self.data_loader = data_loader
        self.job_queue = job_queue
        self.config = config
        self.used_positions: set[int] = set()

    def trigger_correction(
        self,
        original_job: dict[str, Any],
        original_result: dict[str, Any],
    ) -> tuple[str, str] | None:
        """Trigger a correction loop for a failed job.

        Args:
            original_job: The job that triggered correction
            original_result: The result with high CPL

        Returns:
            Tuple of (correction_job_id, control_job_id) or None if failed
        """
        # Check if correction loop is enabled
        if not self.config.get("correction_loop", {}).get("enabled", True):
            return None

        # Select follow-up position
        original_position = {
            "id": original_job["position_id"],
            "fen": original_job["fen"],
            "theme": original_job.get("theme"),
            "difficulty": original_job.get("difficulty"),
        }

        seed = self.config.get("benchmark", {}).get("random_seed", 42)
        # hash() is PYTHONHASHSEED-randomized across process restarts; use a
        # deterministic digest so follow-up positions are reproducible.
        job_id_hash = int(hashlib.sha256(original_job["job_id"].encode()).hexdigest()[:8], 16)
        job_seed = seed + job_id_hash

        follow_up = select_follow_up_position(
            original_position,
            self.data_loader,
            self.used_positions,
            job_seed,
        )

        if follow_up is None:
            logger.warning(
                f"No follow-up position available for job {original_job['job_id']}"
            )
            return None

        self.used_positions.add(follow_up["id"])

        # Generate correction and control jobs
        correction_job, control_job = generate_correction_jobs(
            position=original_position,
            model=original_job["model"],
            prompt_format=original_job["prompt_format"],
            parent_job_id=original_job["job_id"],
            follow_up_position=follow_up,
        )

        # Insert jobs
        inserted_correction = self.job_queue.insert_job(correction_job)
        inserted_control = self.job_queue.insert_job(control_job)

        if inserted_correction and inserted_control:
            logger.info(
                f"Triggered correction loop: {correction_job['job_id']}, "
                f"{control_job['job_id']}"
            )
            return correction_job["job_id"], control_job["job_id"]

        return None

    def get_correction_prompt(
        self,
        correction_job: dict[str, Any],
        parent_result: dict[str, Any],
    ) -> str:
        """Build the correction prompt for a correction job.

        Args:
            correction_job: The correction job
            parent_result: Result from the parent job

        Returns:
            Correction prompt string
        """
        follow_up_position = {
            "fen": correction_job["fen"],
            "pgn_moves": correction_job.get("pgn_moves", ""),
        }

        return build_correction_prompt(
            original_fen=parent_result.get("fen", ""),
            model_move=parent_result.get("t2_move", ""),
            best_move=parent_result.get("t2_best_move", ""),
            follow_up_position=follow_up_position,
            prompt_format=correction_job["prompt_format"],
        )
