"""Generate benchmark jobs from dataset positions."""

import logging
from typing import Any

from .data_loader import DataLoader
from .job_queue import JobQueue
from .utils import compute_hash

logger = logging.getLogger("chess_llm_bench")


def generate_job_id(position_id: int, model: str, prompt_format: str, trial: int = 1) -> str:
    """Generate a unique job ID.

    Args:
        position_id: Position identifier
        model: Model tag
        prompt_format: Prompt format
        trial: Trial number

    Returns:
        Formatted job ID
    """
    model_short = model.replace(":", "_").replace(".", "_")
    return f"job_{position_id:05d}_{model_short}_{prompt_format}_{trial}"


def generate_standard_jobs(
    positions: list[dict[str, Any]],
    models: list[str],
    prompt_formats: list[str],
) -> list[dict[str, Any]]:
    """Generate standard benchmark jobs for all combinations.

    Args:
        positions: List of position dictionaries
        models: List of model tags
        prompt_formats: List of prompt format strings

    Returns:
        List of job dictionaries
    """
    jobs = []

    for pos in positions:
        for model in models:
            for prompt_format in prompt_formats:
                job_id = generate_job_id(
                    pos["id"], model, prompt_format
                )

                # Compute hash for duplicate detection
                job_hash = compute_hash(
                    pos["fen"],
                    model,
                    prompt_format,
                    "standard",
                    "1",  # trial
                )

                job = {
                    "job_id": job_id,
                    "job_type": "standard",
                    "position_id": pos["id"],
                    "fen": pos["fen"],
                    "pgn_moves": pos.get("pgn_moves", ""),
                    "model": model,
                    "prompt_format": prompt_format,
                    "difficulty": pos.get("difficulty"),
                    "phase": pos.get("phase"),
                    "source": pos.get("source"),
                    "theme": pos.get("theme"),
                    "trial": 1,
                    "hash": job_hash,
                }
                jobs.append(job)

    logger.info(f"Generated {len(jobs)} standard jobs")
    return jobs


def generate_correction_jobs(
    position: dict[str, Any],
    model: str,
    prompt_format: str,
    parent_job_id: str,
    follow_up_position: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate a correction job and its paired control job.

    Args:
        position: Original position that triggered correction
        model: Model tag
        prompt_format: Prompt format
        parent_job_id: ID of the job that triggered correction
        follow_up_position: Position for the correction/control test

    Returns:
        Tuple of (correction_job, control_job)
    """
    # Correction job
    correction_job_id = f"{parent_job_id}_correction"
    correction_hash = compute_hash(
        follow_up_position["fen"],
        model,
        prompt_format,
        "correction",
        "2",  # trial 2
    )

    correction_job = {
        "job_id": correction_job_id,
        "job_type": "correction",
        "position_id": follow_up_position["id"],
        "model": model,
        "prompt_format": prompt_format,
        "trial": 2,
        "parent_job_id": parent_job_id,
        "hash": correction_hash,
    }

    # Control job (same position, no feedback)
    control_job_id = f"{parent_job_id}_control"
    control_hash = compute_hash(
        follow_up_position["fen"],
        model,
        prompt_format,
        "control",
        "2",
    )

    control_job = {
        "job_id": control_job_id,
        "job_type": "control",
        "position_id": follow_up_position["id"],
        "model": model,
        "prompt_format": prompt_format,
        "trial": 2,
        "parent_job_id": parent_job_id,
        "paired_control_job_id": correction_job_id,
        "hash": control_hash,
    }

    # Link them
    correction_job["paired_control_job_id"] = control_job_id

    return correction_job, control_job


def populate_job_queue(
    config: dict[str, Any],
    data_loader: DataLoader | None = None,
    job_queue: JobQueue | None = None,
    tier: str | None = None,
    model: str | None = None,
) -> int:
    """Populate the job queue with all benchmark jobs.

    Args:
        config: Configuration dictionary
        data_loader: DataLoader instance (creates new if None)
        job_queue: JobQueue instance (creates new if None)
        tier: Only generate jobs for this difficulty tier
        model: Only generate jobs for this specific model

    Returns:
        Number of jobs inserted
    """
    # Get configuration
    all_models = config.get("models", [])
    models = [model] if model else all_models
    prompt_formats = config.get("benchmark", {}).get(
        "prompt_formats", ["fen_only", "pgn+fen", "cot"]
    )
    data_dir = config.get("paths", {}).get("data_dir", "data")
    db_path = config.get("paths", {}).get("jobs_db", "jobs/jobs.db")

    # Initialize components
    if data_loader is None:
        data_loader = DataLoader(data_dir)
    if job_queue is None:
        job_queue = JobQueue(db_path)

    # Load positions, respecting max_positions_per_tier and optional tier filter
    max_per_tier = config.get("benchmark", {}).get("max_positions_per_tier", 0)
    seed = config.get("benchmark", {}).get("random_seed", 42)
    tiers_to_load = [tier] if tier else ["easy", "medium", "hard", "extreme"]

    positions = []
    for t in tiers_to_load:
        tier_positions = data_loader.load_tier(t)
        if max_per_tier and max_per_tier > 0:
            positions.extend(data_loader.sample(tier_positions, count=max_per_tier, seed=seed))
        else:
            positions.extend(tier_positions)

    if not positions:
        logger.error("No positions loaded from dataset")
        return 0

    if not models:
        logger.error("No models configured")
        return 0

    logger.info(
        f"Generating jobs for {len(positions)} positions, "
        f"{len(models)} models, {len(prompt_formats)} formats"
    )

    # Generate and insert jobs in batches to avoid OOM
    BATCH_SIZE = 10000
    inserted = 0
    total_generated = 0

    batch = []
    for pos in positions:
        for model in models:
            for prompt_format in prompt_formats:
                job_id = generate_job_id(pos["id"], model, prompt_format)
                job_hash = compute_hash(pos["fen"], model, prompt_format, "standard", "1")
                batch.append({
                    "job_id": job_id,
                    "job_type": "standard",
                    "position_id": pos["id"],
                    "model": model,
                    "prompt_format": prompt_format,
                    "trial": 1,
                    "hash": job_hash,
                })
                total_generated += 1

                if len(batch) >= BATCH_SIZE:
                    inserted += job_queue.insert_jobs(batch)
                    batch = []

    if batch:
        inserted += job_queue.insert_jobs(batch)

    logger.info(
        f"Inserted {inserted} jobs ({total_generated - inserted} duplicates skipped)"
    )

    return inserted


def estimate_job_count(
    config: dict[str, Any],
    data_loader: DataLoader | None = None,
) -> dict[str, int]:
    """Estimate the number of jobs without creating them.

    Args:
        config: Configuration dictionary
        data_loader: DataLoader instance

    Returns:
        Dictionary with job count estimates
    """
    models = config.get("models", [])
    prompt_formats = config.get("benchmark", {}).get(
        "prompt_formats", ["fen_only", "pgn+fen", "cot"]
    )
    data_dir = config.get("paths", {}).get("data_dir", "data")

    if data_loader is None:
        data_loader = DataLoader(data_dir)

    positions = data_loader.load_all()

    standard_jobs = len(positions) * len(models) * len(prompt_formats)

    # Estimate correction jobs (assuming ~20% of positions trigger correction)
    correction_estimate = int(standard_jobs * 0.2 * 2)  # *2 for control pairs

    return {
        "positions": len(positions),
        "models": len(models),
        "prompt_formats": len(prompt_formats),
        "standard_jobs": standard_jobs,
        "correction_estimate": correction_estimate,
        "total_estimate": standard_jobs + correction_estimate,
    }
