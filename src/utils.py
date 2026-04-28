"""Utility functions for Chess LLM Benchmark."""

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = "config/config.yaml") -> dict[str, Any]:
    """Load configuration from YAML file with environment variable overrides.

    Environment variables override config values using CHESS_<SECTION>_<KEY> format.
    Example: CHESS_STOCKFISH_DEPTH=30 overrides stockfish.depth
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply environment variable overrides
    for section, values in config.items():
        if isinstance(values, dict):
            for key, value in values.items():
                env_key = f"CHESS_{section.upper()}_{key.upper()}"
                env_value = os.environ.get(env_key)
                if env_value is not None:
                    # Try to preserve type
                    if isinstance(value, bool):
                        config[section][key] = env_value.lower() in ("true", "1", "yes")
                    elif isinstance(value, int):
                        config[section][key] = int(env_value)
                    elif isinstance(value, float):
                        config[section][key] = float(env_value)
                    else:
                        config[section][key] = env_value

    return config


def compute_hash(*args: str) -> str:
    """Compute SHA256 hash from concatenated string arguments.

    Used for duplicate detection in job generation.
    """
    combined = "".join(str(arg) for arg in args)
    return hashlib.sha256(combined.encode()).hexdigest()


def setup_logging(
    name: str = "chess_llm_bench",
    level: int = logging.INFO,
    log_file: str | None = None
) -> logging.Logger:
    """Set up logging with console and optional file output.

    Args:
        name: Logger name
        level: Logging level (default INFO)
        log_file: Optional file path for log output

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear any existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        ensure_dir(Path(log_file).parent)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def ensure_dir(path: Path | str) -> Path:
    """Ensure directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists

    Returns:
        Path object of the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_project_dirs(config: dict[str, Any]) -> None:
    """Ensure all project directories exist based on config.

    Args:
        config: Configuration dictionary with paths section
    """
    paths = config.get("paths", {})

    # Ensure data and jobs directories
    ensure_dir(paths.get("data_dir", "data"))
    ensure_dir(Path(paths.get("jobs_db", "jobs/db/jobs.db")).parent)

    # Ensure results directories
    ensure_dir(Path(paths.get("results_file", "results/evaluations.jsonl")).parent)
    ensure_dir(paths.get("logs_dir", "results/logs"))
    ensure_dir(paths.get("plots_dir", "results/plots"))
    ensure_dir(paths.get("metrics_dir", "results/metrics"))


def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_run_id() -> str:
    """Generate a unique run ID based on current timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def parse_model_info(model_tag: str) -> dict[str, Any]:
    """Parse model tag into family and size information.

    Args:
        model_tag: Ollama model tag (e.g., "qwen2.5:7b")

    Returns:
        Dictionary with family, size_str, and size_b (billions)
    """
    # Map model prefixes to family names
    family_map = {
        "qwen": "qwen",
        "llama": "llama",
        "mistral": "mistral",
        "phi": "phi",
        "gemma": "gemma",
    }

    parts = model_tag.split(":")
    model_name = parts[0]
    size_str = parts[1] if len(parts) > 1 else ""

    # Determine family
    family = "unknown"
    for prefix, fam in family_map.items():
        if model_name.lower().startswith(prefix):
            family = fam
            break

    # Parse size (e.g., "7b" -> 7, "14b" -> 14)
    size_b = 0
    if size_str:
        size_clean = size_str.lower().replace("b", "")
        try:
            size_b = int(size_clean)
        except ValueError:
            try:
                size_b = float(size_clean)
            except ValueError:
                pass

    return {
        "family": family,
        "size_str": size_str,
        "size_b": size_b,
        "full_tag": model_tag,
    }


def clamp(value: int | float, min_val: int | float, max_val: int | float) -> int | float:
    """Clamp a value to a range.

    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))
