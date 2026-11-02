"""Utility functions for Chess LLM Benchmark."""

import hashlib
import logging
import os
import re
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

    Strips -board-ft suffix before parsing so fine-tuned model tags
    (e.g. "llama3.2:3b-board-ft") map to the same family/size as the base.

    Args:
        model_tag: Ollama model tag (e.g., "qwen2.5:7b" or "qwen2.5:7b-board-ft")

    Returns:
        Dictionary with family, size_str, and size_b (billions)
    """
    model_tag = model_tag.removesuffix("-board-ft").removesuffix("-legal-ft")
    model_name, separator, size_str = model_tag.partition(":")
    prefixes = ("codellama", "command-r", "deepseek", "wizardlm", "mixtral",
                "mistral", "qwen", "llama", "phi", "gemma", "solar", "yi")
    family = next((prefix for prefix in prefixes if model_name.lower().startswith(prefix)), None)
    match = re.fullmatch(r"(e)?(?:(\d+)x)?(\d+(?:\.\d+)?)b", size_str.lower())
    if family is None or not separator or match is None:
        raise ValueError(f"Unrecognised model family or parameter size: {model_tag!r}")
    effective, experts, parameters = match.groups()
    total = float(parameters) * (int(experts) if experts else 1)
    if total <= 0:
        raise ValueError(f"Model parameter size must be positive: {model_tag!r}")
    is_moe = experts is not None or model_tag == "gemma4:26b"
    return {
        "family": family, "size_str": size_str, "size_b": total,
        "full_tag": model_tag, "is_moe": is_moe,
        "effective_parameters": effective is not None,
        "active_size_b": 12.9 if model_tag == "mixtral:8x7b" else None,
        "scaling_eligible": not is_moe and effective is None,
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
