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


FINE_TUNED_SUFFIX = "-board-ft"

# Exact Ollama tag -> (family, total params in billions, active params in billions).
# Family is the model generation, so different generations never share a row in
# the family table. See specs/model-registry.md.
MODEL_REGISTRY: dict[str, tuple[str, float, float]] = {
    "llama3.2:3b": ("llama3", 3.0, 3.0),
    "llama3.1:8b": ("llama3", 8.0, 8.0),
    "llama3.3:70b": ("llama3", 70.0, 70.0),
    "gemma3:4b": ("gemma3", 4.0, 4.0),
    "gemma3:12b": ("gemma3", 12.0, 12.0),
    "gemma4:e2b": ("gemma4", 2.0, 2.0),  # "effective" 2B
    "gemma4:e4b": ("gemma4", 4.0, 4.0),  # "effective" 4B
    "gemma4:26b": ("gemma4", 26.0, 26.0),
    "gemma4:31b": ("gemma4", 31.0, 31.0),
    "qwen2.5:7b": ("qwen2.5", 7.0, 7.0),
    "qwen2.5:14b": ("qwen2.5", 14.0, 14.0),
    "qwen2.5:32b": ("qwen2.5", 32.0, 32.0),
    "qwen2.5:72b": ("qwen2.5", 72.0, 72.0),
    "deepseek-r1:7b": ("deepseek-r1", 7.0, 7.0),
    "deepseek-r1:14b": ("deepseek-r1", 14.0, 14.0),
    "mistral:7b": ("mistral", 7.0, 7.0),
    "mixtral:8x7b": ("mixtral", 46.7, 12.9),  # mixture of experts
    "wizardlm2:7b": ("wizardlm2", 7.0, 7.0),
    "phi4:14b": ("phi4", 14.0, 14.0),
    "solar:10.7b": ("solar", 10.7, 10.7),
    "codellama:34b": ("codellama", 34.0, 34.0),
    "yi:34b": ("yi", 34.0, 34.0),
    "command-r:35b": ("command-r", 35.0, 35.0),
}


def parse_model_info(model_tag: str) -> dict[str, Any]:
    """Look up a model's family and size in MODEL_REGISTRY.

    A "-board-ft" suffix (fine-tuned models) resolves like its base model.

    Raises:
        ValueError: if the tag is not in MODEL_REGISTRY. Add new models there
            rather than letting them fall into an "unknown" bucket.
    """
    fine_tuned = model_tag.endswith(FINE_TUNED_SUFFIX)
    base_tag = model_tag.removesuffix(FINE_TUNED_SUFFIX)
    if base_tag not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model tag {model_tag!r}: add it to MODEL_REGISTRY in src/utils.py"
        )
    family, size_b, active_b = MODEL_REGISTRY[base_tag]
    return {
        "family": family,
        "size_str": base_tag.split(":", 1)[1],
        "size_b": size_b,
        "active_b": active_b,
        "is_moe": active_b < size_b,
        "fine_tuned": fine_tuned,
        "full_tag": base_tag,
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
