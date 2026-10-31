"""Wait for read-only Ollama status to confirm that GPU handover is safe."""

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_config, setup_logging  # noqa: E402

DEFAULT_TIMEOUT_SECONDS = 600.0
DEFAULT_POLL_SECONDS = 5.0
REQUEST_TIMEOUT_SECONDS = 5.0
logger = logging.getLogger("chess_llm_bench")


def wait_until_idle(base_url: str, timeout: float = DEFAULT_TIMEOUT_SECONDS,
                    poll_interval: float = DEFAULT_POLL_SECONDS) -> None:
    """Wait for automatic expiry; never unload a model or stop a process."""
    if any(not math.isfinite(value) or value <= 0 for value in (timeout, poll_interval)):
        raise ValueError("Idle timeout and polling interval must be finite and positive")
    deadline = time.monotonic() + timeout
    last_state = "unknown"
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            response = requests.get(base_url.rstrip("/") + "/api/ps",
                                    timeout=min(REQUEST_TIMEOUT_SECONDS, remaining))
            response.raise_for_status()
            models = response.json().get("models")
            if not isinstance(models, list):
                raise ValueError("Status has no model list")
            if not models:
                logger.info("Ollama has no loaded models; GPU handover is ready")
                return
            last_state = f"{len(models)} loaded model(s)"
        except (requests.RequestException, ValueError, TypeError, AttributeError):
            last_state = "unavailable or invalid status"
        logger.info(f"Waiting for Ollama to become idle: {last_state}")
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(poll_interval, remaining))
    raise TimeoutError(f"Ollama did not confirm idle within {timeout:g}s ({last_state}); no processes were stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_SECONDS)
    args = parser.parse_args()
    setup_logging()
    config = load_config(args.config)
    timeout = args.timeout if args.timeout is not None else config.get("lc0", {}).get(
        "handover_timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    wait_until_idle(config.get("ollama", {}).get("base_url", "http://localhost:11434"),
                    timeout=timeout, poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
