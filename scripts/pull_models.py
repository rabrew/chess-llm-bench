#!/usr/bin/env python3
"""CLI script to pull Ollama models."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_client import OllamaClient
from src.utils import load_config, setup_logging


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Pull all configured Ollama models"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Pull a specific model instead of all configured models",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List currently available models",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Setup logging
    import logging
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=level)
    logger = logging.getLogger("chess_llm_bench")

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    # Initialize client
    ollama_config = config.get("ollama", {})
    client = OllamaClient(
        base_url=ollama_config.get("base_url", "http://localhost:11434"),
    )

    # Check if Ollama is running
    if not client.is_available():
        logger.error("Ollama is not running. Please start Ollama first.")
        sys.exit(1)

    if args.list:
        # List available models
        models = client.list_models()
        print("\nAvailable models:")
        for model in models:
            print(f"  - {model}")
        return

    # Determine which models to pull
    if args.model:
        models = [args.model]
    else:
        models = config.get("models", [])

    if not models:
        logger.error("No models configured")
        sys.exit(1)

    # Check which models are already available
    available = set(client.list_models())

    print(f"\nPulling {len(models)} models...")

    success_count = 0
    for model in models:
        if model in available:
            print(f"  {model}: already available")
            success_count += 1
            continue

        print(f"  {model}: pulling...")
        if client.pull_model(model):
            print(f"  {model}: done")
            success_count += 1
        else:
            print(f"  {model}: FAILED")

    print(f"\nSuccessfully pulled/verified {success_count}/{len(models)} models")


if __name__ == "__main__":  # pragma: no cover
    main()
