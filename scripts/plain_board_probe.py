"""Timing and context probe before the plain-board run (specs/plain-board-prompts.md, step 6).

1. Sends every model the longest plain prompt in the sample and checks Ollama read all of it.
2. Times PROBE_POSITIONS positions x every format on the probe models.

Exits 1 if any prompt would be cut off by the model's context window.

    python scripts/plain_board_probe.py --config config/config_plain_board.yaml
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import chess

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader  # noqa: E402
from src.job_generator import subsample_per_tier  # noqa: E402
from src.llm_client import BOARD_TEXT_LEGAL_FORMAT, OllamaClient, build_prompt  # noqa: E402
from src.utils import load_config, setup_logging  # noqa: E402

logger = logging.getLogger("chess_llm_bench")

PROBE_MODELS = ("llama3.2:3b", "qwen2.5:14b", "mixtral:8x7b")
PROBE_POSITIONS = 20
TIERS = ("easy", "medium", "hard", "extreme")


def sampled_positions(config: dict) -> list[dict[str, Any]]:
    """The run's positions, drawn exactly as populate_job_queue draws them."""
    bench = config["benchmark"]
    loader = DataLoader(config["paths"]["data_dir"])
    positions = []
    for tier in TIERS:
        positions += loader.sample(loader.load_tier(tier), count=bench["max_positions_per_tier"],
                                   seed=bench["random_seed"])
    return subsample_per_tier(positions, bench.get("sample_per_tier", 0), bench["random_seed"])


def longest_prompt_position(positions: list[dict[str, Any]]) -> dict[str, Any]:
    """The position whose board_text+legal prompt is longest (most text, most legal moves)."""
    return max(positions, key=lambda p: len(build_prompt(fen=p["fen"], prompt_format=BOARD_TEXT_LEGAL_FORMAT)))


def truncated(prompt_tokens: int | None, num_ctx: int) -> bool:
    """Ollama keeps at most num_ctx tokens; a prompt that fills it was cut off. Unknown counts as cut."""
    return prompt_tokens is None or prompt_tokens >= num_ctx


def check_context(client: OllamaClient, models: list[str], fen: str) -> list[dict[str, Any]]:
    prompt = build_prompt(fen=fen, prompt_format=BOARD_TEXT_LEGAL_FORMAT)
    checks = []
    for model in models:
        num_ctx = client._get_options(model)["num_ctx"]
        result = client.chat(model, prompt)
        if not result["success"]:
            raise RuntimeError(f"{model} failed the context check: {result.get('error')}")
        checks.append({"model": model, "prompt_tokens": result["prompt_tokens"], "num_ctx": num_ctx,
                       "truncated": truncated(result["prompt_tokens"], num_ctx)})
        logger.info(f"context {model}: {result['prompt_tokens']} of {num_ctx} tokens")
    return checks


def time_formats(client: OllamaClient, model: str, positions: list[dict[str, Any]], formats: list[str]) -> dict[str, float]:
    """Mean seconds per prompt for each format (the first call, which loads the model, is not timed)."""
    client.chat(model, build_prompt(fen=chess.STARTING_FEN, prompt_format="fen_only"))
    seconds = {}
    for fmt in formats:
        start = time.time()
        for position in positions:
            client.chat(model, build_prompt(fen=position["fen"], prompt_format=fmt))
        seconds[fmt] = (time.time() - start) / len(positions)
        logger.info(f"timing {model} {fmt}: {seconds[fmt]:.2f} s per prompt")
    return seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="Plain-board timing and context probe.")
    parser.add_argument("--config", default="config/config_plain_board.yaml")
    args = parser.parse_args()
    setup_logging()
    config = load_config(args.config)
    client = OllamaClient(base_url=config["ollama"]["base_url"], timeout=config["ollama"]["timeout"])
    positions = sampled_positions(config)
    context = check_context(client, config["models"], longest_prompt_position(positions)["fen"])
    formats = config["benchmark"]["prompt_formats"]
    spread = positions[::max(1, len(positions) // PROBE_POSITIONS)][:PROBE_POSITIONS]  # every tier
    timing = {model: time_formats(client, model, spread, formats) for model in PROBE_MODELS}
    report = {"positions": len(positions), "context": context, "seconds_per_prompt": timing}
    print(json.dumps(report, indent=2))
    if any(check["truncated"] for check in context):
        raise SystemExit("Some prompts would be cut off: raise num_ctx for those models before the run.")


if __name__ == "__main__":
    main()
