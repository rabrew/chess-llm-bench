"""Illustrative GPT check: 12 seeded positions (3 per tier) from the plain-board sample, FEN prompt."""
import json
import random
import sys

sys.path.insert(0, ".")
from src import claude_batch as cb
from src.data_loader import DataLoader
from src.llm_client import build_prompt, retry_move_prompt
from src.utils import load_config

PER_TIER = 3
SEED = 42
config = load_config("config/config_claude.yaml")
positions = cb.sample_positions(config, DataLoader("data"))
rng = random.Random(SEED)
pack = []
for tier in cb.TIERS:
    tier_positions = sorted((p for p in positions if p["difficulty"] == tier), key=lambda p: p["id"])
    for p in rng.sample(tier_positions, PER_TIER):
        retry_prompt, retry_system = retry_move_prompt(p["fen"], "fen_only")
        pack.append({"position_id": p["id"], "difficulty": tier, "fen": p["fen"],
                     "stockfish_eval": p["stockfish_eval"], "prompt": build_prompt(fen=p["fen"], prompt_format="fen_only"),
                     "retry_prompt": f"{retry_system}\n\n{retry_prompt}"})
print(json.dumps(pack))
