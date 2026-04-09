"""
Live parser test — queries each model with all 3 prompt formats and checks
that parse_response extracts all three fields without errors.

Run BEFORE the full benchmark to catch format bugs early.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.llm_client import OllamaClient, build_prompt, parse_response

FEN      = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
PGN      = "1. e4 e5 2. Nf3 Nc6 3. Bc4"
FORMATS  = ["fen_only", "pgn+fen", "cot"]

MODELS = [
    "llama3.2:3b",
    "gemma3:4b",
    "qwen2.5:7b",
    "mistral:7b",
    "wizardlm2:7b",
    "llama3.1:8b",
    "gemma3:12b",
    "qwen2.5:14b",
    "phi4:14b",
    "solar:10.7b",
    "qwen2.5:32b",
    "codellama:34b",
    "yi:34b",
    "command-r:35b",
    "mixtral:8x7b",
    "llama3.3:70b",
    "qwen2.5:72b",
]

client = OllamaClient(timeout=180)

overall_pass = True

for model in MODELS:
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    model_ok = True

    for fmt in FORMATS:
        prompt = build_prompt(FEN, pgn_moves=PGN, prompt_format=fmt)
        result = client.chat(model, prompt)

        if not result["success"]:
            print(f"  [{fmt:10s}] OLLAMA ERROR: {result.get('error')}")
            model_ok = False
            continue

        raw  = result["response"]
        parsed = parse_response(raw)

        missing = [f for f in ["eval", "move", "explanation"] if parsed[f] is None]
        errors  = parsed["parse_errors"]

        if missing or errors:
            print(f"  [{fmt:10s}] FAIL — missing={missing} errors={errors}")
            print(f"             raw: {repr(raw[:300])}")
            model_ok = False
            overall_pass = False
        else:
            print(f"  [{fmt:10s}] OK   — eval={parsed['eval']} move={parsed['move']} side={parsed['side_claimed']}")

    if model_ok:
        print(f"  ✓ All formats parsed cleanly")

print(f"\n{'='*60}")
print(f"Result: {'ALL PASS' if overall_pass else 'FAILURES DETECTED — fix parser before running benchmark'}")
