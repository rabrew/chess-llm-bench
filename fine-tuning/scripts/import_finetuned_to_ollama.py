"""Import a fine-tuned GGUF model into Ollama.

Usage:
    python fine-tuning/scripts/import_finetuned_to_ollama.py llama3.2:3b
    python fine-tuning/scripts/import_finetuned_to_ollama.py qwen2.5:7b --config fine-tuning/config/finetune_config.yaml
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
FT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FT_SRC))

from src.utils import setup_logging

MODELFILE_TEMPLATE = """\
FROM {gguf_path}
PARAMETER num_ctx 1024
PARAMETER num_predict 400
"""


def load_ft_config(config_path: str) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_model_entry(config: dict, ollama_base: str) -> dict:
    for entry in config.get("models", []):
        if entry["ollama_base"] == ollama_base:
            return entry
    raise ValueError(
        f"Model '{ollama_base}' not found in config. "
        f"Available: {[m['ollama_base'] for m in config.get('models', [])]}"
    )


def run(ollama_base: str, config_path: str, logger: Any) -> None:
    config = load_ft_config(config_path)
    entry = find_model_entry(config, ollama_base)
    paths = config["paths"]

    ollama_ft_tag = entry["ollama_ft_tag"]
    gguf_base = Path(paths["gguf_output_dir"]) / ollama_ft_tag
    modelfiles_dir = Path(paths["ollama_modelfiles_dir"])
    modelfile_path = modelfiles_dir / f"{ollama_ft_tag.replace(':', '_')}.Modelfile"

    # Unsloth appends _gguf to the output dir name; check _gguf variant first.
    gguf_dir = gguf_base
    gguf_files: list = []
    for candidate in [Path(str(gguf_base) + "_gguf"), gguf_base]:
        if candidate.exists():
            found = sorted(candidate.glob("*.gguf"))
            if found:
                gguf_dir = candidate
                gguf_files = found
                break
    if not gguf_files:
        logger.error(
            f"No .gguf file found in {gguf_dir}\n"
            f"Run first: python fine-tuning/scripts/finetune_model.py {ollama_base}"
        )
        sys.exit(1)
    if len(gguf_files) > 1:
        logger.warning(f"Multiple .gguf files found — using: {gguf_files[0]}")
    gguf_path = gguf_files[0]

    logger.info(f"Model tag  : {ollama_ft_tag}")
    logger.info(f"GGUF path  : {gguf_path}")
    logger.info(f"Modelfile  : {modelfile_path}")

    # Write Modelfile
    modelfiles_dir.mkdir(parents=True, exist_ok=True)
    modelfile_path.write_text(MODELFILE_TEMPLATE.format(gguf_path=gguf_path))
    logger.info(f"Wrote Modelfile: {modelfile_path}")

    # Create Ollama model
    logger.info(f"Running: ollama create {ollama_ft_tag} -f {modelfile_path}")
    result = subprocess.run(
        ["ollama", "create", ollama_ft_tag, "-f", str(modelfile_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"ollama create failed:\n{result.stderr}")
        sys.exit(1)
    logger.info(result.stdout.strip() or "ollama create succeeded.")

    # Verify with ollama list
    logger.info("Verifying with 'ollama list'...")
    list_result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
    )
    if ollama_ft_tag in list_result.stdout:
        logger.info(f"Confirmed: {ollama_ft_tag} is now available in Ollama.")
    else:
        logger.warning(
            f"'{ollama_ft_tag}' not found in 'ollama list' output. "
            "Check manually with: ollama list"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a fine-tuned GGUF model into Ollama."
    )
    parser.add_argument("ollama_base", help="Ollama base model tag (e.g. llama3.2:3b)")
    parser.add_argument(
        "--config", default="fine-tuning/config/finetune_config.yaml",
        help="Fine-tuning config path (default: fine-tuning/config/finetune_config.yaml)",
    )
    args = parser.parse_args()

    logger = setup_logging("import_finetuned")
    run(args.ollama_base, args.config, logger)


if __name__ == "__main__":
    main()
