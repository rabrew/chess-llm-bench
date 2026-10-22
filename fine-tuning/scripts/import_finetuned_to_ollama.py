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

BLOCK_QUOTE = '"""'


def _top_level_lines(text: str) -> list[tuple[str, bool]]:
    """Pair each line with whether it starts outside a triple-quoted block."""
    lines, inside = [], False
    for line in text.splitlines():
        lines.append((line, not inside))
        if line.count(BLOCK_QUOTE) % 2 == 1:
            inside = not inside
    return lines


def _is_from(line: str, top_level: bool) -> bool:
    return top_level and line.startswith("FROM ")


def build_ft_modelfile(base_modelfile: str, gguf_path: Path) -> str:
    """The base model's Modelfile with its FROM line(s) pointing at the fine-tuned GGUF.

    TEMPLATE, SYSTEM, PARAMETER, RENDERER and PARSER are kept, so base and
    fine-tuned models are prompted and sampled identically.
    """
    out, replaced = [], False
    for line, top_level in _top_level_lines(base_modelfile):
        if not _is_from(line, top_level):
            out.append(line)
        elif not replaced:
            out.append(f"FROM {gguf_path}")
            replaced = True
    if not replaced:
        raise ValueError("Base Modelfile has no FROM line")
    return "\n".join(out) + "\n"


def modelfile_settings(modelfile: str) -> str:
    """Everything that affects prompting and sampling: the Modelfile minus FROM, comments and blanks.

    Directives are sorted, because Ollama may reorder PARAMETER lines when it saves a model.
    """
    directives: list[list[str]] = []
    for line, top_level in _top_level_lines(modelfile):
        if not top_level:
            directives[-1].append(line)
        elif not (_is_from(line, top_level) or line.startswith("#") or not line.strip()):
            directives.append([line])
    return "\n".join(sorted("\n".join(lines) for lines in directives))


def base_modelfile(ollama_base: str) -> str:
    result = subprocess.run(["ollama", "show", ollama_base, "--modelfile"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ollama show {ollama_base} failed: {result.stderr.strip()}")
    return result.stdout


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


def find_model_gguf(dirs: list[Path], quantization: str) -> Path:
    """Return the language-model GGUF from the first folder that has one.

    Skips multimodal vision projectors (*mmproj*). Among several model files,
    picks the one whose name contains the quantization method.
    """
    for folder in dirs:
        if not folder.exists():
            continue
        models = sorted(f for f in folder.glob("*.gguf") if "mmproj" not in f.name.lower())
        if len(models) == 1:
            return models[0]
        matching = [f for f in models if quantization.lower() in f.name.lower()]
        if len(matching) == 1:
            return matching[0]
        if models:
            raise FileNotFoundError(
                f"Cannot pick one model GGUF in {folder}: {[f.name for f in models]}"
            )
    raise FileNotFoundError(f"No language-model .gguf file found in {[str(d) for d in dirs]}")


def run(ollama_base: str, config_path: str, logger: Any) -> None:
    config = load_ft_config(config_path)
    entry = find_model_entry(config, ollama_base)
    paths = config["paths"]

    ollama_ft_tag = entry["ollama_ft_tag"]
    gguf_base = Path(paths["gguf_output_dir"]) / ollama_ft_tag
    modelfiles_dir = Path(paths["ollama_modelfiles_dir"])
    modelfile_path = modelfiles_dir / f"{ollama_ft_tag.replace(':', '_')}.Modelfile"

    quantization = config["training"].get("gguf_quantization", "q4_k_m")
    try:
        gguf_path = find_model_gguf([Path(str(gguf_base) + "_gguf"), gguf_base], quantization)
    except FileNotFoundError as exc:
        logger.error(f"{exc}\nRun first: python fine-tuning/scripts/finetune_model.py {ollama_base}")
        sys.exit(1)

    logger.info(f"Model tag  : {ollama_ft_tag}")
    logger.info(f"GGUF path  : {gguf_path}")
    logger.info(f"Modelfile  : {modelfile_path}")

    # Write Modelfile
    modelfiles_dir.mkdir(parents=True, exist_ok=True)
    modelfile_path.write_text(build_ft_modelfile(base_modelfile(ollama_base), gguf_path))
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
