"""Rebuild a fine-tuned model's GGUF from its saved LoRA adapter, without Unsloth's merge.

Merges the adapter into the full-precision base with PEFT, checks the merge really
changed the weights, then converts and quantizes with llama.cpp.
See fine-tuning/specs/gguf-export-check.md.

Usage:
    python fine-tuning/scripts/reexport_gguf.py gemma4:e2b
    python fine-tuning/scripts/reexport_gguf.py gemma4:e2b --checkpoint ~/finetuned_checkpoints/gemma4:e2b-board-ft/checkpoint-2476
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from import_finetuned_to_ollama import find_model_entry, load_ft_config  # noqa: E402
from src.utils import setup_logging  # noqa: E402

DEFAULT_CONFIG = "fine-tuning/config/finetune_config.yaml"
QUANTIZED_SUFFIXES = ("-unsloth-bnb-4bit", "-bnb-4bit")
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
)


def full_precision_base_id(base_id: str) -> str:
    for suffix in QUANTIZED_SUFFIXES:
        if base_id.endswith(suffix):
            return base_id[: -len(suffix)]
    return base_id


def latest_checkpoint(run_dir: Path) -> Path:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"No checkpoint folder: {run_dir}")
    steps = [(int(m.group(1)), p) for p in run_dir.iterdir() if (m := CHECKPOINT_PATTERN.match(p.name)) and p.is_dir()]
    if not steps:
        raise FileNotFoundError(f"No checkpoint-* folders in {run_dir}")
    return max(steps)[1]


def require_adapter_config(checkpoint: Path) -> None:
    if not (checkpoint / "adapter_config.json").is_file():
        raise FileNotFoundError(f"No adapter_config.json in {checkpoint}")


def ensure_work_dir_free(work_dir: Path) -> None:
    if work_dir.exists():
        raise FileExistsError(f"Work dir already exists, remove it first: {work_dir}")


def convert_command(llama_dir: Path, merged_dir: Path, outfile: Path) -> list[str]:
    return [
        sys.executable, str(llama_dir / "convert_hf_to_gguf.py"), str(merged_dir),
        "--outtype", "bf16", "--outfile", str(outfile),
    ]


def quantize_command(llama_dir: Path, source: Path, target: Path, quantization: str) -> list[str]:
    return [str(llama_dir / "llama-quantize"), str(source), str(target), quantization.upper()]


def run_step(cmd: list[str], runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> None:
    completed = runner(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed (exit {completed.returncode}):\n{completed.stderr[-3000:]}")


def _lora_b_params(model: Any) -> list[Any]:
    return [param for name, param in model.named_parameters() if "lora_B" in name]


def check_adapter_attached(model: Any) -> int:
    """Return the number of LoRA layers; raise if none or if every lora_B is still zero."""
    lora_b = _lora_b_params(model)
    if not lora_b:
        raise RuntimeError("Model has no LoRA layers: the adapter did not attach to any module")
    if all(float(param.detach().abs().max()) == 0.0 for param in lora_b):
        raise RuntimeError("Every lora_B weight is zero: the adapter is untrained or failed to load")
    return len(lora_b)


def check_weight_changed(before: Any, after: Any) -> None:
    import torch

    if torch.equal(before, after):
        raise RuntimeError("Merged weight is unchanged: the LoRA delta was not applied")


def _first_lora_base_weight(model: Any) -> tuple[str, Any]:
    for name, module in model.named_modules():
        if hasattr(module, "lora_B") and hasattr(module, "base_layer"):
            return name, module.base_layer.weight.detach().clone()
    raise RuntimeError("No LoRA-wrapped module found")


def merge_and_check(peft_model: Any) -> Any:
    """Merge the adapter, verifying one targeted weight actually changed."""
    check_adapter_attached(peft_model)
    name, before = _first_lora_base_weight(peft_model)
    merged = peft_model.merge_and_unload()
    after = merged.get_submodule(name.removeprefix("base_model.model.")).weight.detach()
    check_weight_changed(before, after)
    return merged


def load_merged(checkpoint: Path, logger: Any) -> Any:
    import torch
    import transformers
    from peft import PeftModel

    adapter_base = json.loads((checkpoint / "adapter_config.json").read_text())["base_model_name_or_path"]
    base_id = full_precision_base_id(adapter_base)
    architecture = transformers.AutoConfig.from_pretrained(base_id).architectures[0]
    logger.info(f"Base {base_id} ({architecture}), adapter {checkpoint}")
    base = getattr(transformers, architecture).from_pretrained(base_id, dtype=torch.bfloat16, device_map="cpu")
    peft_model = PeftModel.from_pretrained(base, str(checkpoint))
    logger.info(f"LoRA layers attached: {check_adapter_attached(peft_model)}")
    return merge_and_check(peft_model)


def save_merged(merged: Any, checkpoint: Path, merged_dir: Path) -> None:
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    for filename in TOKENIZER_FILES:
        if (checkpoint / filename).is_file():
            shutil.copy2(checkpoint / filename, merged_dir / filename)


def replace_model_gguf(gguf_dir: Path, new_file: Path, logger: Any) -> Path:
    gguf_dir.mkdir(parents=True, exist_ok=True)
    for old in gguf_dir.glob("*.gguf"):
        if "mmproj" not in old.name.lower():
            logger.info(f"Removing old export {old}")
            old.unlink()
    target = gguf_dir / new_file.name
    shutil.move(str(new_file), target)
    return target


def run(ollama_base: str, config_path: str, checkpoint_arg: str | None) -> Path:
    logger = setup_logging("reexport_gguf")
    config = load_ft_config(config_path)
    entry = find_model_entry(config, ollama_base)
    paths, quantization = config["paths"], config["training"]["gguf_quantization"]
    ft_tag = entry["ollama_ft_tag"]

    checkpoint = Path(checkpoint_arg) if checkpoint_arg else latest_checkpoint(Path(paths["checkpoints_dir"]) / ft_tag)
    require_adapter_config(checkpoint)
    work_dir = Path(paths["merge_work_dir"]) / ft_tag.replace(":", "_")
    ensure_work_dir_free(work_dir)
    merged_dir, llama_dir = work_dir / "merged", Path(paths["llama_cpp_dir"]).expanduser()
    stem = full_precision_base_id(json.loads((checkpoint / "adapter_config.json").read_text())["base_model_name_or_path"]).split("/")[-1]
    bf16_file, quant_file = work_dir / f"{stem}.BF16.gguf", work_dir / f"{stem}.{quantization.upper()}.gguf"

    save_merged(load_merged(checkpoint, logger), checkpoint, merged_dir)
    logger.info("Converting to bf16 GGUF...")
    run_step(convert_command(llama_dir, merged_dir, bf16_file))
    logger.info(f"Quantizing to {quantization.upper()}...")
    run_step(quantize_command(llama_dir, bf16_file, quant_file, quantization))
    target = replace_model_gguf(Path(paths["gguf_output_dir"]) / f"{ft_tag}_gguf", quant_file, logger)
    shutil.rmtree(work_dir)
    logger.info(f"Re-exported GGUF: {target}")
    logger.info(f"Next: import_finetuned_to_ollama.py {ollama_base} && verify_export.py {ollama_base}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ollama_base")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", help="Adapter checkpoint folder (default: latest checkpoint-*)")
    args = parser.parse_args()
    run(args.ollama_base, args.config, args.checkpoint)


if __name__ == "__main__":
    main()
