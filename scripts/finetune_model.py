"""Fine-tune a single model on board comprehension using Unsloth QLoRA.

Each invocation handles exactly one model. Run once per model:
    python scripts/finetune_model.py llama3.2:3b
    python scripts/finetune_model.py qwen2.5:7b --dry-run
    python scripts/finetune_model.py llama3.3:70b --config config/finetune_config.yaml

After training, the GGUF is saved to:
    ~/finetuned_gguf/<ollama_ft_tag>/model-q4_k_m.gguf

Import it into Ollama with:
    python scripts/import_finetuned_to_ollama.py <ollama_base>
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.board_state import fen_to_board_description, verify_board_description
from src.utils import setup_logging

INSTRUCTION = (
    "You are given a chess position in FEN notation. "
    "List every piece on the board and its exact square."
)

# Models >= this size need gradient checkpointing to fit in 24GB
GRAD_CKPT_SIZE_THRESHOLD_B = 30

# 70B needs CPU offload on top of gradient checkpointing
CPU_OFFLOAD_SIZE_THRESHOLD_B = 65


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


def estimate_size_b(ollama_base: str) -> float:
    """Rough size estimate from the Ollama tag name (billions of params)."""
    tag = ollama_base.split(":")[1] if ":" in ollama_base else ""
    tag = tag.lower().replace("b", "")
    # mixtral:8x7b → treat as ~47B total but ~12B active; use 47 for memory planning
    if "x" in tag:
        parts = tag.split("x")
        try:
            return float(parts[0]) * float(parts[1])
        except ValueError:
            pass
    # gemma4:e2b, e4b → strip leading 'e'
    tag = tag.lstrip("e")
    try:
        return float(tag)
    except ValueError:
        return 0.0


def format_alpaca_prompt(instruction: str, input_text: str, output: str = "") -> str:
    """Format a record as an Alpaca-style prompt."""
    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n{output}"
    )
    return prompt


def load_dataset_records(dataset_path: str) -> list[dict]:
    records = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def split_held_out(records: list[dict], seed: int = 42, n: int = 200) -> tuple[list[dict], list[dict]]:
    """Split off n held-out records for epoch-level accuracy logging."""
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    return shuffled[n:], shuffled[:n]


def evaluate_held_out(model: Any, tokenizer: Any, held_out: list[dict]) -> float:
    """Run verify_board_description on held-out samples after each epoch.

    Returns mean accuracy over the sample set.
    """
    import torch

    total_acc = 0.0
    for rec in held_out:
        fen = rec["input"].replace("FEN: ", "").strip()
        prompt = format_alpaca_prompt(rec["instruction"], rec["input"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        result = verify_board_description(fen, generated)
        total_acc += result["accuracy"]

    return total_acc / len(held_out) if held_out else 0.0


class EpochAccuracyCallback:
    """TRL TrainerCallback that logs board accuracy after each epoch."""

    def __init__(self, model: Any, tokenizer: Any, held_out: list[dict], logger: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.held_out = held_out
        self.logger = logger

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        acc = evaluate_held_out(self.model, self.tokenizer, self.held_out)
        self.logger.info(f"[Epoch {state.epoch:.1f}] Held-out board accuracy: {acc:.3f}")


def run_finetuning(
    ollama_base: str,
    config_path: str,
    dry_run: bool,
    logger: Any,
) -> None:
    config = load_ft_config(config_path)
    entry = find_model_entry(config, ollama_base)
    training = config["training"]
    paths = config["paths"]

    hf_id = entry["hf_id"]
    ollama_ft_tag = entry["ollama_ft_tag"]
    dataset_path = training["dataset_path"]
    gguf_dir = Path(paths["gguf_output_dir"]) / ollama_ft_tag
    checkpoint_dir = Path(paths["checkpoints_dir"]) / ollama_ft_tag

    size_b = estimate_size_b(ollama_base)
    use_grad_ckpt = size_b >= GRAD_CKPT_SIZE_THRESHOLD_B
    use_cpu_offload = size_b >= CPU_OFFLOAD_SIZE_THRESHOLD_B

    logger.info(f"Model      : {ollama_base} → {hf_id}")
    logger.info(f"FT tag     : {ollama_ft_tag}")
    logger.info(f"Size est.  : {size_b}B params")
    logger.info(f"Grad ckpt  : {use_grad_ckpt}")
    logger.info(f"CPU offload: {use_cpu_offload}")
    logger.info(f"Dataset    : {dataset_path}")
    logger.info(f"GGUF out   : {gguf_dir}")

    if not Path(dataset_path).exists():
        if dry_run:
            logger.info(
                f"[dry-run] Dataset not found at {dataset_path} — "
                "run 'python scripts/generate_board_dataset.py' first."
            )
            logger.info("[dry-run] Config and model lookup validated successfully.")
            return
        logger.error(
            f"Dataset not found: {dataset_path}\n"
            "Run: python scripts/generate_board_dataset.py"
        )
        sys.exit(1)

    records = load_dataset_records(dataset_path)
    logger.info(f"Loaded {len(records)} training records.")

    train_records, held_out = split_held_out(records, seed=training.get("seed", 42))
    logger.info(f"Train: {len(train_records)} | Held-out: {len(held_out)}")

    if dry_run:
        logger.info("[dry-run] Dataset validated. No training will run.")
        logger.info(f"[dry-run] First record input: {records[0]['input']}")
        return

    # --- Imports deferred so dry-run works without GPU deps installed ---
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, TrainingArguments
        from datasets import Dataset
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git' trl datasets")
        sys.exit(1)

    # --- Load model ---
    load_kwargs: dict[str, Any] = {
        "model_name": hf_id,
        "max_seq_length": training["max_seq_length"],
        "dtype": None,  # auto-detect
        "load_in_4bit": True,
    }
    if use_grad_ckpt:
        load_kwargs["use_gradient_checkpointing"] = True
    if use_cpu_offload:
        load_kwargs["device_map"] = "auto"

    logger.info(f"Loading {hf_id} in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)

    model = FastLanguageModel.get_peft_model(
        model,
        r=training["lora_rank"],
        lora_alpha=training["lora_alpha"],
        target_modules=training["lora_target_modules"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=use_grad_ckpt,
        random_state=training.get("seed", 42),
    )

    # --- Format dataset ---
    def format_record(rec: dict) -> dict:
        text = format_alpaca_prompt(rec["instruction"], rec["input"], rec["output"])
        return {"text": text}

    hf_dataset = Dataset.from_list([format_record(r) for r in train_records])

    # --- Accuracy callback ---
    accuracy_cb = EpochAccuracyCallback(model, tokenizer, held_out, logger)

    # --- Train ---
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_dataset,
        dataset_text_field="text",
        max_seq_length=training["max_seq_length"],
        args=TrainingArguments(
            per_device_train_batch_size=training["per_device_train_batch_size"],
            gradient_accumulation_steps=training["gradient_accumulation_steps"],
            num_train_epochs=training["num_epochs"],
            learning_rate=training["learning_rate"],
            warmup_ratio=training["warmup_ratio"],
            fp16=True,
            logging_steps=50,
            output_dir=str(checkpoint_dir),
            seed=training.get("seed", 42),
        ),
        callbacks=[accuracy_cb],
    )
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")

    # --- Export GGUF ---
    gguf_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exporting GGUF to {gguf_dir}...")
    model.save_pretrained_gguf(
        str(gguf_dir),
        tokenizer,
        quantization_method=training.get("gguf_quantization", "q4_k_m"),
    )
    logger.info(f"GGUF saved: {gguf_dir}/model-q4_k_m.gguf")
    logger.info(f"Done. Run: python scripts/import_finetuned_to_ollama.py {ollama_base}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a single model on chess board comprehension."
    )
    parser.add_argument("ollama_base", help="Ollama base model tag (e.g. llama3.2:3b)")
    parser.add_argument(
        "--config", default="config/finetune_config.yaml",
        help="Fine-tuning config path (default: config/finetune_config.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate dataset and config without running training",
    )
    args = parser.parse_args()

    logger = setup_logging("finetune_model")
    run_finetuning(args.ollama_base, args.config, args.dry_run, logger)


if __name__ == "__main__":
    main()
