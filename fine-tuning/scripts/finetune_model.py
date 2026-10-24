"""Fine-tune a single model on board comprehension using Unsloth QLoRA.

Each invocation handles exactly one model. Run once per model:
    python fine-tuning/scripts/finetune_model.py llama3.2:3b
    python fine-tuning/scripts/finetune_model.py qwen2.5:7b --dry-run
    python fine-tuning/scripts/finetune_model.py llama3.3:70b --config fine-tuning/config/finetune_config.yaml

After training, the GGUF is saved to:
    ~/finetuned_gguf/<ollama_ft_tag>/model-q4_k_m.gguf

Import it into Ollama with:
    python fine-tuning/scripts/import_finetuned_to_ollama.py <ollama_base>
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from transformers import TrainerCallback

REPO_ROOT = Path(__file__).resolve().parents[2]
FT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FT_SRC))

from board_state import fen_to_board_description, verify_board_description
from src.utils import parse_model_info, setup_logging

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_record import DEFAULT_RUNS_DIR, build_run_record, write_run_record  # noqa: E402

INSTRUCTION = (
    "You are given a chess position in FEN notation. "
    "List every piece on the board and its exact square."
)

# Models >= this size need gradient checkpointing to fit in 24GB
GRAD_CKPT_SIZE_THRESHOLD_B = 30

# 70B needs CPU offload on top of gradient checkpointing
CPU_OFFLOAD_SIZE_THRESHOLD_B = 65

TRAINER_LOGGING_STEPS = 50

# Gemma's flex-attention generate path crashes mid-training (fine-tuning/specs/gemma-training-fixes.md)
ALLOWED_ATTN_IMPLEMENTATIONS = {"eager", "sdpa", "flash_attention_2"}


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


def build_load_kwargs(
    entry: dict, training: dict, use_grad_ckpt: bool, use_cpu_offload: bool,
) -> dict[str, Any]:
    """Keyword arguments for FastLanguageModel.from_pretrained for one config entry."""
    load_kwargs: dict[str, Any] = {
        "model_name": entry["hf_id"],
        "max_seq_length": training["max_seq_length"],
        "dtype": None,  # auto-detect
        "load_in_4bit": True,
    }
    if use_grad_ckpt:
        load_kwargs["use_gradient_checkpointing"] = True
    if use_cpu_offload:
        load_kwargs["device_map"] = "auto"
    attn = entry.get("attn_implementation")
    if attn is not None:
        if attn not in ALLOWED_ATTN_IMPLEMENTATIONS:
            raise ValueError(
                f"{entry['ollama_base']}: attn_implementation must be one of "
                f"{sorted(ALLOWED_ATTN_IMPLEMENTATIONS)}, got {attn!r}"
            )
        load_kwargs["attn_implementation"] = attn
    return load_kwargs


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
    """Split off n held-out records for epoch-level accuracy logging.

    Raises:
        ValueError: if a FEN (``input``) appears twice; the positional split
            could then put the same position in both sets (F-2).
    """
    inputs = [r["input"] for r in records]
    if len(set(inputs)) != len(inputs):
        raise ValueError(f"{len(inputs) - len(set(inputs))} duplicate FENs in the dataset; deduplicate before splitting")
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
        # text= keyword: multimodal processors (Gemma 4) treat a positional argument as images
        inputs = tokenizer(text=prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                max_length=None,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        result = verify_board_description(fen, generated)
        total_acc += result["accuracy"]

    return total_acc / len(held_out) if held_out else 0.0


class EpochAccuracyCallback(TrainerCallback):
    """TRL TrainerCallback that logs board accuracy after each epoch."""

    def __init__(self, model: Any, tokenizer: Any, held_out: list[dict], logger: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.held_out = held_out
        self.logger = logger
        self.history: list[dict[str, float]] = []

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        # Progress indicator only (F-9 is the real measurement), so a failure here must not end training.
        try:
            acc = evaluate_held_out(self.model, self.tokenizer, self.held_out)
        except Exception as exc:
            self.logger.error(f"[Epoch {state.epoch:.1f}] Held-out accuracy check failed: {exc}", exc_info=True)
            return
        self.logger.info(f"[Epoch {state.epoch:.1f}] Held-out board accuracy: {acc:.3f}")
        self.history.append({"epoch": float(state.epoch), "held_out_accuracy": acc})


def ensure_outputs_free(dirs: list[Path], resume: bool) -> None:
    """Refuse to start over existing outputs, so failure cleanup can only remove this run's folders."""
    if resume:
        return
    existing = [str(d) for d in dirs if d.exists()]
    if existing:
        raise FileExistsError(
            "Output folders from an earlier run already exist: "
            + ", ".join(existing)
            + ". Delete or move them first, or pass --resume."
        )


def _output_dirs(checkpoint_dir: Path, gguf_dir: Path) -> list[Path]:
    return [checkpoint_dir, gguf_dir, Path(str(gguf_dir) + "_gguf")]


def _cleanup_dirs(checkpoint_dir: Path, gguf_dir: Path, logger: Any) -> None:
    """Remove partial checkpoint and GGUF dirs after a failed run."""
    import shutil
    for d in _output_dirs(checkpoint_dir, gguf_dir):
        if d.exists():
            shutil.rmtree(d)
            logger.warning(f"Cleaned up partial directory: {d}")


@dataclass
class RunPlan:
    ollama_base: str
    entry: dict
    training: dict
    ft_tag: str
    dataset_path: str
    gguf_dir: Path
    checkpoint_dir: Path
    size_b: float
    use_grad_ckpt: bool
    use_cpu_offload: bool


def plan_run(ollama_base: str, config_path: str) -> RunPlan:
    """Resolve the config entry, output paths and memory strategy for one model."""
    config = load_ft_config(config_path)
    entry = find_model_entry(config, ollama_base)
    paths = config["paths"]
    ft_tag = entry["ollama_ft_tag"]
    size_b = parse_model_info(ollama_base)["size_b"]
    return RunPlan(
        ollama_base=ollama_base,
        entry=entry,
        training=config["training"],
        ft_tag=ft_tag,
        dataset_path=config["training"]["dataset_path"],
        gguf_dir=Path(paths["gguf_output_dir"]).expanduser() / ft_tag,
        checkpoint_dir=Path(paths["checkpoints_dir"]).expanduser() / ft_tag,
        size_b=size_b,
        use_grad_ckpt=size_b >= GRAD_CKPT_SIZE_THRESHOLD_B,
        use_cpu_offload=size_b >= CPU_OFFLOAD_SIZE_THRESHOLD_B,
    )


def _log_plan(plan: RunPlan, logger: Any) -> None:
    logger.info(f"Model      : {plan.ollama_base} → {plan.entry['hf_id']}")
    logger.info(f"FT tag     : {plan.ft_tag}")
    logger.info(f"Size       : {plan.size_b}B params (MODEL_REGISTRY)")
    logger.info(f"Grad ckpt  : {plan.use_grad_ckpt}")
    logger.info(f"CPU offload: {plan.use_cpu_offload}")
    logger.info(f"Dataset    : {plan.dataset_path}")
    logger.info(f"GGUF out   : {plan.gguf_dir}")


def _load_split(plan: RunPlan, dry_run: bool, logger: Any) -> tuple[list[dict], list[dict]] | None:
    """Train/held-out records, or None when a dry run stops early."""
    if not Path(plan.dataset_path).exists():
        if dry_run:
            logger.info(f"[dry-run] Dataset not found at {plan.dataset_path} — generate it first.")
            logger.info("[dry-run] Config and model lookup validated successfully.")
            return None
        logger.error(f"Dataset not found: {plan.dataset_path}\nGenerate it with fine-tuning/scripts/generate_board_dataset.py")
        sys.exit(1)
    records = load_dataset_records(plan.dataset_path)
    logger.info(f"Loaded {len(records)} training records.")
    train_records, held_out = split_held_out(records, seed=plan.training.get("seed", 42))
    logger.info(f"Train: {len(train_records)} | Held-out: {len(held_out)}")
    if dry_run:
        logger.info("[dry-run] Dataset validated. No training will run.")
        logger.info(f"[dry-run] First record input: {records[0]['input']}")
        return None
    return train_records, held_out


def _import_training_deps(logger: Any) -> tuple[Any, Any, Any, Any]:
    """Deferred so dry runs work without GPU dependencies installed."""
    try:
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git' 'trl>=1.0' datasets")
        sys.exit(1)
    return FastLanguageModel, SFTTrainer, SFTConfig, Dataset


def _load_model(plan: RunPlan, fast_language_model: Any, logger: Any) -> tuple[Any, Any]:
    training = plan.training
    load_kwargs = build_load_kwargs(plan.entry, training, plan.use_grad_ckpt, plan.use_cpu_offload)
    logger.info(f"Loading {plan.entry['hf_id']} in 4-bit...")
    model, tokenizer = fast_language_model.from_pretrained(**load_kwargs)
    model = fast_language_model.get_peft_model(
        model,
        r=training["lora_rank"],
        lora_alpha=training["lora_alpha"],
        target_modules=training["lora_target_modules"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=plan.use_grad_ckpt,
        random_state=training.get("seed", 42),
    )
    return model, tokenizer


def _build_trainer(plan: RunPlan, deps: tuple, model: Any, tokenizer: Any,
                   train_records: list[dict], callback: Any) -> Any:
    _, sft_trainer, sft_config, dataset_cls = deps
    training = plan.training
    texts = [{"text": format_alpaca_prompt(r["instruction"], r["input"], r["output"])} for r in train_records]
    return sft_trainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset_cls.from_list(texts),
        args=sft_config(
            dataset_text_field="text",
            max_length=training["max_seq_length"],
            per_device_train_batch_size=training["per_device_train_batch_size"],
            gradient_accumulation_steps=training["gradient_accumulation_steps"],
            num_train_epochs=training["num_epochs"],
            learning_rate=training["learning_rate"],
            warmup_ratio=training["warmup_ratio"],
            bf16=True,
            logging_steps=TRAINER_LOGGING_STEPS,
            output_dir=str(plan.checkpoint_dir),
            seed=training.get("seed", 42),
            report_to="none",
        ),
        callbacks=[callback],
    )


def _export_gguf(plan: RunPlan, model: Any, tokenizer: Any, logger: Any) -> Path | None:
    plan.gguf_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exporting GGUF to {plan.gguf_dir}...")
    model.save_pretrained_gguf(
        str(plan.gguf_dir), tokenizer, quantization_method=plan.training.get("gguf_quantization", "q4_k_m"),
    )
    # Unsloth may append _gguf to the output dir name; check both locations.
    actual_dir = plan.gguf_dir if plan.gguf_dir.exists() else Path(str(plan.gguf_dir) + "_gguf")
    gguf_files = sorted(actual_dir.glob("*.gguf")) if actual_dir.exists() else []
    if not gguf_files:
        logger.warning(f"No .gguf file found in {plan.gguf_dir} or {actual_dir} — check Unsloth output above.")
        return None
    logger.info(f"GGUF saved: {gguf_files[0]}")
    return gguf_files[0]


def training_metrics(log_history: list[dict], global_step: int, epoch: float,
                     accuracy_history: list[dict]) -> dict[str, Any]:
    """Loss curve and held-out accuracy for the run record."""
    losses = [[entry["step"], entry["loss"]] for entry in log_history if "loss" in entry]
    return {
        "global_step": global_step,
        "epoch": epoch,
        "final_train_loss": losses[-1][1] if losses else None,
        "loss_history": losses,
        "held_out_accuracy": accuracy_history,
    }


def record_run(plan: RunPlan, config_path: str, metrics: dict, gguf_path: Path | None,
               runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
    record = build_run_record(
        model=plan.ollama_base, config_path=config_path, metrics=metrics,
        outputs={"checkpoint_dir": str(plan.checkpoint_dir), "gguf": str(gguf_path) if gguf_path else None},
    )
    return write_run_record(record, runs_dir)


def _train_and_export(plan: RunPlan, split: tuple[list[dict], list[dict]], resume: bool,
                      logger: Any) -> tuple[Any, Any, Path | None]:
    deps = _import_training_deps(logger)
    model, tokenizer = _load_model(plan, deps[0], logger)
    train_records, held_out = split
    callback = EpochAccuracyCallback(model, tokenizer, held_out, logger)
    plan.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer = _build_trainer(plan, deps, model, tokenizer, train_records, callback)
    logger.info("Starting training...")
    try:
        trainer.train(resume_from_checkpoint=True if resume else None)
        logger.info("Training complete.")
        gguf_path = _export_gguf(plan, model, tokenizer, logger)
    except Exception as exc:  # any failure: remove this run's partial outputs, then re-raise
        logger.error(f"Training or GGUF export failed: {exc}")
        _cleanup_dirs(plan.checkpoint_dir, plan.gguf_dir, logger)
        raise
    return trainer, callback, gguf_path


def run_finetuning(
    ollama_base: str,
    config_path: str,
    dry_run: bool,
    resume: bool,
    logger: Any,
) -> None:
    plan = plan_run(ollama_base, config_path)
    _log_plan(plan, logger)
    split = _load_split(plan, dry_run, logger)
    if split is None:
        return
    ensure_outputs_free(_output_dirs(plan.checkpoint_dir, plan.gguf_dir), resume)
    trainer, callback, gguf_path = _train_and_export(plan, split, resume, logger)
    metrics = training_metrics(trainer.state.log_history, trainer.state.global_step,
                               trainer.state.epoch, callback.history)
    logger.info(f"Run record: {record_run(plan, config_path, metrics, gguf_path)}")
    logger.info(f"Done. Run: python fine-tuning/scripts/import_finetuned_to_ollama.py {ollama_base}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a single model on chess board comprehension."
    )
    parser.add_argument("ollama_base", help="Ollama base model tag (e.g. llama3.2:3b)")
    parser.add_argument(
        "--config", default="fine-tuning/config/finetune_config.yaml",
        help="Fine-tuning config path (default: fine-tuning/config/finetune_config.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate dataset and config without running training",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from the latest checkpoint in the checkpoint dir",
    )
    args = parser.parse_args()

    logger = setup_logging("finetune_model")
    run_finetuning(args.ollama_base, args.config, args.dry_run, args.resume, logger)


if __name__ == "__main__":
    main()
