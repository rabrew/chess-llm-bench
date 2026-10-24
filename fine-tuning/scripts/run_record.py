"""Experiment tracking: one JSON record per training run in runs/ (F-7).

Spec: fine-tuning/specs/chess-board-finetuning.md, "Experiment Tracking".

Usage (from the repo root):
    python fine-tuning/scripts/run_record.py backfill
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.provenance import file_sha256, git_state, line_count  # noqa: E402
from src.utils import setup_logging  # noqa: E402

logger = logging.getLogger("chess_llm_bench")

DEFAULT_CONFIG = "fine-tuning/config/finetune_config.yaml"
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"
CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")
RUN_ID_TIME_FORMAT = "%Y%m%dT%H%M%SZ"
BACKFILL_NOTE = (
    "backfilled 2026-10-24 from trainer_state.json; git SHA at training time unknown; "
    "dataset hash is of the current file, assumed unchanged since training"
)


def _load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _model_entry(config: dict, model: str) -> dict:
    for entry in config.get("models", []):
        if entry["ollama_base"] == model:
            return entry
    raise ValueError(f"{model!r} not in fine-tuning config")


def _run_id(created_at: datetime, ft_tag: str) -> str:
    return f"{created_at.strftime(RUN_ID_TIME_FORMAT)}_{ft_tag.replace(':', '_')}"


def build_run_record(*, model: str, config_path: str | Path, metrics: dict, outputs: dict,
                     repo_dir: str | Path = REPO_ROOT, created_at: datetime | None = None,
                     notes: str = "", git: tuple[str | None, bool | None] | None = None) -> dict[str, Any]:
    """Everything needed to identify and reproduce one training run."""
    config = _load_config(config_path)
    entry = _model_entry(config, model)
    created_at = created_at or datetime.now(timezone.utc)
    dataset = Path(config["training"]["dataset_path"])
    sha, dirty = git if git is not None else git_state(Path(repo_dir))
    return {
        "run_id": _run_id(created_at, entry["ollama_ft_tag"]),
        "created_at": created_at.isoformat(),
        "model": model,
        "ft_tag": entry["ollama_ft_tag"],
        "hf_id": entry["hf_id"],
        "git_sha": sha,
        "git_dirty": dirty,
        "seed": config["training"].get("seed"),
        "config_path": str(config_path),
        "config": config,
        "dataset_path": str(dataset),
        "dataset_sha256": file_sha256(dataset),
        "dataset_records": line_count(dataset),
        "metrics": metrics,
        "outputs": outputs,
        "notes": notes,
    }


def write_run_record(record: dict, runs_dir: str | Path = DEFAULT_RUNS_DIR) -> Path:
    """Write runs/<run_id>.json; never overwrite an existing record."""
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{record['run_id']}.json"
    if path.exists():
        raise FileExistsError(f"Run record already exists: {path}")
    path.write_text(json.dumps(record, indent=2, default=str))
    return path


def metrics_from_trainer_state(path: str | Path) -> dict[str, Any]:
    """Step count, epoch and loss history from a HuggingFace trainer_state.json."""
    state = json.loads(Path(path).read_text())
    losses = [[entry["step"], entry["loss"]] for entry in state.get("log_history", []) if "loss" in entry]
    return {
        "global_step": state.get("global_step"),
        "epoch": state.get("epoch"),
        "final_train_loss": losses[-1][1] if losses else None,
        "loss_history": losses,
    }


def latest_trainer_state(run_dir: Path) -> Path:
    """trainer_state.json of the highest-numbered checkpoint in run_dir."""
    checkpoints = [
        (int(match.group(1)), child)
        for child in run_dir.iterdir()
        if (match := CHECKPOINT_PATTERN.search(child.name)) and (child / "trainer_state.json").exists()
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint with trainer_state.json in {run_dir}")
    return max(checkpoints)[1] / "trainer_state.json"


def backfill(config_path: str | Path, runs_dir: str | Path = DEFAULT_RUNS_DIR,
             repo_dir: str | Path = REPO_ROOT) -> list[Path]:
    """Create run records for already-trained models from their checkpoints."""
    config = _load_config(config_path)
    checkpoints_dir = Path(config["paths"]["checkpoints_dir"]).expanduser()
    written = []
    for entry in config.get("models", []):
        run_dir = checkpoints_dir / entry["ollama_ft_tag"]
        if not run_dir.is_dir():
            continue
        state_path = latest_trainer_state(run_dir)
        created_at = datetime.fromtimestamp(state_path.stat().st_mtime, tz=timezone.utc)
        record = build_run_record(
            model=entry["ollama_base"], config_path=config_path,
            metrics=metrics_from_trainer_state(state_path),
            outputs={"checkpoint_dir": str(run_dir), "trainer_state": str(state_path)},
            repo_dir=repo_dir, created_at=created_at, notes=BACKFILL_NOTE, git=(None, None),
        )
        written.append(write_run_record(record, runs_dir))
        logger.info(f"Backfilled {written[-1]}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Training run records (F-7).")
    sub = parser.add_subparsers(dest="command", required=True)
    back = sub.add_parser("backfill", help="Create records for models already trained")
    back.add_argument("--config", default=DEFAULT_CONFIG)
    back.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    args = parser.parse_args()
    setup_logging()
    if args.command == "backfill":
        written = backfill(args.config, runs_dir=args.runs_dir)
        logger.info(f"{len(written)} run records written")


if __name__ == "__main__":
    main()
