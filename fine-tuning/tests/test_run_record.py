"""F-7: one JSON run record per training run (fine-tuning/specs/chess-board-finetuning.md)."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
FT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FT_ROOT / "scripts"))

import run_record as rr  # noqa: E402

TAG = "qwen2.5:7b-board-ft"


def _config(tmp_path, dataset):
    cfg = {
        "training": {"seed": 42, "dataset_path": str(dataset), "lora_rank": 16},
        "paths": {"checkpoints_dir": str(tmp_path / "ckpt"), "gguf_output_dir": str(tmp_path / "gguf")},
        "models": [{"ollama_base": "qwen2.5:7b", "ollama_ft_tag": TAG, "hf_id": "unsloth/Qwen2.5-7B"}],
    }
    path = tmp_path / "ft.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _dataset(tmp_path, n=3):
    path = tmp_path / "board.jsonl"
    path.write_text("".join(json.dumps({"input": f"FEN: {i}"}) + "\n" for i in range(n)))
    return path


def _trainer_state(directory, steps=(50, 100), losses=(1.2, 0.4), epoch=1.0):
    directory.mkdir(parents=True)
    history = [{"step": s, "loss": l, "epoch": epoch * s / steps[-1]} for s, l in zip(steps, losses)]
    history.append({"step": steps[-1], "train_runtime": 99.0})
    (directory / "trainer_state.json").write_text(json.dumps(
        {"global_step": steps[-1], "epoch": epoch, "log_history": history}))


def test_build_record_fields(tmp_path):
    dataset = _dataset(tmp_path)
    record = rr.build_run_record(
        model="qwen2.5:7b", config_path=_config(tmp_path, dataset),
        metrics={"final_train_loss": 0.4}, outputs={"gguf": "x.gguf"}, repo_dir=tmp_path,
    )
    assert record["ft_tag"] == TAG
    assert record["hf_id"] == "unsloth/Qwen2.5-7B"
    assert record["seed"] == 42
    assert record["dataset_sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert record["dataset_records"] == 3
    assert record["config"]["training"]["lora_rank"] == 16
    assert record["git_sha"] is None  # tmp_path is not a repo
    assert record["run_id"].endswith("_qwen2.5_7b-board-ft")


def test_write_refuses_overwrite(tmp_path):
    record = {"run_id": "r1", "x": 1}
    path = rr.write_run_record(record, tmp_path / "runs")
    assert json.loads(path.read_text()) == record
    with pytest.raises(FileExistsError):
        rr.write_run_record(record, tmp_path / "runs")


def test_metrics_from_trainer_state(tmp_path):
    _trainer_state(tmp_path / "checkpoint-100")
    metrics = rr.metrics_from_trainer_state(tmp_path / "checkpoint-100" / "trainer_state.json")
    assert metrics["global_step"] == 100
    assert metrics["final_train_loss"] == pytest.approx(0.4)
    assert metrics["loss_history"] == [[50, 1.2], [100, 0.4]]


def test_latest_checkpoint_picked_by_step(tmp_path):
    run_dir = tmp_path / TAG
    _trainer_state(run_dir / "checkpoint-500", steps=(250, 500))
    _trainer_state(run_dir / "checkpoint-2476", steps=(1000, 2476))
    assert rr.latest_trainer_state(run_dir).parent.name == "checkpoint-2476"


def test_latest_checkpoint_missing_raises(tmp_path):
    (tmp_path / TAG).mkdir()
    with pytest.raises(FileNotFoundError):
        rr.latest_trainer_state(tmp_path / TAG)


def test_backfill_writes_one_record_per_model(tmp_path):
    dataset = _dataset(tmp_path)
    config = _config(tmp_path, dataset)
    _trainer_state(tmp_path / "ckpt" / TAG / "checkpoint-100")
    written = rr.backfill(config, runs_dir=tmp_path / "runs", repo_dir=tmp_path)
    assert len(written) == 1
    record = json.loads(written[0].read_text())
    assert record["metrics"]["global_step"] == 100
    assert "backfilled" in record["notes"]
    assert record["git_sha"] is None


def test_backfill_skips_models_without_checkpoints(tmp_path):
    config = _config(tmp_path, _dataset(tmp_path))
    assert rr.backfill(config, runs_dir=tmp_path / "runs", repo_dir=tmp_path) == []
