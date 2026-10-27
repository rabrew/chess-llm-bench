"""finetune_model.py run steps: plan, callback history, metrics, run record (F-7)."""

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
FT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FT_ROOT / "src"))
sys.path.insert(0, str(FT_ROOT / "scripts"))

pytest.importorskip("transformers")

import finetune_model as fm  # noqa: E402

MODELS = [
    {"ollama_base": "qwen2.5:7b", "hf_id": "Qwen/Qwen2.5-7B-Instruct", "ollama_ft_tag": "qwen2.5:7b-board-ft"},
    {"ollama_base": "qwen2.5:32b", "hf_id": "Qwen/Qwen2.5-32B-Instruct", "ollama_ft_tag": "qwen2.5:32b-board-ft"},
    {"ollama_base": "llama3.3:70b", "hf_id": "meta-llama/Llama-3.3-70B-Instruct", "ollama_ft_tag": "llama3.3:70b-board-ft"},
    {"ollama_base": "mixtral:8x7b", "hf_id": "mistralai/Mixtral-8x7B-Instruct-v0.1", "ollama_ft_tag": "mixtral:8x7b-board-ft"},
]


def _config(tmp_path):
    dataset = tmp_path / "board.jsonl"
    dataset.write_text(json.dumps({"instruction": "i", "input": "FEN: x", "output": "o"}) + "\n")
    cfg = {
        "models": MODELS,
        "training": {"seed": 42, "dataset_path": str(dataset)},
        "paths": {"gguf_output_dir": str(tmp_path / "gguf"), "checkpoints_dir": str(tmp_path / "ckpt")},
    }
    path = tmp_path / "ft.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.mark.parametrize("model, grad_ckpt, offload", [
    ("qwen2.5:7b", False, False),
    ("qwen2.5:32b", True, False),
    ("mixtral:8x7b", True, False),
    ("llama3.3:70b", True, True),
])
def test_plan_uses_registry_sizes(tmp_path, model, grad_ckpt, offload):
    plan = fm.plan_run(model, str(_config(tmp_path)))
    assert plan.use_grad_ckpt is grad_ckpt
    assert plan.use_cpu_offload is offload


def test_plan_paths_and_tag(tmp_path):
    plan = fm.plan_run("qwen2.5:7b", str(_config(tmp_path)))
    assert plan.ft_tag == "qwen2.5:7b-board-ft"
    assert plan.gguf_dir == tmp_path / "gguf" / "qwen2.5:7b-board-ft"
    assert plan.checkpoint_dir == tmp_path / "ckpt" / "qwen2.5:7b-board-ft"


class _Logger:
    def __init__(self):
        self.errors, self.infos = [], []

    def info(self, msg):
        self.infos.append(msg)

    def error(self, msg, exc_info=False):
        self.errors.append(msg)

    def warning(self, msg):
        self.infos.append(msg)


class _State:
    epoch = 2.0


def test_callback_records_history(monkeypatch):
    monkeypatch.setattr(fm, "evaluate_held_out", lambda *args: 0.75)
    callback = fm.EpochAccuracyCallback(None, None, [], _Logger())
    callback.on_epoch_end(None, _State(), None)
    assert callback.history == [{"epoch": 2.0, "held_out_accuracy": 0.75}]


def test_callback_failure_adds_no_history(monkeypatch):
    def boom(*args):
        raise RuntimeError("generate exploded")
    monkeypatch.setattr(fm, "evaluate_held_out", boom)
    callback = fm.EpochAccuracyCallback(None, None, [], _Logger())
    callback.on_epoch_end(None, _State(), None)
    assert callback.history == []


def test_training_metrics_merges_losses_and_accuracy():
    log_history = [{"step": 50, "loss": 1.0}, {"step": 100, "loss": 0.5}, {"train_runtime": 3.0}]
    metrics = fm.training_metrics(log_history, global_step=100, epoch=1.0,
                                  accuracy_history=[{"epoch": 1.0, "held_out_accuracy": 0.9}])
    assert metrics["final_train_loss"] == 0.5
    assert metrics["loss_history"] == [[50, 1.0], [100, 0.5]]
    assert metrics["held_out_accuracy"] == [{"epoch": 1.0, "held_out_accuracy": 0.9}]


def test_record_run_writes_runs_file(tmp_path):
    config_path = _config(tmp_path)
    plan = fm.plan_run("qwen2.5:7b", str(config_path))
    path = fm.record_run(plan, str(config_path), {"final_train_loss": 0.5}, tmp_path / "x.gguf",
                         runs_dir=tmp_path / "runs")
    record = json.loads(path.read_text())
    assert record["ft_tag"] == "qwen2.5:7b-board-ft"
    assert record["outputs"]["gguf"] == str(tmp_path / "x.gguf")


def test_unsloth_is_imported_before_trl(monkeypatch):
    """Unsloth patches TRL on import; importing trl first breaks SFTConfig (warmup_ratio)."""
    import builtins
    import types
    order = []
    real_import = builtins.__import__
    fakes = {
        "unsloth": types.SimpleNamespace(FastLanguageModel=object),
        "trl": types.SimpleNamespace(SFTConfig=object, SFTTrainer=object),
        "datasets": types.SimpleNamespace(Dataset=object),
    }

    def recording_import(name, *args, **kwargs):
        if name in fakes:
            order.append(name)
            return fakes[name]
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", recording_import)
    fm._import_training_deps(_Logger())
    assert order[0] == "unsloth"


def test_trainer_build_failure_cleans_up(tmp_path, monkeypatch):
    plan = fm.plan_run("qwen2.5:7b", str(_config(tmp_path)))
    fake_model = type("M", (), {})()
    monkeypatch.setattr(fm, "_import_training_deps", lambda logger: (None, None, None, None))
    monkeypatch.setattr(fm, "_load_model", lambda plan, flm, logger: (fake_model, None))

    def broken_build(*args, **kwargs):
        plan.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        raise TypeError("unexpected keyword argument 'warmup_ratio'")

    monkeypatch.setattr(fm, "_build_trainer", broken_build)
    with pytest.raises(TypeError):
        fm._train_and_export(plan, ([], []), resume=False, logger=_Logger())
    assert not plan.checkpoint_dir.exists()


class _Trainer:
    def __init__(self, plan, fail=False):
        self.plan, self.fail = plan, fail

    def train(self, resume_from_checkpoint=None):
        (self.plan.checkpoint_dir / "checkpoint-10").mkdir(parents=True)
        if self.fail:
            raise RuntimeError("CUDA out of memory")


def _fake_training(monkeypatch, plan, fail_training=False):
    monkeypatch.setattr(fm, "_import_training_deps", lambda logger: (None, None, None, None))
    monkeypatch.setattr(fm, "_load_model", lambda plan, flm, logger: (object(), None))
    monkeypatch.setattr(fm, "_build_trainer", lambda *args: _Trainer(plan, fail_training))


def test_training_failure_cleans_up(tmp_path, monkeypatch):
    plan = fm.plan_run("qwen2.5:7b", str(_config(tmp_path)))
    _fake_training(monkeypatch, plan, fail_training=True)
    with pytest.raises(RuntimeError):
        fm._train_and_export(plan, ([], []), resume=False, logger=_Logger())
    assert not plan.checkpoint_dir.exists()


def test_export_failure_keeps_checkpoint(tmp_path, monkeypatch):
    plan = fm.plan_run("qwen2.5:7b", str(_config(tmp_path)))
    _fake_training(monkeypatch, plan)

    def full_disk(plan, model, tokenizer, logger):
        plan.gguf_dir.mkdir(parents=True)
        raise RuntimeError("Unsloth: Not enough disk space to convert to GGUF.")

    monkeypatch.setattr(fm, "_export_gguf", full_disk)
    logger = _Logger()
    with pytest.raises(RuntimeError, match="disk space"):
        fm._train_and_export(plan, ([], []), resume=False, logger=logger)
    assert (plan.checkpoint_dir / "checkpoint-10").is_dir()
    assert not plan.gguf_dir.exists()
    assert any("reexport_gguf.py qwen2.5:7b" in msg for msg in logger.errors)


class _GgufModel:
    """Writes files where Unsloth does: the 16-bit merge in gguf_dir, the GGUF in gguf_dir or gguf_dir_gguf."""

    def __init__(self, gguf_in_suffix_dir):
        self.gguf_in_suffix_dir = gguf_in_suffix_dir

    def save_pretrained_gguf(self, out_dir, tokenizer, quantization_method):
        merge_dir = Path(out_dir)
        merge_dir.mkdir(parents=True, exist_ok=True)
        (merge_dir / "model-00001-of-00002.safetensors").write_text("weights")
        gguf_dir = Path(out_dir + "_gguf") if self.gguf_in_suffix_dir else merge_dir
        gguf_dir.mkdir(parents=True, exist_ok=True)
        (gguf_dir / "Model.Q4_K_M.gguf").write_text("gguf")


def test_export_removes_16bit_merge(tmp_path):
    plan = fm.plan_run("qwen2.5:7b", str(_config(tmp_path)))
    gguf = fm._export_gguf(plan, _GgufModel(gguf_in_suffix_dir=True), None, _Logger())
    assert gguf == Path(str(plan.gguf_dir) + "_gguf") / "Model.Q4_K_M.gguf"
    assert gguf.is_file()
    assert not plan.gguf_dir.exists()


def test_export_keeps_dir_holding_the_gguf(tmp_path):
    plan = fm.plan_run("qwen2.5:7b", str(_config(tmp_path)))
    gguf = fm._export_gguf(plan, _GgufModel(gguf_in_suffix_dir=False), None, _Logger())
    assert gguf == plan.gguf_dir / "Model.Q4_K_M.gguf"
    assert gguf.is_file()
