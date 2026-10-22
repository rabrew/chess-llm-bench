"""GGUF export check and manual re-export (specs/gguf-export-check.md)."""

import subprocess
import sys
from pathlib import Path

import pytest

FT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FT_ROOT / "scripts"))
sys.path.insert(0, str(FT_ROOT / "src"))

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
pytest.importorskip("transformers")
import reexport_gguf as rx  # noqa: E402
import verify_export as vx  # noqa: E402


# --- verify_export ---------------------------------------------------------

def test_export_passes_at_and_above_threshold():
    assert vx.export_passes([0.8, 0.8], 0.8) is True
    assert vx.export_passes([1.0, 0.9, 0.95], 0.8) is True


def test_export_fails_below_threshold():
    # the broken gemma4 export: mostly unparsed, recall near zero
    assert vx.export_passes([0.0, 0.0, 0.8, 0.0], 0.8) is False


def test_export_passes_rejects_empty():
    with pytest.raises(ValueError):
        vx.export_passes([], 0.8)


def test_threshold_sits_between_broken_and_working_models():
    assert 0.149 < vx.MIN_EXPORT_RECALL < 0.965


def test_sample_positions_caps_at_held_out_size():
    held = [{"input": f"FEN: {i}"} for i in range(5)]
    assert vx.sample_positions(held, 3) == held[:3]
    assert vx.sample_positions(held, 50) == held


def test_sample_positions_rejects_non_positive():
    with pytest.raises(ValueError):
        vx.sample_positions([{"input": "x"}], 0)


# --- reexport_gguf: paths and ids ------------------------------------------

@pytest.mark.parametrize("given,expected", [
    ("unsloth/gemma-4-E2B-it-unsloth-bnb-4bit", "unsloth/gemma-4-E2B-it"),
    ("unsloth/Qwen2.5-7B-Instruct-bnb-4bit", "unsloth/Qwen2.5-7B-Instruct"),
    ("google/gemma-4-E2B-it", "google/gemma-4-E2B-it"),
])
def test_full_precision_base_id(given, expected):
    assert rx.full_precision_base_id(given) == expected


def test_latest_checkpoint_sorts_numerically(tmp_path):
    for step in (500, 1000, 2476):
        (tmp_path / f"checkpoint-{step}").mkdir()
    (tmp_path / "README.md").write_text("x")
    assert rx.latest_checkpoint(tmp_path).name == "checkpoint-2476"


def test_latest_checkpoint_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        rx.latest_checkpoint(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        rx.latest_checkpoint(tmp_path)


def test_require_adapter_config(tmp_path):
    with pytest.raises(FileNotFoundError):
        rx.require_adapter_config(tmp_path)
    (tmp_path / "adapter_config.json").write_text("{}")
    rx.require_adapter_config(tmp_path)


def test_work_dir_must_be_free(tmp_path):
    rx.ensure_work_dir_free(tmp_path / "new")
    with pytest.raises(FileExistsError):
        rx.ensure_work_dir_free(tmp_path)


def test_convert_and_quantize_commands(tmp_path):
    convert = rx.convert_command(tmp_path / "llama", tmp_path / "merged", tmp_path / "m.BF16.gguf")
    assert convert[1].endswith("convert_hf_to_gguf.py")
    assert convert[-4:] == ["--outtype", "bf16", "--outfile", str(tmp_path / "m.BF16.gguf")]
    quant = rx.quantize_command(tmp_path / "llama", tmp_path / "a.gguf", tmp_path / "b.gguf", "q4_k_m")
    assert quant == [str(tmp_path / "llama" / "llama-quantize"), str(tmp_path / "a.gguf"), str(tmp_path / "b.gguf"), "Q4_K_M"]


def test_run_step_raises_with_stderr():
    def failing(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom: bad tensor")
    with pytest.raises(RuntimeError, match="boom: bad tensor"):
        rx.run_step(["convert"], runner=failing)


def test_run_step_ok():
    def ok(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="done", stderr="")
    rx.run_step(["convert"], runner=ok)


# --- reexport_gguf: adapter and merge checks -------------------------------

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.q_proj(x)


def _lora(trained: bool):
    torch.manual_seed(0)
    model = peft.get_peft_model(Tiny(), peft.LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"]))
    if trained:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_B" in name:
                    param.fill_(0.5)
    return model


def test_adapter_check_accepts_trained_adapter():
    assert rx.check_adapter_attached(_lora(trained=True)) == 1


def test_adapter_check_rejects_untrained_adapter():
    with pytest.raises(RuntimeError, match="lora_B"):
        rx.check_adapter_attached(_lora(trained=False))


def test_adapter_check_rejects_no_lora_layers():
    with pytest.raises(RuntimeError, match="no LoRA"):
        rx.check_adapter_attached(Tiny())


def test_merge_changes_weights_for_trained_adapter():
    merged = rx.merge_and_check(_lora(trained=True))
    assert isinstance(merged, Tiny)


def test_merge_check_catches_unchanged_weights():
    before = torch.ones(2, 2)
    with pytest.raises(RuntimeError, match="unchanged"):
        rx.check_weight_changed(before, before.clone())
