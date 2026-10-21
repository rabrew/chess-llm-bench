"""Tests for fine-tuning/specs/gemma-training-fixes.md."""

import re
import sys
from pathlib import Path

import pytest
import yaml

FT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FT_ROOT / "scripts"))

pytest.importorskip("transformers")
from finetune_model import build_load_kwargs  # noqa: E402

CONFIG_PATH = FT_ROOT / "config" / "finetune_config.yaml"
HF_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
VERIFIED_GEMMA4_IDS = {
    "gemma4:e2b": "google/gemma-4-E2B-it",
    "gemma4:e4b": "google/gemma-4-E4B-it",
    "gemma4:26b": "google/gemma-4-26B-A4B-it",
    "gemma4:31b": "google/gemma-4-31B-it",
}
TRAINING = {"max_seq_length": 512}


@pytest.fixture(scope="module")
def entries():
    return yaml.safe_load(CONFIG_PATH.read_text())["models"]


def test_no_unverified_hf_ids():
    assert "VERIFY" not in CONFIG_PATH.read_text()


def test_hf_ids_are_org_slash_name(entries):
    for entry in entries:
        assert HF_ID_PATTERN.match(entry["hf_id"]), entry


def test_gemma4_ids_match_huggingface(entries):
    by_tag = {entry["ollama_base"]: entry["hf_id"] for entry in entries}
    for tag, hf_id in VERIFIED_GEMMA4_IDS.items():
        assert by_tag[tag] == hf_id


def test_every_gemma_entry_uses_eager_attention(entries):
    gemma = [entry for entry in entries if entry["ollama_base"].startswith("gemma")]
    assert gemma
    for entry in gemma:
        assert entry.get("attn_implementation") == "eager", entry


def test_non_gemma_load_kwargs_unchanged():
    entry = {"ollama_base": "qwen2.5:7b", "hf_id": "Qwen/Qwen2.5-7B-Instruct"}
    assert build_load_kwargs(entry, TRAINING, use_grad_ckpt=False, use_cpu_offload=False) == {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "max_seq_length": 512,
        "dtype": None,
        "load_in_4bit": True,
    }


def test_attn_implementation_passed_through():
    entry = {"ollama_base": "gemma3:4b", "hf_id": "google/gemma-3-4b-it", "attn_implementation": "eager"}
    kwargs = build_load_kwargs(entry, TRAINING, use_grad_ckpt=True, use_cpu_offload=True)
    assert kwargs["attn_implementation"] == "eager"
    assert kwargs["use_gradient_checkpointing"] is True
    assert kwargs["device_map"] == "auto"


def test_bad_attn_implementation_rejected():
    entry = {"ollama_base": "gemma3:4b", "hf_id": "google/gemma-3-4b-it", "attn_implementation": "flex"}
    with pytest.raises(ValueError, match="gemma3:4b"):
        build_load_kwargs(entry, TRAINING, use_grad_ckpt=False, use_cpu_offload=False)


# --- Part 2: import file choice, output protection, processor input, resilient callback ---

from finetune_model import EpochAccuracyCallback, ensure_outputs_free, evaluate_held_out  # noqa: E402

sys.path.insert(0, str(FT_ROOT / "scripts"))
from import_finetuned_to_ollama import find_model_gguf  # noqa: E402


def _touch(folder, *names):
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"x")


def test_mmproj_never_chosen(tmp_path):
    _touch(tmp_path / "m_gguf", "gemma-3-4b-it.BF16-mmproj.gguf", "gemma-3-4b-it.Q4_K_M.gguf")
    chosen = find_model_gguf([tmp_path / "m_gguf"], "q4_k_m")
    assert chosen.name == "gemma-3-4b-it.Q4_K_M.gguf"


def test_quantization_match_wins(tmp_path):
    _touch(tmp_path / "m_gguf", "model.BF16.gguf", "model.Q4_K_M.gguf")
    assert find_model_gguf([tmp_path / "m_gguf"], "q4_k_m").name == "model.Q4_K_M.gguf"


def test_single_model_file_used(tmp_path):
    _touch(tmp_path / "m_gguf", "unsloth.Q4_K_M.gguf")
    assert find_model_gguf([tmp_path / "missing", tmp_path / "m_gguf"], "q4_k_m").name == "unsloth.Q4_K_M.gguf"


def test_only_mmproj_raises(tmp_path):
    _touch(tmp_path / "m_gguf", "x-mmproj.gguf")
    with pytest.raises(FileNotFoundError):
        find_model_gguf([tmp_path / "m_gguf"], "q4_k_m")


def test_ambiguous_files_raise(tmp_path):
    _touch(tmp_path / "m_gguf", "a.BF16.gguf", "b.F16.gguf")
    with pytest.raises(FileNotFoundError):
        find_model_gguf([tmp_path / "m_gguf"], "q4_k_m")


def test_existing_outputs_block_new_run(tmp_path):
    existing = tmp_path / "gemma3:4b-board-ft"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="gemma3:4b-board-ft"):
        ensure_outputs_free([existing, tmp_path / "other"], resume=False)
    assert existing.exists()


def test_existing_outputs_allowed_with_resume(tmp_path):
    existing = tmp_path / "ckpt"
    existing.mkdir()
    ensure_outputs_free([existing], resume=True)


def test_no_outputs_is_fine(tmp_path):
    ensure_outputs_free([tmp_path / "a", tmp_path / "b"], resume=False)


class _FakeProcessor:
    """Mimics a multimodal processor: positional input is treated as images."""

    eos_token_id = 0

    def __call__(self, images=None, text=None, return_tensors=None):
        if images is not None:
            raise ValueError("Incorrect image source")
        import torch
        return _FakeBatch({"input_ids": torch.tensor([[1, 2, 3]])})

    def decode(self, ids, **kwargs):
        return "White: King e1\nBlack: King e8\nSide to move: White"


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeModel:
    device = "cpu"

    def generate(self, input_ids, **kwargs):
        import torch
        return torch.cat([input_ids, torch.tensor([[4, 5]])], dim=1)


HELD_OUT = [{"instruction": "List pieces.", "input": "FEN: 4k3/8/8/8/8/8/8/4K3 w - - 0 1"}]


def test_evaluate_held_out_passes_text_keyword():
    pytest.importorskip("torch")
    assert evaluate_held_out(_FakeModel(), _FakeProcessor(), HELD_OUT) == 1.0


class _RecordingLogger:
    def __init__(self):
        self.errors = []
        self.infos = []

    def info(self, msg):
        self.infos.append(msg)

    def error(self, msg, exc_info=False):
        self.errors.append(msg)


class _State:
    epoch = 1.0


def test_callback_survives_evaluation_failure():
    class _Broken:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("generate exploded")

    logger = _RecordingLogger()
    callback = EpochAccuracyCallback(_FakeModel(), _Broken(), HELD_OUT, logger)
    callback.on_epoch_end(None, _State(), None)
    assert logger.errors and "generate exploded" in logger.errors[0]
