"""Regression tests for experiment comparison and artifact preservation."""
import importlib
import sys
import types
from unittest.mock import MagicMock
import chess
import pytest
from src.board_state import fen_to_board_description, verify_board_description
from scripts import compare_finetuning_results as comparison

FEN = chess.STARTING_FEN
BASE = "llama3.2:3b"
FT = BASE + "-board-ft"

def observation(model=BASE, position=1, prompt="fen_only", **values):
    return {"model": model, "position_id": position, "fen": FEN,
            "prompt_format": prompt, "t1_direction_correct": True,
            "t2_legal": True, "t3_score": 1, **values}

def test_real_t1_schema_and_task_denominators():
    records = [observation(), observation(position=2, prompt="eval_only", t2_legal=None),
               observation(position=3, prompt="explanation_only", t2_legal=None)]
    metrics = comparison.aggregate_model_metrics(records)[BASE]
    assert metrics["t1_acc"] == 1
    assert metrics["t2_legal_rate"] == 1
    assert metrics["t1_n"] == 2
    assert metrics["t2_n"] == 1

def test_missing_move_counts_as_failure():
    metrics = comparison.aggregate_model_metrics([observation(t2_legal=None)])[BASE]
    assert metrics["t2_legal_rate"] == 0

def test_missing_t1_stays_missing():
    metrics = comparison.aggregate_model_metrics([observation(t1_direction_correct=None)])[BASE]
    assert metrics["t1_acc"] is None

def test_missing_t0_is_not_zero():
    rows = comparison.compare_matched_records([observation()], [observation(model=FT)])
    assert rows[0]["ft_t0_mean"] is None
    assert rows[0]["delta_t0"] is None

def test_comparison_matches_positions_and_reports_coverage():
    base = [observation(position=1), observation(position=2, t1_direction_correct=False)]
    tuned = [observation(model=FT, position=1)]
    row = comparison.compare_matched_records(base, tuned)[0]
    assert row["delta_t1_acc"] == 0
    assert row["matched_n"] == 1
    assert row["base_unmatched_n"] == 1
    assert row["ft_unmatched_n"] == 0

def test_duplicate_keys_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        comparison.compare_matched_records([observation(), observation()], [observation(model=FT)])

def test_different_trials_are_not_matched():
    assert comparison.compare_matched_records(
        [observation(attempt=1)], [observation(model=FT, attempt=2)]) == []

def test_nonstandard_jobs_are_not_matched():
    assert comparison.compare_matched_records(
        [observation(job_type="correction")], [observation(model=FT)]) == []

def test_conflicting_fens_are_rejected():
    with pytest.raises(ValueError, match="FEN"):
        comparison.compare_matched_records([observation()], [observation(model=FT, fen="other")])

def test_endpoint_cpl_can_exceed_1000():
    rec = observation(t2_cpl=1800, t2_reference_eval_before=900,
                      t2_reference_eval_after=-900, t2_reference_white_to_move=True)
    assert comparison.aggregate_model_metrics([rec])[BASE]["mean_cpl"] == 1800

def test_legacy_cpl_without_endpoints_is_unavailable():
    assert comparison.aggregate_model_metrics([observation(t2_cpl=1800)])[BASE]["mean_cpl"] is None

def test_hallucinations_reduce_f1_but_legacy_recall_is_explicit():
    desc = fen_to_board_description(FEN).replace("White: King e1", "White: King e1, Queen a5")
    verified = verify_board_description(FEN, desc)
    assert verified["accuracy"] == verified["recall"] == 1
    assert verified["precision"] < 1
    assert verified["f1"] < 1
    assert verified["correct"] is False

def test_t0_requires_dedicated_response():
    rec = observation(response="Eval: 0")
    assert comparison.compute_board_accuracy(rec) is None
    assert comparison.compute_board_accuracy(observation(t0_response=fen_to_board_description(FEN))) == 1

def test_failure_cleanup_preserves_existing_artifacts(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(TrainerCallback=object))
    module = importlib.import_module("scripts.finetune_model")
    checkpoint = tmp_path / "checkpoint"
    export = tmp_path / "export"
    for folder in (checkpoint, export, tmp_path / "export_gguf"):
        folder.mkdir()
        (folder / "saved").write_text("valuable")
    module._cleanup_dirs(checkpoint, export, MagicMock())
    assert all((folder / "saved").read_text() == "valuable"
               for folder in (checkpoint, export, tmp_path / "export_gguf"))

@pytest.mark.parametrize("failure_stage", ["train", "export"])
def test_failed_training_or_export_preserves_checkpoint(tmp_path, monkeypatch, failure_stage):
    module = importlib.import_module("scripts.finetune_model")
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"instruction":"pieces","input":"FEN: ' + FEN + '","output":"pieces"}\n')
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / FT
    checkpoint.mkdir(parents=True)
    saved = checkpoint / "checkpoint-1"
    saved.mkdir()
    (saved / "weights").write_text("valuable")
    config = {
        "models": [{"ollama_base": BASE, "hf_id": "test/model", "ollama_ft_tag": FT}],
        "paths": {"checkpoints_dir": str(checkpoint_root), "gguf_output_dir": str(tmp_path / "gguf")},
        "training": {"dataset_path": str(dataset), "max_seq_length": 32,
                     "lora_rank": 1, "lora_alpha": 1, "lora_target_modules": [],
                     "per_device_train_batch_size": 1, "gradient_accumulation_steps": 1,
                     "num_epochs": 1, "learning_rate": 0.01, "warmup_ratio": 0.1},
    }
    monkeypatch.setattr(module, "load_ft_config", lambda _: config)
    model = MagicMock()
    trainer = MagicMock()
    if failure_stage == "train":
        trainer.train.side_effect = RuntimeError("mock train failure")
    else:
        model.save_pretrained_gguf.side_effect = RuntimeError("mock export failure")
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(TrainerCallback=object))
    fast_model = types.SimpleNamespace(
        from_pretrained=lambda **kwargs: (model, MagicMock()),
        get_peft_model=lambda original, **kwargs: original,
    )
    monkeypatch.setitem(sys.modules, "unsloth", types.SimpleNamespace(FastLanguageModel=fast_model))
    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(
        SFTTrainer=lambda **kwargs: trainer, SFTConfig=lambda **kwargs: kwargs))
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(
        Dataset=types.SimpleNamespace(from_list=lambda rows: rows)))
    with pytest.raises(RuntimeError, match="mock"):
        module.run_finetuning(BASE, "ignored", False, True, MagicMock())
    assert (saved / "weights").read_text() == "valuable"
    trainer.train.assert_called_once_with(resume_from_checkpoint=True)

def test_metric_delta_uses_paired_availability():
    base = [observation(position=1), observation(position=2, t1_direction_correct=False)]
    tuned = [observation(model=FT, position=1), observation(model=FT, position=2, t1_direction_correct=None)]
    row = comparison.compare_matched_records(base, tuned)[0]
    assert row["base_t1_acc"] == row["ft_t1_acc"] == 1
    assert row["t1_paired_n"] == 1

def test_held_out_persists_dedicated_t0_and_exactness(tmp_path, monkeypatch):
    import contextlib
    import json
    module = importlib.import_module("scripts.finetune_model")
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(no_grad=contextlib.nullcontext))
    class Inputs(dict):
        def to(self, device):
            return self
    tokenizer = MagicMock()
    tokenizer.return_value = Inputs(input_ids=types.SimpleNamespace(shape=(1, 2)))
    tokenizer.decode.return_value = fen_to_board_description(FEN)
    model = MagicMock()
    model.generate.return_value = [[1, 2, 3]]
    output = tmp_path / "held-out.jsonl"
    score = module.evaluate_held_out(
        model, tokenizer, [{"input": "FEN: " + FEN, "instruction": "pieces"}], output)
    assert score == 1
    record = json.loads(output.read_text())
    assert record["t0_exact"] is True
    assert record["t0_f1"] == record["t0_recall"] == record["t0_precision"] == 1
    assert record["t0_response"] == fen_to_board_description(FEN)
    with pytest.raises(FileExistsError):
        module.evaluate_held_out(model, tokenizer, [], output)

def test_exclusion_file_missing_is_an_error(tmp_path):
    from scripts.generate_board_dataset import load_benchmark_fens
    with pytest.raises(FileNotFoundError):
        load_benchmark_fens(str(tmp_path / "absent.jsonl"))

def test_malformed_exclusion_file_is_an_error(tmp_path):
    from scripts.generate_board_dataset import load_benchmark_fens
    source = tmp_path / "invalid.jsonl"
    source.write_text("not json\n")
    with pytest.raises(ValueError, match="line 1"):
        load_benchmark_fens(str(source))

def test_ambiguous_import_does_not_invoke_ollama(tmp_path, monkeypatch):
    from scripts import import_finetuned_to_ollama as importer
    target = tmp_path / FT
    target.mkdir()
    (target / "one.gguf").write_text("fixture")
    (target / "two.gguf").write_text("fixture")
    config = {"models": [{"ollama_base": BASE, "ollama_ft_tag": FT}],
              "paths": {"gguf_output_dir": str(tmp_path), "ollama_modelfiles_dir": str(tmp_path / "modelfiles")}}
    monkeypatch.setattr(importer, "load_ft_config", lambda _: config)
    invoke = MagicMock()
    monkeypatch.setattr(importer.subprocess, "run", invoke)
    with pytest.raises(ValueError, match="Multiple"):
        importer.run(BASE, "ignored", MagicMock())
    invoke.assert_not_called()

@pytest.mark.parametrize("metadata", [{}, {"t2_reference_engine": "other"}])
def test_cpl_delta_requires_matching_recorded_provenance(metadata):
    reference = {"t2_reference_eval_before": 900, "t2_reference_eval_after": -900,
                 "t2_reference_white_to_move": True, "t2_reference_engine": "lc0",
                 "t2_reference_budget": {"nodes": 800}, "t2_reference_units": "centipawns_white",
                 "t2_scoring_version": "endpoints-v1"}
    base = observation(**reference)
    tuned = observation(model=FT, **({**reference, **metadata} if metadata else {
        key: value for key, value in reference.items() if key not in {
            "t2_reference_engine", "t2_reference_budget", "t2_reference_units", "t2_scoring_version"}}))
    row = comparison.compare_matched_records([base], [tuned])[0]
    assert row["delta_cpl"] is None
    assert row["t2_cpl_paired_n"] == 0

@pytest.mark.parametrize("base_setting,tuned_setting", [(True, False), (None, True), (None, None)])
def test_legality_delta_requires_known_compatible_policy(base_setting, tuned_setting):
    base = observation(t2_rescue_enabled=base_setting, t2_rescue_path="none")
    tuned = observation(model=FT, t2_rescue_enabled=tuned_setting, t2_rescue_path="none")
    row = comparison.compare_matched_records([base], [tuned])[0]
    assert row["delta_t2_legal"] is None
    assert row["t2_legal_paired_n"] == 0
    assert row["t2_legal_comparability"] != "compatible_recorded_assistance_policy"

def test_same_rescue_policy_allows_different_success_routes():
    base = observation(t2_rescue_enabled=True, t2_rescue_path="response_scan")
    tuned = observation(model=FT, t2_rescue_enabled=True, t2_rescue_path="reprompt")
    row = comparison.compare_matched_records([base], [tuned])[0]
    assert row["delta_t2_legal"] == 0
    assert row["t2_legal_paired_n"] == 1
    assert row["t2_legal_comparability"] == "compatible_recorded_assistance_policy"

def test_move_only_legal_list_policy_independent_of_combined_rescue_switch():
    base = observation(prompt="move_only", t2_rescue_enabled=True, t2_rescue_path="legal_list_scan")
    tuned = observation(model=FT, prompt="move_only", t2_rescue_enabled=False, t2_rescue_path="legal_list_scan")
    row = comparison.compare_matched_records([base], [tuned])[0]
    assert row["delta_t2_legal"] == 0
    assert row["t2_legal_paired_n"] == 1

def test_move_only_unknown_assistance_is_not_assumed():
    row = comparison.compare_matched_records(
        [observation(prompt="move_only")],
        [observation(model=FT, prompt="move_only", t2_rescue_path="legal_list_scan")])[0]
    assert row["delta_t2_legal"] is None
    assert row["t2_legal_paired_n"] == 0
    assert "unknown_assistance" in row["t2_legal_comparability"]
