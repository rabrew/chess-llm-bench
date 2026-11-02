"""Audit metadata tests; all model calls and storage are mocked."""
import pytest
from tests.test_worker import TestProcessJob as WorkerHarness, SAMPLE_JOB, LLM_LEGAL_RESPONSE, worker_config

def make_worker(worker_config, response, rescue=True):
    worker_config["evaluation"]["rescue_enabled"] = rescue
    return WorkerHarness()._setup_worker(worker_config, llm_response={
        "response": response, "inference_ms": 10, "success": True,
    })

def test_first_legal_attempt(worker_config):
    worker = make_worker(worker_config, LLM_LEGAL_RESPONSE["response"])
    record = worker.process_job(SAMPLE_JOB)
    assert record["t2_move_attempt_1"] == "Nf6"
    assert record["t2_legal_attempt_1"] is True
    assert record["t2_rescue_path"] == "none"

@pytest.mark.parametrize("rescue", [True, False])
def test_failed_first_attempt_is_preserved(worker_config, rescue):
    worker = make_worker(worker_config, "Eval: 45\nMove: Qa5\nExplanation: Equal.", rescue)
    worker.llm_client.chat.side_effect = [
        {"response": "Eval: 45\nMove: Qa5\nExplanation: Equal.", "inference_ms": 10, "success": True},
        {"response": "Nf6", "inference_ms": 20, "success": True},
    ]
    record = worker.process_job(SAMPLE_JOB)
    assert record["t2_move_attempt_1"] == "Qa5"
    assert record["t2_legal_attempt_1"] is False
    assert record["t2_move"] == ("Nf6" if rescue else "Qa5")
    assert record["t2_rescue_path"] == ("reprompt" if rescue else "disabled")
    assert worker.llm_client.chat.call_count == (2 if rescue else 1)

def test_scan_rescue_is_disclosed(worker_config):
    worker = make_worker(worker_config, "Eval: 45\nMove: Qa5\nExplanation: Equal. Nf6")
    record = worker.process_job(SAMPLE_JOB)
    assert record["t2_move_attempt_1"] == "Qa5"
    assert record["t2_legal_attempt_1"] is False
    assert record["t2_rescue_path"] == "response_scan"
    assert worker.llm_client.chat.call_count == 1

def test_failed_rescue_is_disclosed(worker_config):
    worker = make_worker(worker_config, "Eval: 45\nMove: Qa5\nExplanation: Equal.")
    worker.llm_client.chat.side_effect = [
        {"response": "Eval: 45\nMove: Qa5\nExplanation: Equal.", "inference_ms": 10, "success": True},
        {"response": "", "inference_ms": 20, "success": False},
    ]
    record = worker.process_job(SAMPLE_JOB)
    assert record["t2_move_attempt_1"] == "Qa5"
    assert record["t2_move"] is None
    assert record["t2_rescue_path"] == "failed"

def test_non_move_format_has_no_legality_attempt(worker_config):
    worker = make_worker(worker_config, "45")
    record = worker.process_job({**SAMPLE_JOB, "prompt_format": "eval_only"})
    assert record["t2_move_attempt_1"] is None
    assert record["t2_legal_attempt_1"] is None
    assert record["t2_rescue_path"] == "not_applicable"

def test_move_only_discloses_legal_list_scan(worker_config):
    worker = make_worker(worker_config, "Nf6")
    record = worker.process_job({**SAMPLE_JOB, "prompt_format": "move_only"})
    assert record["t2_rescue_path"] == "legal_list_scan"
    assert record["t2_move_attempt_1"] == "Nf6"

def test_invalid_rescue_config_rejected(worker_config):
    with pytest.raises(ValueError, match="rescue_enabled"):
        make_worker(worker_config, "Nf6", rescue="false")

def test_empty_response_still_fails_without_record(worker_config):
    worker = make_worker(worker_config, "", rescue=False)
    assert worker.process_job(SAMPLE_JOB) is None
    worker.job_queue.fail_job.assert_called_once()
    worker.result_writer.write_result.assert_not_called()

def test_dataset_reference_provenance_reaches_record(worker_config):
    worker = make_worker(worker_config, LLM_LEGAL_RESPONSE["response"])
    position = dict(worker.data_loader.get_by_id.return_value)
    position["reference_provenance"] = {"engine": "lc0-onnx", "version": "unrecorded", "units": "centipawns", "perspective": "white"}
    worker.data_loader.get_by_id.return_value = position
    record = worker.process_job(SAMPLE_JOB)
    assert record["t1_reference_provenance"] == position["reference_provenance"]
