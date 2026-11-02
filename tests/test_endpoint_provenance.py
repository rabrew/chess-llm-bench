from unittest.mock import MagicMock
from scripts.enrich_cpl import enrich_records
from src.result_writer import build_result_record
from tests.test_enrich_cpl import make_record, make_lc0_result

def test_enrichment_retains_actual_endpoints():
    engine = MagicMock()
    engine.nodes = 800
    engine.evaluate.side_effect = [make_lc0_result(500, "d4"), make_lc0_result(-500)]
    record = enrich_records([make_record(t1_stockfish_eval=1500)], engine)[0]
    assert record["t2_reference_eval_before"] == 500
    assert record["t2_reference_eval_after"] == -500
    assert record["t2_reference_budget"] == {"nodes": 800}
    assert record["t2_reference_units"] == "centipawns_white"

def test_best_move_endpoint_zero_convention():
    engine = MagicMock()
    engine.nodes = 800
    engine.evaluate.return_value = make_lc0_result(500, "e4")
    record = enrich_records([make_record()], engine)[0]
    assert record.get("t2_reference_eval_after") is None
    assert record["t2_best_move_zero_convention"] is True
    assert record["t2_cpl"] == 0

def test_failed_post_move_has_no_endpoint_pair():
    engine = MagicMock()
    engine.evaluate.side_effect = [make_lc0_result(500, "d4"), RuntimeError("failed")]
    record = enrich_records([make_record()], engine)[0]
    assert record.get("t2_reference_eval_after") is None

def test_writer_keeps_reference_metadata():
    job = {"job_id": "test", "position_id": 1, "model": "llama3.2:3b", "t1_reference_engine": "lc0"}
    scores = {"t2_reference_eval_before": 900, "t2_reference_eval_after": -900, "t2_scoring_version": "test"}
    record = build_result_record(job, {}, scores, 1)
    assert record["t2_reference_eval_before"] == 900
    assert record["t1_reference_engine"] == "lc0"
