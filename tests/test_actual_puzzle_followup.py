"""Offline tests for the isolated actual-position follow-up."""

import json
import sys
from types import SimpleNamespace

import chess
import pytest

from scripts import actual_puzzle_followup as follow


def position():
    board = chess.Board()
    setup = board.fen()
    board.push_uci("e2e4")
    return {"parent_position_id": 1, "setup_fen": setup, "fen": board.fen(),
            "position_id": follow.fen_id(board.fen()), "puzzle_id": "toy",
            "difficulty": "easy", "themes_all": ["crushing", "fork"],
            "solution_uci": "e7e5", "setup_move_uci": "e2e4", "rating": 1000}


def test_sample_balanced_seeded():
    rows = [{"parent_position_id": n + i * 40, "difficulty": tier}
            for i, tier in enumerate(follow.TIERS) for n in range(30)]
    one = follow.sample_positions(rows)
    assert one == follow.sample_positions(list(reversed(rows)))
    assert len(one) == 100
    assert all(sum(r["difficulty"] == t for r in one) == 25 for t in follow.TIERS)
    with pytest.raises(ValueError):
        follow.sample_positions(rows[:30])


def test_solution_validation_entire_line():
    p = position()
    row = {"PuzzleId": "toy", "FEN": p["setup_fen"], "Moves": "e2e4 e7e5 g1f3",
           "Rating": "1000", "Themes": "crushing fork"}
    assert follow.validate_solution(p, row)["solution_uci"] == "e7e5"
    for field, value in [("Moves", "e2e4 e7e5 e7e6"), ("FEN", p["fen"]),
                         ("PuzzleId", "other"), ("Themes", "mate")]:
        with pytest.raises(ValueError):
            follow.validate_solution(p, {**row, field: value})


def test_payload_options_formats_and_no_rescue():
    for model in follow.MODELS:
        for fmt in follow.FORMATS:
            payload = follow.request_payload(model, position(), fmt)
            assert payload["options"]["seed"] == 42
            assert payload["options"]["num_thread"] == 4
            assert "temperature" not in payload["options"]
            assert payload["keep_alive"] == "30s"
            if model.startswith("gemma4"):
                assert payload["think"] is False
            else:
                assert "think" not in payload
            assert "Legal moves:" not in payload["messages"][0]["content"]
    with pytest.raises(ValueError):
        follow.request_payload("llama3.3:70b", position(), "fen_only")


def test_reference_actual_fen_and_black_pov():
    seen = []
    def network(fen):
        seen.append(fen)
        return 120  # side-to-move POV, black in this toy
    def search(fen):
        seen.append(fen)
        return {"eval": -110, "best_move": "e5"}
    ref = follow.reference_position(position(), network, search)
    assert seen == [position()["fen"]] * 2
    assert ref["stockfish_eval"] == -120
    assert ref["search_eval"] == -110


def test_ledger_durable_resume_unknown_and_mutation(tmp_path):
    path = tmp_path / "ledger.jsonl"
    payload = {"model": "toy"}
    assert follow.read_ledger(path, ["a"], "hash") == {}
    follow.append_jsonl(path, {"event": "reserved", "job_id": "a", "manifest_hash": "hash",
                              "payload": payload})
    with pytest.raises(ValueError, match="unresolved"):
        follow.read_ledger(path, ["a"], "hash")
    follow.append_jsonl(path, {"event": "completed", "job_id": "a", "manifest_hash": "hash",
                              "response": {"done_reason": "length"}, "payload": payload})
    assert follow.read_ledger(path, ["a"], "hash")["a"]["response"]["done_reason"] == "length"
    with pytest.raises(ValueError):
        follow.read_ledger(path, ["a"], "changed")
    follow.append_jsonl(path, {"event": "completed", "job_id": "a", "manifest_hash": "hash"})
    with pytest.raises(ValueError):
        follow.read_ledger(path, ["a"], "hash")


@pytest.mark.parametrize("content", ["", "{}\n", "{", "[]\n"])
def test_partial_or_malformed_ledger_fails(tmp_path, content):
    path = tmp_path / "ledger.jsonl"
    path.write_text(content)
    with pytest.raises(ValueError):
        follow.read_ledger(path, ["a"], "hash")


def test_lock_excludes_second_writer(tmp_path):
    with follow.execution_lock(tmp_path):
        with pytest.raises(RuntimeError):
            with follow.execution_lock(tmp_path):
                pass


def test_build_records_raw_capture_illegal_no_recovery():
    p = {**position(), "stockfish_eval": -120, "stockfish_best_move": "e5"}
    job = follow.make_jobs([p])[0]
    event = {"response": {"message": {"content": "Eval: -120\nMove: e4\nExplanation: Black — fork. e5"},
                          "done_reason": "length", "eval_count": 400}, "elapsed_ms": 12}
    r = follow.answer_record(job, event)
    assert not r["t2_legal"]
    assert r["done_reason"] == "length" and r["ollama_response"]["eval_count"] == 400
    assert r["t3_p2_theme_correct"] == 1
    assert r["solution_match"] is False


def test_request_exception_is_terminal_error(tmp_path):
    p = {**position(), "stockfish_eval": -120, "stockfish_best_move": "e5"}
    jobs = follow.make_jobs([p])[:1]
    def fail(payload):
        raise TimeoutError("unknown")
    with pytest.raises(TimeoutError):
        follow.collect_jobs(jobs, tmp_path / "raw.jsonl", "h", fail, lambda: None)
    assert len((tmp_path / "raw.jsonl").read_text().splitlines()) == 2
    done = follow.read_ledger(tmp_path / "raw.jsonl", [jobs[0]["job_id"]], "h")
    assert done[jobs[0]["job_id"]]["event"] == "error"


def test_collection_serial_resume_and_idle(tmp_path):
    jobs = follow.make_jobs([{**position(), "stockfish_eval": -120, "stockfish_best_move": "e5"}])
    calls = []
    def send(payload):
        calls.append(payload["model"])
        return {"message": {"content": "Eval: -120\nMove: e5\nExplanation: Black — fork"}, "done": True}
    idle = []
    path = tmp_path / "raw.jsonl"
    follow.collect_jobs(jobs, path, "h", send, lambda: idle.append(True))
    assert calls == [m for m in follow.MODELS for _ in follow.FORMATS]
    assert len(idle) == 3
    follow.collect_jobs(jobs, path, "h", send, lambda: idle.append(True))
    assert len(calls) == 6


def test_mean_random_baseline_and_bootstrap():
    rows = [{"position_id": 1, "random_solution_rate": .5, "solution_match": True,
             "t2_legal": True, "t2_cpl": 0, "stockfish_eval": -100,
             "t1_model_eval": -100, "t3_side_claimed": "Black", "fen": position()["fen"]},
            {"position_id": 2, "random_solution_rate": .1, "solution_match": False,
             "t2_legal": False, "t2_cpl": None, "stockfish_eval": -200,
             "t1_model_eval": -200, "t3_side_claimed": "White", "fen": position()["fen"]}]
    stats = follow.describe(rows)
    assert stats["random_solution_rate"] == pytest.approx(.3)
    assert stats["legal_nonblunder_per_request"] == .5
    assert stats["side_to_move_baseline"] == 1
    assert stats["side_accuracy"] == .5
    assert follow.bootstrap(rows, resamples=20) == follow.bootstrap(rows, resamples=20)


def test_dry_run_no_reads_writes_network(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("output_dir: results/actual_puzzle_followup\n")
    monkeypatch.setattr(follow, "load_manifest", lambda *a: pytest.fail("manifest read"))
    follow.main(["collect", "--config", str(config), "--dry-run"])
    assert '600' in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == [config]


def test_output_path_fails_closed(tmp_path):
    for path in ("results/plain_board", "data/new", "results/actual_puzzle_followup/../claude"):
        with pytest.raises(ValueError):
            follow.output_directory({"output_dir": path})


def test_output_symlink_cannot_redirect_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(follow, "ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "results/actual_puzzle_followup").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    with pytest.raises(ValueError):
        follow.output_directory({})


def test_manifest_collision_or_mutation_rejected(tmp_path):
    p = position()
    with pytest.raises(ValueError):
        follow.validate_positions([p, p])
    manifest = {"positions": [p], "models": list(follow.MODELS), "formats": list(follow.FORMATS)}
    manifest["manifest_hash"] = follow.digest(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert follow.load_manifest(path)["positions"] == [p]
    manifest["positions"][0]["fen"] = chess.STARTING_FEN
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        follow.load_manifest(path)


def test_lost_established_state_and_valid_prefix_fail(tmp_path):
    p = {**position(), "stockfish_eval": -120, "stockfish_best_move": "e5"}
    jobs = follow.make_jobs([p])[:1]
    path = tmp_path / "raw.jsonl"
    reply = {"message": {"content": "Move: e5"}}
    follow.collect_jobs(jobs, path, "h", lambda _: reply, lambda: None)
    text = path.read_text()
    path.write_text(text.splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="truncated"):
        follow.collect_jobs(jobs, path, "h", lambda _: pytest.fail("resend"), lambda: None)
    path.unlink()
    path.with_suffix(".guard.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        follow.collect_jobs(jobs, path, "h", lambda _: pytest.fail("resend"), lambda: None)


def test_reference_network_then_search_released_and_white_pov(monkeypatch):
    order = []
    class Session:
        def __init__(self, *a, **k):
            order.append("network-open")
        def get_inputs(self):
            return [SimpleNamespace(name="planes")]
        def run(self, *a):
            order.append("network")
            return [None, [[120]]]
        def __del__(self):
            order.append("network-released")
    class Engine:
        def evaluate(self, fen):
            assert fen == position()["fen"]
            order.append("search")
            return {"eval": -110, "best_move": "e5"}
        def close(self):
            order.append("search-closed")
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=Session))
    monkeypatch.setitem(sys.modules, "scripts.precompute_lc0_batch_v2", SimpleNamespace(
        encode_position=lambda _: __import__('numpy').zeros((1,)),
        wdl_to_centipawns=lambda value: value[0]))
    manifest = {"positions": [position()], "config": {"lc0": {"onnx_model": "fake"}}}
    refs = follow.references(manifest, lambda: order.append("idle"), Engine)
    assert refs[0]["stockfish_eval"] == -120
    assert order == ["idle", "network-open", "network", "network-released", "idle", "search", "search-closed"]


def test_unknown_idle_never_opens_reference_resources():
    def fail():
        raise TimeoutError("unknown Ollama state")
    with pytest.raises(TimeoutError):
        follow.references({}, fail, lambda: pytest.fail("engine opened"))


def test_score_preserves_raw_and_uses_actual_search_reference(tmp_path):
    p = {**position(), "stockfish_eval": -120, "stockfish_best_move": "e5",
         "search_eval": -110, "search_nodes": 800}
    manifest = {"positions": [position()], "manifest_hash": "h"}
    follow.save_artifact(tmp_path / "references.json", manifest, [p])
    jobs = follow.make_jobs([p])
    response = {"message": {"content": "Eval: -100\nMove: d5\nExplanation: Black — fork"}}
    follow.collect_jobs(jobs, tmp_path / "raw.jsonl", "h", lambda _: response, lambda: None)
    before = (tmp_path / "raw.jsonl").read_bytes()
    order = []
    class Engine:
        def evaluate(self, fen):
            assert fen != p["setup_fen"] and fen != p["fen"]
            order.append("post-move")
            return {"eval": -50}
        def close(self):
            order.append("closed")
    records = follow.score_answers(manifest, tmp_path, lambda: order.append("idle"), Engine)
    assert order == ["idle", "post-move", "closed"]
    assert all(row["t2_cpl_eval_before"] == -110 and row["t2_cpl"] == 60 for row in records)
    assert all(row["raw_response"] == response["message"]["content"] for row in records)
    assert (tmp_path / "raw.jsonl").read_bytes() == before


def test_endpoint_is_only_local_ollama():
    for base in ("https://api.openai.com", "http://example.com:11434", "http://localhost/api"):
        with pytest.raises(ValueError):
            follow.local_endpoint({"ollama": {"base_url": base}})


def test_malformed_transport_reply_saved_without_resend(tmp_path):
    job = follow.make_jobs([position()])[:1]
    def send(_):
        return {"_validation_error": "bad JSON", "_raw_http_body": "oops"}
    path = tmp_path / "raw.jsonl"
    with pytest.raises(ValueError):
        follow.collect_jobs(job, path, "h", send, lambda: None)
    completed = follow.read_ledger(path, [job[0]["job_id"]], "h")
    assert completed[job[0]["job_id"]]["response"]["_raw_http_body"] == "oops"
    follow.collect_jobs(job, path, "h", lambda _: pytest.fail("resend"), lambda: None)


def test_paired_blunders_require_common_legal_scored_positions(monkeypatch):
    monkeypatch.setattr(follow, "RESAMPLES", 20)
    base = {"fen": position()["fen"], "stockfish_eval": -100,
            "random_solution_rate": .1, "t1_model_eval": -100,
            "t3_side_claimed": "Black", "solution_match": False}
    left = [{**base, "position_id": 1, "t2_legal": True, "t2_cpl": 0},
            {**base, "position_id": 2, "t2_legal": False, "t2_cpl": None}]
    right = [{**base, "position_id": 1, "t2_legal": False, "t2_cpl": None},
             {**base, "position_id": 2, "t2_legal": True, "t2_cpl": 1500}]
    result = follow.paired_effect(left, right)
    assert result["difference_board_minus_fen"]["blunder_rate"] is None
    assert result["measure_n_pairs"]["blunder_rate"] == 0
    assert result["ci"]["blunder_rate"] is None
    assert result["measure_n_pairs"]["first_answer_legal"] == 2
    assert result["difference_board_minus_fen"]["first_answer_legal"] == 0
    assert result["difference_board_minus_fen"]["legal_nonblunder_per_request"] == -.5


def test_paired_blunders_exclude_unshared_legal_outputs(monkeypatch):
    monkeypatch.setattr(follow, "RESAMPLES", 20)
    base = {"fen": position()["fen"], "stockfish_eval": -100,
            "random_solution_rate": .1, "solution_match": False}
    left = [{**base, "position_id": 1, "t2_legal": True, "t2_cpl": 0},
            {**base, "position_id": 2, "t2_legal": True, "t2_cpl": 1500}]
    right = [{**base, "position_id": 1, "t2_legal": True, "t2_cpl": 1500},
             {**base, "position_id": 2, "t2_legal": False, "t2_cpl": None}]
    result = follow.paired_effect(left, right)
    assert result["measure_n_pairs"]["blunder_rate"] == 1
    assert result["difference_board_minus_fen"]["blunder_rate"] == 1
    assert result["measure_n_pairs"]["t1_rho"] == 0


def test_side_baselines_share_recognized_claim_cohort():
    rows = [{"fen": position()["fen"], "stockfish_eval": -100, "t3_side_claimed": "Black"},
            {"fen": position()["fen"], "stockfish_eval": 100, "t3_side_claimed": None}]
    result = follow.side_statistics(rows)
    assert result["side_accuracy"] == 1
    assert result["majority_baseline"] == 1
    assert result["side_to_move_baseline"] == 1
    assert result["side_n"] == result["baseline_n"] == 1
    assert result["all_request_side_to_move_baseline"] == .5
    assert result["all_request_baseline_n"] == 2
