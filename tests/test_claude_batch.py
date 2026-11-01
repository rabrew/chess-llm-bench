"""Tests for the Claude batch runner (specs/commercial-models.md). No network: a fake client."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import chess
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import claude_batch as cb  # noqa: E402
from src.job_generator import populate_job_queue, subsample_per_tier  # noqa: E402
from src.llm_client import build_prompt, retry_move_prompt  # noqa: E402
from src.utils import API_MODEL_FAMILIES, parse_model_info  # noqa: E402

CONFIG_PATH = REPO_ROOT / "config" / "config_claude.yaml"
START_FEN = chess.STARTING_FEN
# Black to move; Nf6 is legal, e4 is not (it's White's move).
BLACK_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
PRICES = {"claude-haiku-5-5": {"input": 0.10, "output": 0.50},
          "claude-sonnet-5-5": {"input": 2.00, "output": 10.00}}


def _config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _variants() -> list[cb.Variant]:
    return cb.load_variants(_config())


def _variant(tag: str) -> cb.Variant:
    return next(v for v in _variants() if v.tag == tag)


def _position(pid: int, tier: str, fen: str = BLACK_FEN) -> dict:
    return {"id": pid, "fen": fen, "difficulty": tier, "phase": "opening", "source": "lichess_puzzles",
            "theme": "fork", "stockfish_eval": 40, "stockfish_best_move": "Nf6", "pgn_moves": ""}


def _job(fen: str = BLACK_FEN, fmt: str = "fen_only", tag: str = "claude-haiku:5.5") -> dict:
    return cb.build_jobs([_position(7, "easy", fen)], [_variant(tag)], [fmt])[0]


def _message(text: str, input_tokens: int = 100, output_tokens: int = 50, stop: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason=stop, model="claude-haiku-5-5",
    )


def _succeeded(custom_id: str, text: str, **usage) -> SimpleNamespace:
    return SimpleNamespace(custom_id=custom_id,
                           result=SimpleNamespace(type="succeeded", message=_message(text, **usage)))


class FakeBatches:
    """Records submitted batches; answers every request with `reply_for(custom_id)`."""

    def __init__(self, reply_for):
        self.reply_for = reply_for
        self.created: list[list[dict]] = []
        self.requests: dict[str, list[dict]] = {}

    def create(self, requests):
        batch_id = f"batch_{len(self.created)}"
        self.created.append(requests)
        self.requests[batch_id] = requests
        return SimpleNamespace(id=batch_id, processing_status="in_progress")

    def retrieve(self, batch_id):
        return SimpleNamespace(id=batch_id, processing_status="ended",
                               request_counts=SimpleNamespace(processing=0, succeeded=0, errored=0))

    def results(self, batch_id):
        return [self.reply_for(r["custom_id"]) for r in self.requests[batch_id]]


def _client(reply_for) -> SimpleNamespace:
    return SimpleNamespace(messages=SimpleNamespace(batches=FakeBatches(reply_for)))


# ---------- config and registry ----------

def test_every_variant_is_registered_without_a_guessed_size():
    for variant in _variants():
        info = parse_model_info(variant.tag)
        assert info["family"] == "claude"
        assert info["size_b"] is None
    assert set(API_MODEL_FAMILIES) == {v.tag for v in _variants()}


def test_every_variant_has_a_price():
    prices = _config()["claude"]["prices_per_mtok"]
    assert all(v.model in prices for v in _variants())


# ---------- sampling ----------

def test_sample_matches_the_per_tier_job_queue(tmp_path):
    tiers = {t: [_position(1000 * i + n, t) for n in range(40)] for i, t in enumerate(cb.TIERS)}
    loader = SimpleNamespace(load_tier=lambda t: tiers[t],
                             sample=lambda positions, count, seed: positions[:count])
    config = {"benchmark": {"max_positions_per_tier": 30, "random_seed": 42, "sample_per_tier": 5}}
    expected = []
    for tier in cb.TIERS:
        expected += subsample_per_tier(loader.sample(tiers[tier], 30, 42), 5, 42)
    assert [p["id"] for p in cb.sample_positions(config, loader)] == [p["id"] for p in expected]


def test_populate_job_queue_draws_the_same_ids_per_tier():
    """The pairing depends on this: run_all.sh calls populate_job_queue one tier at a time."""
    tiers = {t: [_position(1000 * i + n, t) for n in range(40)] for i, t in enumerate(cb.TIERS)}
    loader = SimpleNamespace(load_tier=lambda t: tiers[t],
                             sample=lambda positions, count, seed: positions[:count])
    config = {"models": ["llama3.2:3b"], "benchmark": {"max_positions_per_tier": 30, "random_seed": 42,
                                                       "sample_per_tier": 5, "prompt_formats": ["fen_only"]}}
    queued = []
    queue = SimpleNamespace(insert_jobs=lambda jobs: queued.extend(jobs) or len(jobs))
    for tier in cb.TIERS:
        populate_job_queue(config, data_loader=loader, job_queue=queue, tier=tier)
    assert sorted(j["position_id"] for j in queued) == sorted(p["id"] for p in cb.sample_positions(config, loader))


def test_pilot_takes_the_first_positions_of_each_tier():
    positions = [_position(n, t) for t in cb.TIERS for n in range(5)]
    pilot = cb.pilot_positions(positions, 2)
    assert len(pilot) == 8
    assert all(sum(p["difficulty"] == t for p in pilot) == 2 for t in cb.TIERS)
    assert cb.pilot_positions(positions, 0) == positions


# ---------- requests ----------

def test_jobs_carry_the_worker_fields():
    job = _job()
    assert job["job_id"] == "job_00007_claude-haiku_5_5_fen_only_1"
    assert job["model"] == "claude-haiku:5.5" and job["prompt_format"] == "fen_only"
    assert job["fen"] == BLACK_FEN and job["stockfish_eval"] == 40 and job["difficulty"] == "easy"


def test_first_request_is_the_worker_prompt_with_no_system_prompt():
    job = _job(fmt="board_text")
    request = cb.first_round_request(job, _variant("claude-haiku:5.5"))
    params = request["params"]
    assert params["messages"] == [{"role": "user", "content": build_prompt(fen=BLACK_FEN, prompt_format="board_text")}]
    assert "system" not in params
    assert request["custom_id"].endswith("-r1")


def test_retry_request_uses_the_worker_reask():
    job = _job(fmt="board_text+legal")
    params = cb.retry_request(job, _variant("claude-haiku:5.5"))["params"]
    prompt, system = retry_move_prompt(BLACK_FEN, "board_text+legal")
    assert params["messages"][0]["content"] == prompt and params["system"] == system
    assert BLACK_FEN.split()[0] not in prompt


@pytest.mark.parametrize("tag, thinking, effort, max_tokens", [
    ("claude-haiku:5.5", None, None, 4000),
    ("claude-sonnet:5.5-nothink", {"type": "between_tools"}, "high", 3000),
    ("claude-sonnet:5.5-think", {"type": "adaptive"}, "high", 8000),
    ("claude-opus:5.5-low", None, "low", 4000),
])
def test_variant_parameters(tag, thinking, effort, max_tokens):
    params = cb.request_params(_variant(tag), "prompt")
    assert params.get("thinking") == thinking
    assert params.get("output_config", {}).get("effort") == effort
    assert params["max_tokens"] == max_tokens
    assert "temperature" not in params


def test_custom_ids_are_api_safe_unique_and_short():
    jobs = cb.build_jobs([_position(9999999, "easy")], _variants(), ["fen_only", "board_text", "board_text+legal"])
    ids = [cb.custom_id(j["job_id"], r) for j in jobs for r in (1, 2)]
    assert len(set(ids)) == len(ids)
    assert all(len(i) <= cb.CUSTOM_ID_MAX and cb.SAFE_ID.fullmatch(i) for i in ids)


def test_overlong_custom_id_is_hashed_stably():
    long_id = "job_" + "x" * 100
    assert cb.custom_id(long_id, 1) == cb.custom_id(long_id, 1)
    assert len(cb.custom_id(long_id, 1)) <= cb.CUSTOM_ID_MAX


# ---------- the move rescue ----------

def test_legal_first_move_needs_no_retry():
    parsed, needs_retry = cb.first_answer(_job(), "Eval: 30\nMove: Nf6\nExplanation: White — development")
    assert parsed["move"] == "Nf6" and not needs_retry


def test_illegal_move_rescued_from_the_reply_text():
    parsed, needs_retry = cb.first_answer(_job(), "Eval: 30\nMove: e4\nExplanation: White — I like Nf6 here")
    assert parsed["move"] == "Nf6" and not needs_retry


def test_illegal_move_with_nothing_to_rescue_needs_retry():
    parsed, needs_retry = cb.first_answer(_job(), "Eval: 30\nMove: e4\nExplanation: White — space")
    assert needs_retry


def test_retry_reply_supplies_the_move_or_none():
    parsed, _ = cb.first_answer(_job(), "Eval: 30\nMove: e4\nExplanation: White — space")
    assert cb.apply_retry(dict(parsed), BLACK_FEN, "Nf6")["move"] == "Nf6"
    assert cb.apply_retry(dict(parsed), BLACK_FEN, "e4")["move"] is None
    assert cb.apply_retry(dict(parsed), BLACK_FEN, "   ")["move"] is None


# ---------- records and cost ----------

def test_cost_uses_the_batch_discount():
    assert cb.request_cost(1_000_000, 1_000_000, PRICES["claude-sonnet-5-5"], 0.5) == pytest.approx(6.0)


def test_record_has_the_usual_schema_and_the_claude_fields():
    job = _job()
    first = {"text": "Eval: 30\nMove: e4\nExplanation: White — space", "input_tokens": 100, "output_tokens": 40,
             "stop_reason": "end_turn", "api_model": "claude-haiku-5-5"}
    retry = {"text": "Nf6", "input_tokens": 60, "output_tokens": 10, "stop_reason": "end_turn",
             "api_model": "claude-haiku-5-5"}
    variant = _variant("claude-haiku:5.5")
    record = cb.build_record(job, variant, first, retry, (-2000, 2000), PRICES, 0.5)
    assert record["t2_move"] == "Nf6" and record["t2_legal"] is True
    assert record["raw_response"] == first["text"] and record["raw_move_retry_response"] == "Nf6"
    assert record["model"] == "claude-haiku:5.5" and record["model_family"] == "claude"
    assert record["input_tokens"] == 160 and record["output_tokens"] == 50
    assert record["cost_usd"] == pytest.approx((160 * 0.10 + 50 * 0.50) / 1e6 * 0.5)
    assert record["api_model"] == "claude-haiku-5-5" and record["stop_reason"] == "end_turn"


def test_record_none_when_every_field_is_missing():
    first = {"text": "", "input_tokens": 1, "output_tokens": 1, "stop_reason": "max_tokens", "api_model": "x"}
    assert cb.build_record(_job(), _variant("claude-haiku:5.5"), first, {"text": ""}, (-2000, 2000), PRICES, 0.5) is None


# ---------- budget ----------

def test_budget_guard_refuses_over_budget():
    with pytest.raises(cb.BudgetExceeded):
        cb.check_budget(spent=20.0, estimate=3.0, budget=22.0)
    cb.check_budget(spent=10.0, estimate=3.0, budget=22.0)


def test_estimate_uses_defaults_then_stored_means():
    config = _config()["claude"]
    variant = _variant("claude-haiku:5.5")
    before = cb.estimate_usd({variant: 100}, [], config)
    defaults = config["default_estimate_tokens"]
    assert before == pytest.approx(cb.request_cost(100 * defaults["input"], 100 * defaults["output"],
                                                   config["prices_per_mtok"]["claude-haiku-5-5"], 0.5))
    stored = [{"tag": variant.tag, "input_tokens": 10, "output_tokens": 10}] * config["min_samples_for_estimate"]
    after = cb.estimate_usd({variant: 100}, stored, config)
    assert after == pytest.approx(cb.request_cost(1000, 1000, config["prices_per_mtok"]["claude-haiku-5-5"], 0.5))


def test_spent_sums_stored_costs():
    rows = [{"cost_usd": 0.25}, {"cost_usd": 0.5}]
    assert cb.spent_usd(rows) == pytest.approx(0.75)


# ---------- batches: store, resume, errors ----------

def test_run_round_stores_successes_and_skips_errors(tmp_path):
    store, ledger = cb.ResponseStore(tmp_path / "r.jsonl"), cb.BatchLedger(tmp_path / "b.json")
    jobs = cb.build_jobs([_position(1, "easy"), _position(2, "easy")], [_variant("claude-haiku:5.5")], ["fen_only"])
    requests = [cb.first_round_request(j, _variant("claude-haiku:5.5")) for j in jobs]
    errored_id = requests[1]["custom_id"]

    def reply(cid):
        if cid == errored_id:
            return SimpleNamespace(custom_id=cid, result=SimpleNamespace(type="errored", error=SimpleNamespace(type="api_error")))
        return _succeeded(cid, "Move: Nf6")

    client = _client(reply)
    lookup = cb.request_lookup(jobs, _variants())
    cb.run_round(client, requests, 1, ledger, store, lookup, PRICES, 0.5, poll_seconds=0, sleep=lambda s: None)
    rows = store.load()
    assert set(rows) == {requests[0]["custom_id"]}
    assert rows[requests[0]["custom_id"]]["text"] == "Move: Nf6"
    assert ledger.open_batches() == []


def test_open_batch_is_collected_not_resubmitted(tmp_path):
    store, ledger = cb.ResponseStore(tmp_path / "r.jsonl"), cb.BatchLedger(tmp_path / "b.json")
    job = _job()
    request = cb.first_round_request(job, _variant("claude-haiku:5.5"))
    client = _client(lambda cid: _succeeded(cid, "Move: Nf6"))
    client.messages.batches.requests["batch_old"] = [request]
    ledger.add("batch_old", 1)
    lookup = cb.request_lookup([job], _variants())
    cb.collect_open_batches(client, ledger, store, lookup, PRICES, 0.5, poll_seconds=0, sleep=lambda s: None)
    assert client.messages.batches.created == []
    assert request["custom_id"] in store.load()
    assert ledger.open_batches() == []


def test_pending_requests_skip_stored_jobs(tmp_path):
    store = cb.ResponseStore(tmp_path / "r.jsonl")
    jobs = cb.build_jobs([_position(1, "easy"), _position(2, "easy")], [_variant("claude-haiku:5.5")], ["fen_only"])
    variants = {v.tag: v for v in _variants()}
    store.append([{"custom_id": cb.custom_id(jobs[0]["job_id"], 1), "text": "x", "cost_usd": 0}])
    pending = cb.first_round_pending(jobs, variants, store.load())
    assert [r["custom_id"] for r in pending] == [cb.custom_id(jobs[1]["job_id"], 1)]


def test_retry_pending_only_for_unrescued_first_answers(tmp_path):
    jobs = cb.build_jobs([_position(1, "easy"), _position(2, "easy")], [_variant("claude-haiku:5.5")], ["fen_only"])
    variants = {v.tag: v for v in _variants()}
    stored = {cb.custom_id(jobs[0]["job_id"], 1): {"text": "Move: Nf6"},
              cb.custom_id(jobs[1]["job_id"], 1): {"text": "Move: e4"}}
    pending = cb.retry_pending(jobs, variants, stored)
    assert [r["custom_id"] for r in pending] == [cb.custom_id(jobs[1]["job_id"], 2)]


def test_finished_records_written_once(tmp_path):
    jobs = cb.build_jobs([_position(1, "easy")], [_variant("claude-haiku:5.5")], ["fen_only"])
    variants = {v.tag: v for v in _variants()}
    stored = {cb.custom_id(jobs[0]["job_id"], 1): {"text": "Eval: 10\nMove: Nf6\nExplanation: Equal — level",
                                                   "input_tokens": 1, "output_tokens": 1, "stop_reason": "end_turn",
                                                   "api_model": "claude-haiku-5-5"}}
    results, failures = tmp_path / "e.jsonl", tmp_path / "f.jsonl"
    for _ in range(2):
        cb.write_finished(jobs, variants, stored, results, failures, (-2000, 2000), PRICES, 0.5)
    lines = results.read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["t2_move"] == "Nf6"


# ---------- API key ----------

def test_key_from_environment_first(tmp_path):
    key_file = tmp_path / "k.env"
    key_file.write_text("ANTHROPIC_API_KEY=from-file\n")
    assert cb.load_api_key({"ANTHROPIC_API_KEY": "from-env"}, key_file) == "from-env"


def test_key_from_file_when_no_env(tmp_path):
    key_file = tmp_path / "k.env"
    key_file.write_text("# comment\nANTHROPIC_API_KEY = from-file \n")
    os.chmod(key_file, 0o600)
    assert cb.load_api_key({}, key_file) == "from-file"


def test_key_missing_raises_with_the_path(tmp_path):
    with pytest.raises(cb.MissingApiKey, match="k.env"):
        cb.load_api_key({}, tmp_path / "k.env")


def test_key_file_readable_by_others_warns(tmp_path, caplog):
    key_file = tmp_path / "k.env"
    key_file.write_text("ANTHROPIC_API_KEY=x\n")
    os.chmod(key_file, 0o644)
    cb.load_api_key({}, key_file)
    assert "chmod 600" in caplog.text
