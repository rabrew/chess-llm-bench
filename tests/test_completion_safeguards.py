"""Safe handover and collection evidence, without live service calls."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from scripts import wait_for_ollama_idle as idle
from scripts import plain_board_compare as comparison


@pytest.fixture
def clock(monkeypatch):
    elapsed = [0.0]
    monkeypatch.setattr(idle.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(idle.time, "sleep", lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds))
    return elapsed


def response(models):
    result = Mock()
    result.json.return_value = {"models": models}
    return result


def test_waits_for_expiry_without_unloading_or_inference(monkeypatch, clock):
    query = Mock(side_effect=[response([{"name": "local"}]), response([])])
    monkeypatch.setattr(idle.requests, "get", query)
    idle.wait_until_idle("http://example", timeout=10, poll_interval=2)
    assert clock[0] == 2
    assert query.call_count == 2
    assert all(call.args == ("http://example/api/ps",) for call in query.call_args_list)


@pytest.mark.parametrize("failure", [requests.ConnectionError("offline"), ValueError("invalid JSON")])
def test_unknown_state_is_not_idle(monkeypatch, clock, failure):
    monkeypatch.setattr(idle.requests, "get", Mock(side_effect=failure))
    with pytest.raises(TimeoutError, match="idle"):
        idle.wait_until_idle("http://example", timeout=4, poll_interval=2)
    assert clock[0] == 4


def test_malformed_or_busy_status_expires(monkeypatch, clock):
    query = Mock(return_value=response(None))
    monkeypatch.setattr(idle.requests, "get", query)
    with pytest.raises(TimeoutError):
        idle.wait_until_idle("http://example", timeout=3, poll_interval=2)
    assert clock[0] == 3


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_wait_deadline_makes_no_request(monkeypatch, timeout):
    query = Mock()
    monkeypatch.setattr(idle.requests, "get", query)
    with pytest.raises(ValueError):
        idle.wait_until_idle("http://example", timeout=timeout)
    query.assert_not_called()


def test_pipeline_waits_before_enrichment():
    script = (Path(__file__).resolve().parents[1] / "bin/run_all.sh").read_text()
    assert script.index("wait_for_ollama_idle.py") < script.index("python scripts/enrich_cpl.py")


@pytest.mark.parametrize("contents, expected", [
    ("still collecting\n", False), ("EXIT 1\n", False),
    ("EXIT 0\n", True), ("EXIT 0\nmore work\nEXIT 1\n", False),
    ("EXIT 1\nEXIT 0\n", True),
    ("EXIT 0\nMODEL: another run is in progress\n", False),
    ("EXIT 0\n\n", True),
])
def test_completion_log_uses_last_exit_marker(tmp_path, contents, expected):
    path = tmp_path / "run.log"
    path.write_text(contents)
    evidence = comparison.completion_evidence(path)
    assert evidence["complete"] is expected
    assert evidence["sha256"]


def test_missing_completion_log_is_not_evidence(tmp_path):
    with pytest.raises(FileNotFoundError):
        comparison.completion_evidence(tmp_path / "missing.log")
