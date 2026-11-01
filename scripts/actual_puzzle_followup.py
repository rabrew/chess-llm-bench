"""Isolated exploratory solver-position study. Every live stage is explicit.

Run --dry-run to inspect the fixed plan without loading data or contacting services.
See specs/actual-puzzle-followup.md; never use the historical benchmark queues.
"""

import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from urllib.parse import urlparse

import chess
import numpy as np
import requests
from scipy.stats import spearmanr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset_builder import _fen_to_id as fen_id  # noqa: E402
from src.evaluator import (NON_TACTICAL_THEMES, get_direction, score_all,
                           theme_mentioned)  # noqa: E402
from src.llm_client import OllamaClient, build_prompt, parse_response  # noqa: E402
from src.metrics import cohen_kappa  # noqa: E402
from src.provenance import file_sha256  # noqa: E402
from src.result_writer import build_result_record  # noqa: E402

MODELS = ("qwen2.5:7b", "qwen2.5:32b", "gemma4:31b")
FORMATS = ("fen_only", "board_text")
TIERS = ("easy", "medium", "hard", "extreme")
SEED = 42
PER_TIER = 25
RESAMPLES = 1000
NODES = 800
BLUNDER = 1000
OUTPUT = Path("results/actual_puzzle_followup")
ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("JSONL entry must be an object")
            yield row


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def append_jsonl(path, value):
    encoded = json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("a") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def output_directory(config):
    raw = Path(config.get("output_dir", OUTPUT))
    if ".." in raw.parts:
        raise ValueError("Output traversal is forbidden")
    path = (ROOT / raw).resolve()
    if path != ROOT / OUTPUT:
        raise ValueError("Only the isolated actual_puzzle_followup output is allowed")
    return path


@contextmanager
def execution_lock(directory):
    with (Path(directory) / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Follow-up stage already running") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def sample_positions(rows):
    rng = random.Random(SEED)
    selected = []
    for tier in TIERS:
        pool = sorted((r for r in rows if r["difficulty"] == tier),
                      key=lambda row: row["parent_position_id"])
        if len(pool) < PER_TIER:
            raise ValueError(f"Insufficient {tier} parents")
        selected.extend(rng.sample(pool, PER_TIER))
    return selected


def validate_positions(positions):
    for field in ("position_id", "fen", "puzzle_id", "parent_position_id"):
        values = [row[field] for row in positions]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate/colliding {field}")
    for row in positions:
        board = chess.Board(row["setup_fen"])
        board.push_uci(row["setup_move_uci"])
        if board.fen() != row["fen"] or fen_id(row["fen"]) != row["position_id"]:
            raise ValueError("Invalid actual position or stable ID")
        if not board.is_valid() or chess.Move.from_uci(row["solution_uci"]) not in board.legal_moves:
            raise ValueError("Invalid solver position/target")


def validate_solution(position, row):
    fields = (("PuzzleId", "puzzle_id"), ("FEN", "setup_fen"))
    if any(row[a] != position[b] for a, b in fields):
        raise ValueError("CSV identity/FEN conflict")
    if int(row["Rating"]) != position["rating"] or row["Themes"].split() != position["themes_all"]:
        raise ValueError("CSV metadata conflict")
    moves = row["Moves"].split()
    if len(moves) < 2 or moves[0] != position["setup_move_uci"]:
        raise ValueError("Missing/conflicting setup or solver move")
    board = chess.Board(row["FEN"])
    for index, move in enumerate(moves):
        board.push_uci(move)
        if index == 0 and board.fen() != position["fen"]:
            raise ValueError("CSV setup replay differs from actual FEN")
    return {**position, "solution_uci": moves[1], "solution_line_uci": moves}


def parent_positions(results, meta):
    parents = {}
    for row in read_jsonl(results):
        pid, fen, tier = row["position_id"], row["fen"], row["difficulty"]
        if pid in parents and (parents[pid]["setup_fen"], parents[pid]["difficulty"]) != (fen, tier):
            raise ValueError("Conflicting parent identity")
        item = meta.get(fen)
        if item is None or not item.get("puzzle_fen"):
            raise ValueError("Missing puzzle metadata")
        parents[pid] = {"parent_position_id": pid, "setup_fen": fen,
                        "fen": item["puzzle_fen"], "position_id": fen_id(item["puzzle_fen"]),
                        "difficulty": tier, **{k: item[k] for k in
                        ("puzzle_id", "rating", "themes_all", "setup_move_uci")}}
    return list(parents.values())


def prepare_manifest(config, config_path):
    from scripts.build_puzzle_meta import read_puzzle_rows
    meta_path, parent_path = Path(config["puzzle_meta"]), Path(config["parent_results"])
    source_paths = (parent_path, meta_path, Path(config["puzzle_csv"]), Path(config_path),
                    *(Path(config["lc0"][key]) for key in ("binary", "weights", "onnx_model")))
    sources = {str(path): file_sha256(path) for path in source_paths}
    meta = json.loads(meta_path.read_text())["positions"]
    parents = parent_positions(parent_path, meta)
    if len(parents) != 500:
        raise ValueError("Parent cohort must contain exactly 500 positions")
    selected = sample_positions(parents)
    wanted = {r["puzzle_id"]: r for r in selected}
    found = {}
    for row in read_puzzle_rows(Path(config["puzzle_csv"])):
        pid = row["PuzzleId"]
        if pid in wanted:
            if pid in found:
                raise ValueError("Duplicate CSV puzzle ID")
            found[pid] = validate_solution(wanted[pid], row)
    if set(found) != set(wanted):
        raise ValueError("Missing selected CSV solutions")
    positions = [found[row["puzzle_id"]] for row in selected]
    validate_positions(positions)
    if sources != {str(path): file_sha256(path) for path in source_paths}:
        raise ValueError("A source changed during manifest preparation")
    manifest = {"positions": positions, "models": list(MODELS), "formats": list(FORMATS),
                "seed": SEED, "sources": sources, "config": config,
                "code_hashes": code_hashes(),
                "payloads": {m: request_payload(m, positions[0], FORMATS[0])["options"] for m in MODELS}}
    manifest["manifest_hash"] = digest(manifest)
    return manifest


def code_hashes():
    paths = ("scripts/actual_puzzle_followup.py", "scripts/precompute_lc0_batch_v2.py",
             "src/llm_client.py", "src/evaluator.py", "src/engine_wrapper.py",
             "src/result_writer.py", "src/metrics.py", "scripts/enrich_cpl.py")
    return {name: file_sha256(ROOT / name) for name in paths}


def load_manifest(path):
    value = json.loads(Path(path).read_text())
    body = {k: v for k, v in value.items() if k != "manifest_hash"}
    if value.get("manifest_hash") != digest(body):
        raise ValueError("Manifest changed or incomplete")
    if value["models"] != list(MODELS) or value["formats"] != list(FORMATS):
        raise ValueError("Manifest model/format plan differs")
    validate_positions(value["positions"])
    return value


def request_payload(model, position, fmt):
    if model not in MODELS or fmt not in FORMATS:
        raise ValueError("Unregistered model/format")
    options = {**OllamaClient._get_options(model), "seed": SEED, "num_thread": 4}
    options.pop("temperature", None)
    payload = {"model": model, "messages": [{"role": "user", "content":
               build_prompt(position["fen"], prompt_format=fmt)}], "stream": False,
               "options": options, "keep_alive": "30s"}
    if model.startswith("gemma4"):
        payload["think"] = False
    return payload


def make_jobs(positions):
    return [{**position, "model": model, "prompt_format": fmt,
             "job_id": f"{model}|{fmt}|{position['position_id']}"}
            for model in MODELS for position in positions for fmt in FORMATS]


def read_ledger(path, job_ids, manifest_hash):
    path = Path(path)
    if not path.exists():
        return {}
    reserved, done = set(), {}
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError("Existing ledger is empty")
    for row in rows:
        jid, event = row.get("job_id"), row.get("event")
        if jid not in job_ids or row.get("manifest_hash") != manifest_hash or jid in done:
            raise ValueError("Unknown, changed or duplicate ledger record")
        if event == "reserved" and jid not in reserved:
            reserved.add(jid)
        elif event in ("completed", "error") and jid in reserved:
            reserved.remove(jid)
            done[jid] = row
        else:
            raise ValueError("Malformed ledger transition")
    if reserved:
        raise ValueError("unresolved request: manual review required; never auto-resend")
    return done


def verify_guard(path):
    guard = path.with_suffix(".guard.json")
    if guard.exists() != path.exists():
        raise ValueError("Partial/missing request state; manual review required")
    if guard.exists() and json.loads(guard.read_text())["sha256"] != file_sha256(path):
        raise ValueError("Ledger changed, truncated or lost entries")


def guarded_append(path, event):
    append_jsonl(path, event)
    atomic_json(path.with_suffix(".guard.json"), {"sha256": file_sha256(path)})


def collect_jobs(jobs, path, manifest_hash, send, idle):
    path = Path(path)
    started = path.with_suffix(".started.json")
    if started.exists() and not path.exists():
        raise ValueError("Established request ledger missing; manual review required")
    verify_guard(path)
    done = read_ledger(path, {j["job_id"] for j in jobs}, manifest_hash)
    for job in jobs:
        if job["job_id"] in done and done[job["job_id"]].get("payload") != request_payload(
                job["model"], job, job["prompt_format"]):
            raise ValueError("Frozen request payload changed")
    if not started.exists():
        atomic_json(started, {"manifest_hash": manifest_hash})
    previous = None
    for job in jobs:
        if job["job_id"] in done:
            continue
        if job["model"] != previous:
            idle()
            previous = job["model"]
        payload = request_payload(job["model"], job, job["prompt_format"])
        base = {"job_id": job["job_id"], "manifest_hash": manifest_hash, "payload": payload}
        collect_one(path, base, send)


def collect_one(path, base, send):
    guarded_append(path, {**base, "event": "reserved"})
    start = time.monotonic()
    try:
        response = send(base["payload"])
    except Exception as exc:
        reply = getattr(exc, "response", None)
        guarded_append(path, {**base, "event": "error", "error": str(exc),
                              "raw_http_body": reply.text if reply is not None else None,
                              "elapsed_ms": int((time.monotonic() - start) * 1000)})
        raise
    if not isinstance(response, dict) or response.get("_validation_error"):
        guarded_append(path, {**base, "event": "error", "response": response,
                              "error": "Malformed Ollama response", "elapsed_ms": 0})
        raise ValueError("Malformed Ollama response; saved as terminal error")
    guarded_append(path, {**base, "event": "completed", "response": response,
                          "elapsed_ms": int((time.monotonic() - start) * 1000)})


def reference_position(position, network, search):
    fen = position["fen"]
    cp = network(fen)
    result = search(fen)
    sign = 1 if chess.Board(fen).turn == chess.WHITE else -1
    return {**position, "stockfish_eval": sign * cp, "stockfish_best_move": result["best_move"],
            "search_eval": result["eval"], "search_nodes": NODES}


def answer_record(job, event):
    response = event.get("response", {})
    if not isinstance(response, dict):
        response = {}
    text = response.get("message", {}).get("content", "") if isinstance(response, dict) else ""
    parsed = parse_response(text)
    scores = score_all(parsed, job, engine=None)
    tactical = [t for t in job["themes_all"] if t not in NON_TACTICAL_THEMES]
    scores["t3_p2_theme_correct"] = (int(any(theme_mentioned(t, parsed["explanation"])
                                             for t in tactical)) if tactical and parsed["explanation"] else None)
    scores["t3_score"] = None  # no opaque composite of missing theme/side fields
    record = build_result_record(job, parsed, scores, event.get("elapsed_ms", 0),
                                 {"raw_response": text})
    board = chess.Board(job["fen"])
    move = board.parse_san(parsed["move"]) if scores["t2_legal"] else None
    record.update({k: job[k] for k in ("parent_position_id", "setup_fen", "puzzle_id",
                                      "themes_all", "solution_uci", "search_eval") if k in job})
    record.update(ollama_response=response, done_reason=response.get("done_reason"),
                  request_error=event.get("error"), parse_errors=parsed["parse_errors"],
                  completed=event.get("event") == "completed", options=event.get("payload"),
                  solution_match=move is not None and move.uci() == job["solution_uci"],
                  random_solution_rate=1 / board.legal_moves.count())
    record["theme_available"] = bool(tactical)
    return record


def finite(value):
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool) and math.isfinite(value)


def mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(v)]
    return sum(values) / len(values) if values else None


def correlation(rows):
    pairs = [(r["t1_model_eval"], r["stockfish_eval"]) for r in rows
             if finite(r.get("t1_model_eval")) and finite(r.get("stockfish_eval"))]
    if len(pairs) < 3 or any(len(set(values)) < 2 for values in zip(*pairs)):
        return None
    return float(spearmanr(*zip(*pairs)).statistic)


def side_statistics(rows):
    valid = [r for r in rows if r.get("t3_side_claimed") in ("White", "Black", "Equal")]
    truth = [get_direction(r["stockfish_eval"]) for r in valid]
    predicted = [r["t3_side_claimed"] for r in valid]
    def solver_baseline(selected):
        return mean([get_direction(r["stockfish_eval"]) ==
                     ("White" if chess.Board(r["fen"]).turn else "Black") for r in selected])
    return {"side_n": len(valid), "side_accuracy": mean([a == b for a, b in zip(truth, predicted)]),
            "side_kappa": float(cohen_kappa(np.array(predicted), np.array(truth))) if valid else None,
            "majority_baseline": max((truth.count(v) / len(truth) for v in set(truth)), default=None),
            "side_to_move_baseline": solver_baseline(valid), "baseline_n": len(valid),
            "all_request_side_to_move_baseline": solver_baseline(rows),
            "all_request_baseline_n": len(rows)}


def describe(rows):
    legal = [r for r in rows if r.get("t2_legal") and finite(r.get("t2_cpl"))]
    t1 = [r for r in rows if finite(r.get("t1_model_eval")) and abs(r["stockfish_eval"]) < 10000]
    result = {"requests": len(rows), "first_answer_legal": mean([bool(r.get("t2_legal")) for r in rows]),
              "completion_rate": mean([bool(r.get("completed")) for r in rows]),
              "truncation_rate": mean([r.get("done_reason") == "length" for r in rows]),
              "parse_failure_rate": mean([bool(r.get("parse_errors")) for r in rows]),
              "request_error_rate": mean([bool(r.get("request_error")) for r in rows]),
              "score_coverage": len(legal) / len(rows) if rows else None,
              "theme_coverage": mean([bool(r.get("theme_available")) for r in rows]),
              "solution_match": mean([bool(r.get("solution_match")) for r in rows]),
              "random_solution_rate": mean([r["random_solution_rate"] for r in rows]),
              "legal_scored_n": len(legal), "cpl_mean": mean([r["t2_cpl"] for r in legal]),
              "blunder_rate": mean([r["t2_cpl"] >= BLUNDER for r in legal]),
              "legal_nonblunder_per_request": sum(r["t2_cpl"] < BLUNDER for r in legal) / len(rows) if rows else None,
              "t1_n": sum(finite(r.get("t1_model_eval")) for r in rows), "t1_rho": correlation(rows),
              "nonmate_mae": mean([abs(r["t1_model_eval"] - r["stockfish_eval"]) for r in t1]),
              "nonmate_zero_mae": mean([abs(r["stockfish_eval"]) for r in t1]),
              "theme_match": mean([r.get("t3_p2_theme_correct") for r in rows])}
    result.update(side_statistics(rows))
    for color, label in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        subset = [r for r in rows if chess.Board(r["fen"]).turn == color]
        result[label + "_to_move_rho"] = correlation(subset)
    return {key: value if not isinstance(value, float) or math.isfinite(value) else None
            for key, value in result.items()}


def bootstrap(rows, resamples=RESAMPLES):
    point = describe(rows)
    if not rows:
        return {"point": point, "ci": {}}
    rng = np.random.default_rng(SEED)
    draws = [describe([rows[i] for i in rng.integers(0, len(rows), len(rows))]) for _ in range(resamples)]
    intervals = {}
    for key in point:
        if key.endswith("_n") or key == "requests":
            continue
        values = [d[key] for d in draws if finite(d.get(key))]
        intervals[key] = [float(v) for v in np.quantile(values, [.025, .975])] if values else None
    return {"point": point, "ci": intervals, "resamples": resamples, "seed": SEED}


def local_endpoint(config):
    base = config.get("ollama", {}).get("base_url", "http://localhost:11434").rstrip("/")
    url = urlparse(base)
    if url.scheme != "http" or url.hostname not in ("localhost", "127.0.0.1", "::1") or url.path:
        raise ValueError("Only the local Ollama HTTP endpoint is permitted")
    return base


def idle_wait(config):
    from scripts.wait_for_ollama_idle import wait_until_idle
    wait_until_idle(local_endpoint(config), timeout=config["ollama"].get("idle_timeout", 600))


def engine(config):
    from src.engine_wrapper import Lc0Engine
    return Lc0Engine(path=config["lc0"]["binary"], weights=config["lc0"]["weights"], nodes=NODES)


def references(manifest, idle, make_engine):
    idle()
    import onnxruntime as ort
    from scripts.precompute_lc0_batch_v2 import encode_position, wdl_to_centipawns
    session = ort.InferenceSession(manifest["config"]["lc0"]["onnx_model"],
                                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    positions = []
    for position in manifest["positions"]:
        board = chess.Board(position["fen"])
        inputs = {session.get_inputs()[0].name: encode_position(board)[None]}
        output = session.run(None, inputs)
        # Base precompute uses output[1] WDL, not output[0] policy.
        cp = wdl_to_centipawns(output[1][0])
        positions.append({**position, "stockfish_eval": cp if board.turn else -cp})
    del session
    gc.collect()
    idle()
    search = make_engine()
    try:
        for position in positions:
            result = search.evaluate(position["fen"])
            position.update(stockfish_best_move=result["best_move"], search_eval=result["eval"], search_nodes=NODES)
    finally:
        search.close()
    return positions


def verify_reference_assets(manifest):
    for key in ("binary", "weights", "onnx_model"):
        path = Path(manifest["config"]["lc0"][key])
        if file_sha256(path) != manifest["sources"].get(str(path)):
            raise ValueError("Frozen engine/network reference asset changed")


def save_artifact(path, manifest, positions):
    if path.exists():
        raise ValueError(f"Stage artifact already exists: {path.name}")
    value = {"manifest_hash": manifest["manifest_hash"], "positions": positions}
    value["sha256"] = digest(value)
    atomic_json(path, value)


def load_artifact(path, manifest):
    value = json.loads(path.read_text())
    if value.get("manifest_hash") != manifest["manifest_hash"]:
        raise ValueError("Artifact manifest differs")
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("Artifact changed")
    positions = value["positions"]
    expected = {r["position_id"]: r["fen"] for r in manifest["positions"]}
    if "references" in path.name:
        if len(positions) != len(expected) or {r["position_id"]: r["fen"] for r in positions} != expected:
            raise ValueError("Reference positions differ from manifest")
        for row in positions:
            if not finite(row.get("stockfish_eval")) or not finite(row.get("search_eval")):
                raise ValueError("Missing/invalid actual-position reference")
            chess.Board(row["fen"]).parse_san(row["stockfish_best_move"])
            if row.get("search_nodes") != NODES:
                raise ValueError("Wrong search-reference budget")
    return positions


def score_answers(manifest, directory, idle, make_engine):
    refs = load_artifact(directory / "references.json", manifest)
    jobs = make_jobs(refs)
    path = directory / "raw.jsonl"
    verify_guard(path)
    done = read_ledger(path, {j["job_id"] for j in jobs}, manifest["manifest_hash"])
    if len(done) != len(jobs):
        raise ValueError("Collection must finish before engine scoring")
    records = [answer_record(job, done[job["job_id"]]) for job in jobs]
    idle()
    search = make_engine()
    cache = {}
    from scripts.enrich_cpl import compute_cpl
    try:
        for record in records:
            if not record["t2_legal"]:
                continue
            if record["t2_move"] == record["t2_best_move"]:
                record["t2_cpl"] = 0
                record["t2_cpl_eval_before"] = record["search_eval"]
                record["t2_cpl_nodes"] = NODES
                continue
            board = chess.Board(record["fen"])
            board.push_san(record["t2_move"])
            if board.fen() not in cache:
                cache[board.fen()] = search.evaluate(board.fen())["eval"]
            record["t2_cpl"] = compute_cpl(record["fen"], record["search_eval"], cache[board.fen()])
            record["t2_cpl_eval_before"] = record["search_eval"]
            record["t2_cpl_nodes"] = NODES
    finally:
        search.close()
    return records


def measure_pairs(pairs, measure):
    def usable(row):
        if measure == "blunder_rate":
            return bool(row.get("t2_legal")) and finite(row.get("t2_cpl"))
        if measure == "t1_rho":
            return finite(row.get("t1_model_eval")) and finite(row.get("stockfish_eval"))
        if measure == "side_kappa":
            return row.get("t3_side_claimed") in ("White", "Black", "Equal")
        return True
    return [(left, right) for left, right in pairs if usable(left) and usable(right)]


def paired_difference(pairs):
    result = {}
    for measure in ("first_answer_legal", "solution_match", "legal_nonblunder_per_request",
                    "blunder_rate", "t1_rho", "side_kappa"):
        selected = measure_pairs(pairs, measure)
        left, right = describe([v[0] for v in selected]), describe([v[1] for v in selected])
        a, b = left[measure], right[measure]
        result[measure] = b - a if finite(a) and finite(b) else None
    return result


def paired_effect(left, right):
    by_id = {r["position_id"]: r for r in right}
    pairs = [(row, by_id[row["position_id"]]) for row in left if row["position_id"] in by_id]
    point = paired_difference(pairs)
    counts = {key: len(measure_pairs(pairs, key)) for key in point}
    if not pairs:
        return {"n_pairs": 0, "measure_n_pairs": counts, "difference_board_minus_fen": point, "ci": {}}
    rng = np.random.default_rng(SEED)
    draws = [paired_difference([pairs[i] for i in rng.integers(0, len(pairs), len(pairs))]) for _ in range(RESAMPLES)]
    ci = {}
    for key in point:
        values = [r[key] for r in draws if finite(r[key])]
        ci[key] = list(map(float, np.quantile(values, [.025, .975]))) if values else None
    return {"n_pairs": len(pairs), "measure_n_pairs": counts,
            "difference_board_minus_fen": point, "ci": ci}


def report(manifest, records):
    groups = {}
    effects = {}
    for model in MODELS:
        selected = [r for r in records if r["model"] == model]
        by_format = {fmt: [r for r in selected if r["prompt_format"] == fmt] for fmt in FORMATS}
        groups[model] = {fmt: bootstrap(rows) for fmt, rows in by_format.items()}
        effects[model] = paired_effect(by_format["fen_only"], by_format["board_text"])
    return {"manifest_hash": manifest["manifest_hash"], "groups": groups, "paired_formats": effects,
            "setup_controls": setup_controls(manifest, records),
            "interpretation": "Exploratory first solver move only; not full puzzle solving or a reasoning mechanism.",
            "missing_requests": len(manifest["positions"]) * len(MODELS) * len(FORMATS) - len(records)}


def setup_controls(manifest, records):
    parent_path = Path(manifest["config"]["parent_results"])
    if file_sha256(parent_path) != manifest["sources"][str(parent_path)]:
        raise ValueError("Setup-control source changed")
    wanted = {(r["model"], r["prompt_format"], r["parent_position_id"]): r for r in records}
    output = {m: {fmt: [] for fmt in FORMATS} for m in MODELS}
    seen = set()
    for row in read_jsonl(parent_path):
        key = (row.get("model"), row.get("prompt_format"), row.get("position_id"))
        if key not in wanted:
            continue
        if key in seen:
            raise ValueError("Duplicate matched setup control")
        seen.add(key)
        actual = wanted[key]
        raw = row.get("raw_response")
        parsed = parse_response(raw or "")
        try:
            chess.Board(row["fen"]).parse_san(parsed.get("move") or "")
            legal = True
        except ValueError:
            legal = False if isinstance(raw, str) else None
        output[key[0]][key[1]].append({"parent_position_id": key[2], "setup_first_legal": legal,
            "actual_first_legal": bool(actual.get("t2_legal")), "setup_move": parsed.get("move"),
            "actual_move": actual.get("t2_move"), "setup_fen": row["fen"], "actual_fen": actual["fen"]})
    return {"matched": output, "missing_controls": len(wanted) - len(seen),
            "caveat": "Board position and turn changed; descriptive, not a causal notation effect."}


def validate_config(config):
    for key, expected in (("models", list(MODELS)), ("formats", list(FORMATS)),
                          ("seed", SEED), ("positions_per_tier", PER_TIER), ("resamples", RESAMPLES)):
        if config.get(key) != expected:
            raise ValueError(f"Fixed study setting changed: {key}")
    if config["lc0"]["nodes"] != NODES:
        raise ValueError("Reference search must use 800 nodes")
    local_endpoint(config)


def collect_stage(manifest, directory, config):
    positions = load_artifact(directory / "references.json", manifest)
    def send(payload):
        reply = requests.post(local_endpoint(config) + "/api/chat", json=payload,
                              timeout=config["ollama"]["timeout"])
        reply.raise_for_status()
        try:
            value = reply.json()
        except ValueError:
            return {"_validation_error": "Invalid JSON", "_raw_http_body": reply.text}
        if (not isinstance(value, dict) or not isinstance(value.get("message"), dict)
                or not isinstance(value["message"].get("content"), str) or value.get("done") is not True):
            return {"_validation_error": "Malformed/incomplete reply", "_raw_http_body": reply.text}
        value["_raw_http_body"] = reply.text
        value["_http_status"] = reply.status_code
        return value
    collect_jobs(make_jobs(positions), directory / "raw.jsonl", manifest["manifest_hash"],
                 send, lambda: idle_wait(config))


def run_stage(stage, directory, config, config_path):
    manifest_path = directory / "manifest.json"
    if stage == "prepare-manifest":
        if any(directory.iterdir()) and set(p.name for p in directory.iterdir()) != {".lock"}:
            raise ValueError("Prepare requires an empty isolated output directory")
        atomic_json(manifest_path, prepare_manifest(config, config_path))
        return
    manifest = load_manifest(manifest_path)
    if manifest["config"] != config or len(manifest["positions"]) != PER_TIER * len(TIERS):
        raise ValueError("Config/manifest no longer matches fixed plan")
    if manifest.get("code_hashes") != code_hashes():
        raise ValueError("Frozen study implementation changed; review before continuing")
    idle, make_engine = lambda: idle_wait(config), lambda: engine(config)
    if stage == "reference":
        if (directory / "references.json").exists():
            raise ValueError("Reference stage already completed")
        verify_reference_assets(manifest)
        save_artifact(directory / "references.json", manifest, references(manifest, idle, make_engine))
    elif stage == "collect":
        collect_stage(manifest, directory, config)
    elif stage == "score":
        if (directory / "evaluations.json").exists():
            raise ValueError("Scoring stage already completed")
        verify_reference_assets(manifest)
        save_artifact(directory / "evaluations.json", manifest, score_answers(manifest, directory, idle, make_engine))
    else:
        records = load_artifact(directory / "evaluations.json", manifest)
        atomic_json(directory / "summary.json", report(manifest, records))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare-manifest", "reference", "collect", "score", "report"))
    parser.add_argument("--config", default="config/config_actual_puzzle_followup.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    directory = output_directory(config)
    if args.dry_run:
        print(json.dumps({"stage": args.stage, "positions": 100, "initial_requests": 600,
                          "models": MODELS, "formats": FORMATS, "output_dir": str(directory),
                          "network_or_writes": False}))
        return
    validate_config(config)
    os.chdir(ROOT)
    directory.mkdir(parents=True, exist_ok=True)
    with execution_lock(directory):
        run_stage(args.stage, directory, config, args.config)


if __name__ == "__main__":
    main()
