"""Claude models through the Message Batches API (specs/commercial-models.md).

The prompts, the move rescue and the scoring are the worker's (src/worker.py, combined prompts).
Only the transport differs: requests go out in batches at half price, and every reply is kept in
an append-only store, which is also the resume cache.
"""

import hashlib
import json
import logging
import re
import stat
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluator import score_all
from .job_generator import generate_job_id, subsample_per_tier
from .llm_client import build_prompt, extract_move_from_text, parse_response, retry_move_prompt
from .metrics import _is_legal_san
from .result_writer import ResultWriter, build_result_record

logger = logging.getLogger("chess_llm_bench")

TIERS = ("easy", "medium", "hard", "extreme")
FIRST_ROUND = 1
RETRY_ROUND = 2
CUSTOM_ID_MAX = 64  # API limit
SAFE_ID = re.compile(r"[A-Za-z0-9_-]+")
UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_-]")
HASH_CHARS = 16
TOKENS_PER_MTOK = 1_000_000
API_KEY_VAR = "ANTHROPIC_API_KEY"
EXPORT_PREFIX = "export "
GROUP_OTHER_BITS = stat.S_IRWXG | stat.S_IRWXO
BATCH_ENDED = "ended"
SUCCEEDED = "succeeded"
PARSED_FIELDS = ("eval", "move", "explanation")


class BudgetExceeded(RuntimeError):
    """Submitting the next batch could take spending past the budget."""


class MissingApiKey(RuntimeError):
    """No API key in the environment or the key file."""


@dataclass(frozen=True)
class Variant:
    """One Claude configuration: the tag records carry, and the request settings behind it."""

    tag: str
    model: str
    max_tokens: int
    thinking: str | None = None
    effort: str | None = None


def load_variants(config: dict[str, Any]) -> list[Variant]:
    return [Variant(**variant) for variant in config["claude"]["variants"]]


# ---------- positions and jobs ----------

def sample_positions(config: dict[str, Any], data_loader: Any) -> list[dict[str, Any]]:
    """The positions run_all.sh draws: populate_job_queue one tier at a time, then the second draw."""
    bench = config["benchmark"]
    seed = bench["random_seed"]
    per_tier = bench.get("max_positions_per_tier", 0)
    positions = []
    for tier in TIERS:
        tier_positions = data_loader.load_tier(tier)
        if per_tier:
            tier_positions = data_loader.sample(tier_positions, count=per_tier, seed=seed)
        positions += subsample_per_tier(tier_positions, bench.get("sample_per_tier", 0), seed)
    return positions


def pilot_positions(positions: list[dict[str, Any]], per_tier: int) -> list[dict[str, Any]]:
    """The first `per_tier` positions of each tier, in sample order; 0 keeps all."""
    if not per_tier:
        return positions
    taken: Counter[str] = Counter()
    pilot = []
    for position in positions:
        if taken[position["difficulty"]] < per_tier:
            taken[position["difficulty"]] += 1
            pilot.append(position)
    return pilot


def build_jobs(positions: list[dict[str, Any]], variants: list[Variant], formats: list[str]) -> list[dict[str, Any]]:
    """One job per variant x position x format, with the fields the worker adds from the dataset."""
    return [
        {
            "job_id": generate_job_id(pos["id"], variant.tag, fmt),
            "job_type": "standard",
            "trial": 1,
            "parent_job_id": None,
            "position_id": pos["id"],
            "model": variant.tag,
            "prompt_format": fmt,
            "fen": pos["fen"],
            "pgn_moves": pos.get("pgn_moves", ""),
            "difficulty": pos.get("difficulty"),
            "phase": pos.get("phase"),
            "source": pos.get("source"),
            "theme": pos.get("theme", ""),
            "stockfish_eval": pos.get("stockfish_eval", 0),
            "stockfish_best_move": pos.get("stockfish_best_move", ""),
        }
        for variant in variants
        for pos in positions
        for fmt in formats
    ]


# ---------- requests ----------

def custom_id(job_id: str, round_no: int) -> str:
    """The job ID in the API's allowed characters, plus the round; hashed if too long."""
    candidate = f"{UNSAFE_CHARS.sub('_', job_id)}-r{round_no}"
    if len(candidate) <= CUSTOM_ID_MAX:
        return candidate
    return f"h{hashlib.sha1(job_id.encode()).hexdigest()[:HASH_CHARS]}-r{round_no}"


def request_params(variant: Variant, prompt: str, system: str | None = None) -> dict[str, Any]:
    """Messages API parameters. No sampling settings: current Claude models reject them."""
    params: dict[str, Any] = {
        "model": variant.model,
        "max_tokens": variant.max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        params["system"] = system
    if variant.thinking:
        params["thinking"] = {"type": variant.thinking}
    if variant.effort:
        params["output_config"] = {"effort": variant.effort}
    return params


def first_round_request(job: dict[str, Any], variant: Variant) -> dict[str, Any]:
    prompt = build_prompt(fen=job["fen"], pgn_moves=job.get("pgn_moves"), prompt_format=job["prompt_format"])
    return {"custom_id": custom_id(job["job_id"], FIRST_ROUND), "params": request_params(variant, prompt)}


def retry_request(job: dict[str, Any], variant: Variant) -> dict[str, Any]:
    prompt, system = retry_move_prompt(job["fen"], job["prompt_format"])
    return {"custom_id": custom_id(job["job_id"], RETRY_ROUND), "params": request_params(variant, prompt, system)}


def request_lookup(jobs: list[dict[str, Any]], variants: list[Variant]) -> dict[str, tuple[dict, Variant, int]]:
    """custom ID -> (job, variant, round), for both rounds of every job."""
    by_tag = {variant.tag: variant for variant in variants}
    return {
        custom_id(job["job_id"], round_no): (job, by_tag[job["model"]], round_no)
        for job in jobs
        for round_no in (FIRST_ROUND, RETRY_ROUND)
    }


# ---------- the worker's move rescue ----------

def first_answer(job: dict[str, Any], text: str) -> tuple[dict[str, Any], bool]:
    """Parse the combined reply; rescue a legal move from its text. True if a re-ask is needed."""
    parsed = parse_response(text)
    if _is_legal_san(job["fen"], parsed.get("move")):
        return parsed, False
    rescued = extract_move_from_text(job["fen"], text)
    if rescued:
        parsed["move"] = rescued
        return parsed, False
    return parsed, True


def apply_retry(parsed: dict[str, Any], fen: str, retry_text: str | None) -> dict[str, Any]:
    """The re-ask's first legal move, or None: the original illegal move never reaches the scorer."""
    parsed["move"] = extract_move_from_text(fen, retry_text) if retry_text and retry_text.strip() else None
    return parsed


# ---------- cost and budget ----------

def request_cost(input_tokens: float, output_tokens: float, price: Mapping[str, float], discount: float) -> float:
    return (input_tokens * price["input"] + output_tokens * price["output"]) / TOKENS_PER_MTOK * discount


def spent_usd(rows: Iterable[Mapping[str, Any]]) -> float:
    return sum(row.get("cost_usd", 0.0) for row in rows)


def _mean_tokens(rows: list[Mapping[str, Any]], claude_config: Mapping[str, Any]) -> tuple[float, float]:
    if len(rows) >= claude_config["min_samples_for_estimate"]:
        return (sum(r["input_tokens"] for r in rows) / len(rows), sum(r["output_tokens"] for r in rows) / len(rows))
    defaults = claude_config["default_estimate_tokens"]
    return defaults["input"], defaults["output"]


def estimate_usd(counts: Mapping[Variant, int], rows: list[Mapping[str, Any]], claude_config: Mapping[str, Any]) -> float:
    """Expected cost of `counts` requests per variant, from stored token use once there is enough."""
    total = 0.0
    for variant, count in counts.items():
        mean_in, mean_out = _mean_tokens([r for r in rows if r.get("tag") == variant.tag], claude_config)
        total += request_cost(count * mean_in, count * mean_out, claude_config["prices_per_mtok"][variant.model],
                              claude_config["batch_discount"])
    return total


def check_budget(spent: float, estimate: float, budget: float) -> None:
    if spent + estimate > budget:
        raise BudgetExceeded(
            f"Spent ${spent:.2f} + this batch's estimate ${estimate:.2f} would exceed the ${budget:.2f} budget"
        )


def count_by_variant(requests: list[dict[str, Any]], lookup: Mapping[str, tuple]) -> Counter:
    return Counter(lookup[request["custom_id"]][1] for request in requests)


# ---------- storage ----------

class ResponseStore:
    """Append-only JSONL of API results: the raw record of the run and its resume cache."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        with open(self.path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return {row["custom_id"]: row for row in rows}

    def append(self, rows: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


class BatchLedger:
    """Batches submitted but not yet collected, so a restart polls them instead of paying twice."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def open_batches(self) -> list[dict[str, Any]]:
        return json.loads(self.path.read_text()) if self.path.exists() else []

    def _save(self, batches: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(batches, indent=2) + "\n")

    def add(self, batch_id: str, round_no: int) -> None:
        self._save(self.open_batches() + [{"id": batch_id, "round": round_no}])

    def remove(self, batch_id: str) -> None:
        self._save([batch for batch in self.open_batches() if batch["id"] != batch_id])


# ---------- batches ----------

def reply_text(message: Any) -> str:
    return "".join(block.text for block in message.content if block.type == "text")


def stored_row(result: Any, batch_id: str, lookup: Mapping[str, tuple], prices: Mapping[str, Mapping[str, float]],
               discount: float) -> dict[str, Any] | None:
    """A storable row for a succeeded result; None (logged) for errored, expired or cancelled ones."""
    if result.result.type != SUCCEEDED:
        logger.warning(f"{result.custom_id}: {result.result.type}; it will be sent again on the next run")
        return None
    if result.custom_id not in lookup:
        logger.warning(f"{result.custom_id}: not a job of this config; skipped")
        return None
    job, variant, round_no = lookup[result.custom_id]
    message = result.result.message
    usage = message.usage
    return {
        "custom_id": result.custom_id,
        "job_id": job["job_id"],
        "round": round_no,
        "tag": variant.tag,
        "api_model": message.model,
        "text": reply_text(message),
        "stop_reason": message.stop_reason,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": request_cost(usage.input_tokens, usage.output_tokens, prices[variant.model], discount),
        "batch_id": batch_id,
    }


def wait_for_batch(client: Any, batch_id: str, poll_seconds: float, sleep: Callable[[float], None]) -> None:
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == BATCH_ENDED:
            return
        logger.info(f"{batch_id}: {batch.processing_status}, {batch.request_counts.processing} still processing")
        sleep(poll_seconds)


def collect_batch(client: Any, batch_id: str, store: ResponseStore, lookup: Mapping[str, tuple],
                  prices: Mapping[str, Mapping[str, float]], discount: float) -> int:
    rows = [stored_row(result, batch_id, lookup, prices, discount) for result in client.messages.batches.results(batch_id)]
    kept = [row for row in rows if row is not None]
    store.append(kept)
    logger.info(f"{batch_id}: stored {len(kept)} of {len(rows)} results")
    return len(kept)


def _finish(client: Any, batch_id: str, ledger: BatchLedger, store: ResponseStore, lookup: Mapping[str, tuple],
            prices: Mapping[str, Mapping[str, float]], discount: float, poll_seconds: float,
            sleep: Callable[[float], None]) -> None:
    """Wait for the batch, store its results, then drop it from the ledger (in that order)."""
    wait_for_batch(client, batch_id, poll_seconds, sleep)
    collect_batch(client, batch_id, store, lookup, prices, discount)
    ledger.remove(batch_id)


def collect_open_batches(client: Any, ledger: BatchLedger, store: ResponseStore, lookup: Mapping[str, tuple],
                         prices: Mapping[str, Mapping[str, float]], discount: float, poll_seconds: float,
                         sleep: Callable[[float], None] = time.sleep) -> None:
    for batch in ledger.open_batches():
        logger.info(f"Collecting open batch {batch['id']} (round {batch['round']})")
        _finish(client, batch["id"], ledger, store, lookup, prices, discount, poll_seconds, sleep)


def run_round(client: Any, requests: list[dict[str, Any]], round_no: int, ledger: BatchLedger, store: ResponseStore,
              lookup: Mapping[str, tuple], prices: Mapping[str, Mapping[str, float]], discount: float,
              poll_seconds: float, sleep: Callable[[float], None] = time.sleep) -> None:
    """Submit one batch, record it before waiting, then collect it."""
    if not requests:
        return
    batch = client.messages.batches.create(requests=requests)
    ledger.add(batch.id, round_no)
    logger.info(f"Submitted {batch.id}: {len(requests)} round-{round_no} requests")
    _finish(client, batch.id, ledger, store, lookup, prices, discount, poll_seconds, sleep)


def first_round_pending(jobs: list[dict[str, Any]], variants: Mapping[str, Variant],
                        stored: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [first_round_request(job, variants[job["model"]]) for job in jobs
            if custom_id(job["job_id"], FIRST_ROUND) not in stored]


def retry_pending(jobs: list[dict[str, Any]], variants: Mapping[str, Variant],
                  stored: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Re-asks for stored first answers whose move couldn't be rescued, not yet sent."""
    pending = []
    for job in jobs:
        first = stored.get(custom_id(job["job_id"], FIRST_ROUND))
        if first is None or custom_id(job["job_id"], RETRY_ROUND) in stored:
            continue
        if first_answer(job, first["text"])[1]:
            pending.append(retry_request(job, variants[job["model"]]))
    return pending


# ---------- records ----------

def build_record(job: dict[str, Any], variant: Variant, first: Mapping[str, Any], retry: Mapping[str, Any] | None,
                 eval_range: tuple[int, int], prices: Mapping[str, Mapping[str, float]],
                 discount: float) -> dict[str, Any] | None:
    """The worker's scored record plus the Claude fields; None if the reply had no field at all."""
    parsed, needs_retry = first_answer(job, first["text"])
    retry_text = None
    if needs_retry:
        retry_text = (retry or {}).get("text", "")
        parsed = apply_retry(parsed, job["fen"], retry_text)
    if all(parsed.get(field) is None for field in PARSED_FIELDS):
        return None
    position = {"id": job["position_id"], "fen": job["fen"], "stockfish_eval": job["stockfish_eval"],
                "stockfish_best_move": job["stockfish_best_move"], "theme": job["theme"]}
    scores = score_all(parsed_response=parsed, position=position, engine=None, eval_range=eval_range)
    record = build_result_record(job=job, parsed_response=parsed, scores=scores, inference_ms=None,
                                 raw_responses={"raw_response": first["text"], "raw_move_retry_response": retry_text})
    used = [first] + ([retry] if needs_retry and retry else [])
    input_tokens = sum(row.get("input_tokens", 0) for row in used)
    output_tokens = sum(row.get("output_tokens", 0) for row in used)
    record.update({
        "api_model": first.get("api_model"),
        "effort": variant.effort,
        "thinking": variant.thinking,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": request_cost(input_tokens, output_tokens, prices[variant.model], discount),
        "stop_reason": first.get("stop_reason"),
    })
    return record


def _job_ids_in(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path) as f:
        return {json.loads(line)["job_id"] for line in f if line.strip()}


def write_finished(jobs: list[dict[str, Any]], variants: Mapping[str, Variant], stored: Mapping[str, Any],
                   results_path: str | Path, failures_path: str | Path, eval_range: tuple[int, int],
                   prices: Mapping[str, Mapping[str, float]], discount: float) -> tuple[int, int]:
    """Score every job whose replies are all in, once. Returns (records written, failures written)."""
    results_path, failures_path = Path(results_path), Path(failures_path)
    done = _job_ids_in(results_path) | _job_ids_in(failures_path)
    records, failures = [], []
    for job in jobs:
        first = stored.get(custom_id(job["job_id"], FIRST_ROUND))
        if job["job_id"] in done or first is None:
            continue
        retry = stored.get(custom_id(job["job_id"], RETRY_ROUND))
        if first_answer(job, first["text"])[1] and retry is None:
            continue
        record = build_record(job, variants[job["model"]], first, retry, eval_range, prices, discount)
        if record is None:
            failures.append({"job_id": job["job_id"], "model": job["model"], "position_id": job["position_id"],
                             "prompt_format": job["prompt_format"], "raw_response": first["text"]})
        else:
            records.append(record)
    if records:
        ResultWriter(str(results_path)).write_results(records)
    if failures:
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failures_path, "a") as f:
            f.writelines(json.dumps(row) + "\n" for row in failures)
    return len(records), len(failures)


def summarise(stored: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per variant: results stored, mean tokens per request, spend, stop reasons."""
    summary: dict[str, dict[str, Any]] = {}
    for row in stored.values():
        entry = summary.setdefault(row["tag"], {"results": 0, "input_tokens": 0, "output_tokens": 0,
                                                "cost_usd": 0.0, "stop_reasons": Counter()})
        entry["results"] += 1
        entry["input_tokens"] += row["input_tokens"]
        entry["output_tokens"] += row["output_tokens"]
        entry["cost_usd"] += row["cost_usd"]
        entry["stop_reasons"][row["stop_reason"]] += 1
    for entry in summary.values():
        entry["mean_input_tokens"] = entry["input_tokens"] / entry["results"]
        entry["mean_output_tokens"] = entry["output_tokens"] / entry["results"]
        entry["stop_reasons"] = dict(entry["stop_reasons"])
    return summary


# ---------- API key ----------

def _key_from_file(path: Path) -> str | None:
    for line in path.read_text().splitlines():
        line = line.strip().removeprefix(EXPORT_PREFIX)
        if not line or line.startswith("#"):
            continue
        name, sep, value = line.partition("=")
        if sep and name.strip() == API_KEY_VAR and value.strip():
            return value.strip().strip("\"'")
    return None


def load_api_key(env: Mapping[str, str], key_file: str | Path) -> str:
    """The key from the environment, else from the key file outside the repo. Never logged."""
    if env.get(API_KEY_VAR):
        return env[API_KEY_VAR]
    path = Path(key_file).expanduser()
    if not path.exists():
        raise MissingApiKey(f"No {API_KEY_VAR} in the environment and no key file at {path}. "
                            f"Create it with one line: {API_KEY_VAR}=<your key>, then chmod 600 {path}")
    if path.stat().st_mode & GROUP_OTHER_BITS:
        logger.warning(f"{path} can be read by other users: run chmod 600 {path}")
    key = _key_from_file(path)
    if key is None:
        raise MissingApiKey(f"{path} has no line {API_KEY_VAR}=<your key>")
    return key
