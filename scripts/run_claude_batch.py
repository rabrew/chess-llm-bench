"""Run the benchmark on Claude models through the Message Batches API (specs/commercial-models.md).

    python scripts/run_claude_batch.py --dry-run                      # count and cost, no API call
    python scripts/run_claude_batch.py --positions-per-tier 2         # pilot: 96 requests
    python scripts/run_claude_batch.py --variants claude-haiku:5.5    # one variant, all positions
    python scripts/run_claude_batch.py --status                       # spend and token use so far

Resumable: stored replies are never resent, and an open batch is collected before anything new.
The API key comes from ANTHROPIC_API_KEY or the key file named in the config (outside the repo).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import claude_batch as cb
from src.data_loader import DataLoader
from src.utils import load_config, setup_logging

logger = logging.getLogger("chess_llm_bench")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Claude models through the Message Batches API.")
    parser.add_argument("--config", default="config/config_claude.yaml")
    parser.add_argument("--positions-per-tier", type=int, default=0,
                        help="Only the first N sampled positions of each tier (pilot); 0 = all 125")
    parser.add_argument("--variants", default="", help="Comma-separated variant tags; default all")
    parser.add_argument("--dry-run", action="store_true", help="Count requests and estimate cost; no API call")
    parser.add_argument("--status", action="store_true", help="Print spend and token use so far")
    return parser.parse_args()


def chosen_variants(variants: list[cb.Variant], wanted: str) -> list[cb.Variant]:
    tags = [tag.strip() for tag in wanted.split(",") if tag.strip()]
    unknown = set(tags) - {v.tag for v in variants}
    if unknown:
        raise SystemExit(f"Unknown variants {sorted(unknown)}; choose from {[v.tag for v in variants]}")
    return [v for v in variants if not tags or v.tag in tags]


def print_status(store: cb.ResponseStore, budget: float) -> None:
    stored = store.load()
    print(json.dumps(cb.summarise(stored), indent=2))
    print(f"Spent ${cb.spent_usd(stored.values()):.2f} of ${budget:.2f}")


def run_rounds(client: Any, jobs: list[dict], variants: dict[str, cb.Variant], lookup: dict, store: cb.ResponseStore,
               ledger: cb.BatchLedger, claude: dict[str, Any]) -> None:
    """Round 1 (first answers), then round 2 (re-asks), each behind the budget guard."""
    for round_no, pending_for in ((cb.FIRST_ROUND, cb.first_round_pending), (cb.RETRY_ROUND, cb.retry_pending)):
        stored = store.load()
        pending = pending_for(jobs, variants, stored)
        estimate = cb.estimate_usd(cb.count_by_variant(pending, lookup), list(stored.values()), claude)
        cb.check_budget(cb.spent_usd(stored.values()), estimate, claude["budget_usd"])
        logger.info(f"Round {round_no}: {len(pending)} requests, estimated ${estimate:.2f}")
        cb.run_round(client, pending, round_no, ledger, store, lookup, claude["prices_per_mtok"],
                     claude["batch_discount"], claude["poll_seconds"])


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(args.config)
    claude, paths = config["claude"], config["paths"]
    formats = config["benchmark"]["prompt_formats"]
    all_variants = cb.load_variants(config)
    variants = chosen_variants(all_variants, args.variants)
    store, ledger = cb.ResponseStore(paths["responses_file"]), cb.BatchLedger(paths["batches_file"])
    if args.status:
        print_status(store, claude["budget_usd"])
        return

    positions = cb.sample_positions(config, DataLoader(paths["data_dir"]))
    lookup = cb.request_lookup(cb.build_jobs(positions, all_variants, formats), all_variants)
    jobs = cb.build_jobs(cb.pilot_positions(positions, args.positions_per_tier), variants, formats)
    by_tag = {v.tag: v for v in all_variants}

    if args.dry_run:
        stored = store.load()
        pending = cb.first_round_pending(jobs, by_tag, stored)
        estimate = cb.estimate_usd(cb.count_by_variant(pending, lookup), list(stored.values()), claude)
        print(f"{len(jobs)} jobs, {len(pending)} first-round requests to send, estimated ${estimate:.2f} "
              f"(spent so far ${cb.spent_usd(stored.values()):.2f} of ${claude['budget_usd']:.2f}); "
              f"re-asks come on top")
        return

    try:
        api_key = cb.load_api_key(os.environ, claude["key_file"])
    except cb.MissingApiKey as error:
        raise SystemExit(str(error))
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    eval_range = tuple(config["evaluation"]["centipawn_eval_range"][k] for k in ("min", "max"))
    try:
        cb.collect_open_batches(client, ledger, store, lookup, claude["prices_per_mtok"], claude["batch_discount"],
                                claude["poll_seconds"])
        run_rounds(client, jobs, by_tag, lookup, store, ledger, claude)
    except cb.BudgetExceeded as error:
        logger.error(str(error))
    written, failed = cb.write_finished(jobs, by_tag, store.load(), paths["results_file"], paths["failures_file"],
                                        eval_range, claude["prices_per_mtok"], claude["batch_discount"])
    logger.info(f"Wrote {written} records and {failed} failures")
    print_status(store, claude["budget_usd"])


if __name__ == "__main__":
    main()
