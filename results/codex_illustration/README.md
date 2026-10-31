# GPT illustration via Codex (not part of the benchmark)

**Illustrative only.** 12 positions answered by GPT through Ryan's free Codex (ChatGPT) subscription,
31 Oct 2026. It shows how a second frontier model behaves; it is not a benchmark result.

- **Positions:** 3 per tier, `random.Random(42)` from the 500 plain-board positions (`make_pack.py`).
- **Prompt:** the `fen_only` combined prompt, exactly as the other models got it. Re-ask for the 2
  illegal first moves: the worker's `retry_move_prompt`, system prompt and prompt joined into one
  message (Codex has no separate system prompt).
- **How it ran:** `codex exec` (CLI 0.166.0), model reported as `gpt-6.1-sol`, from an empty folder,
  read-only sandbox, **every tool disabled** (`run_one.sh`). The first test, before tools were
  disabled, tried to run Stockfish on the Mac; it was discarded. `raw/r*.jsonl` (answers) and
  `raw/rr*.jsonl` (re-asks) are the event logs: none contains a tool call. `raw/test.jsonl` is the discarded first test (it shows the attempted Stockfish call); `raw/test2.jsonl` the first clean run.
- **Scoring:** the worker's rescue and `score_all` (`score.py`, `compare.py`, run on the PC).

| Model (same 12 positions, FEN) | First answer legal | Legal after re-ask | Better side correct |
|---|---|---|---|
| Claude Opus 5.5 (low effort) | 12/12 | 12/12 | 7/12 |
| GPT (`gpt-6.1-sol`, Codex) | 10/12 | 12/12 | 6/12 |
| Claude Haiku 5.5 | 10/12 | 12/12 | 5/12 |
| Local models (13) | 0–3/12 | 10–12/12 | 0–4/12 |

12 positions are too few for the evaluation correlation (ρ): it is noise at this size.
Limitations: a coding-agent wrapper around the model, a consumer subscription, the model version
whatever the subscription served, no sampling control. Bulk runs this way are not appropriate.
