# Spec: Claude Models Through the Batch API

**Created:** 2026-05-06 (first draft, never built). **Rewritten:** 2026-10-31, before any Claude results.
**Author:** Ryan Brew

---

## Goal

Every model tested so far is open-source and run locally (2B–70B). The obvious question from a
reader is whether frontier commercial models behave differently. This experiment runs four Claude
configurations on the **same 500 positions and the same three prompt formats** as the plain-board
run (`specs/plain-board-prompts.md`), so they pair position by position with the 21 local models.

It also tests something the local models can't: the **same model with thinking off and on**
(Claude Sonnet 5.5). If step-by-step reasoning before answering makes the moves better, that is
direct evidence on the project's question.

The budget is about €20 of API credit, so requests go through the **Message Batches API** (half
price) and a budget guard refuses to submit anything that would exceed it.

## Success criteria (fixed before running)

All comparisons pair records on (model, position, prompt format) and use seeded bootstrap 95% CIs
over positions (1,000 resamples, seed 42), as in `scripts/plain_board_compare.py`. "Clearly" means
the gap meets the size stated **and** the two CIs don't overlap.

| # | Question | Comparison (`fen_only` unless stated) | Finding if… |
|---|---|---|---|
| C1 | Do frontier models play better than the best local model? | Each Claude variant against gemma4:31b (best local blunder rate) on the same positions | Blunder rate clearly lower by ≥ 5 points → "frontier model plays clearly better" |
| C2 | Does thinking improve move choice? | Sonnet 5.5 thinking on against off | Blunder rate clearly lower by ≥ 5 points → "thinking improves move choice" |
| C3 | Does thinking improve judgement? | Same pair | T1 ρ higher by ≥ 0.10 with non-overlapping CIs → "thinking improves evaluation" |
| C4 | Is notation a barrier for Claude? | `board_text` against `fen_only`, per variant | The plain-board rule (N1–N4) applied unchanged |

Anything that doesn't meet its bar is reported as "no clear difference", with the numbers. Also
reported, not judged: first-answer legality, best-move rate against random, side κ, cost per variant.

## Pilot result and change of plan (2026-10-31, before the full run)

The pilot (8 positions × 3 formats × 4 variants, 96 requests, $0.48) changed two things. Its
records stay in the results; nothing about the criteria above was changed after seeing them.

- **Cost.** Measured per request: Haiku $0.0006, Sonnet off $0.0051, Sonnet on $0.0076, Opus $0.0067.
  All four variants on all formats would cost about $30, over the budget.
- **"Thinking off" is not "no reasoning".** With thinking off, Sonnet writes its reasoning into the
  visible reply instead. 11 of 24 replies hit the 1,024-token limit (most still had the eval and
  move before the cut). So C2/C3 would compare hidden thinking with reasoning written in the reply.

**Decision (Ryan):** the full run now covers **Haiku 5.5 and Opus 5.5 (low effort), all three
formats** (about $11). Both Sonnet variants are deferred to a later run, so **C2 and C3 are not
tested now**; C1 and C4 apply as written. For that later run, `claude-sonnet:5.5-nothink` gets
`max_tokens: 3000` so replies aren't cut off, and its 24 pilot replies (1,024-token limit) are
re-sent rather than reused.

## Inputs / Outputs

**Inputs**
- The plain-board sample: 500 positions (125 per tier), drawn exactly as `run_all.sh` draws them
  (`populate_job_queue` per tier, `subsample_per_tier`, seed 42), so position IDs match the local runs.
- `config/config_claude.yaml`: variants, prices, budget, paths.
- The API key, from the environment variable `ANTHROPIC_API_KEY`, or from a key file **outside the
  repository** (default `~/.config/chess-llm-bench/anthropic.env`, one line `ANTHROPIC_API_KEY=...`).
  The key is never written to config, logs, records or git.

**Outputs** (`results/claude/`)
- `responses.jsonl`: every API result as returned (custom ID, round, variant, reply text, stop reason,
  token usage, batch ID). This is the raw record and the resume cache.
- `batches.json`: batches submitted but not yet collected, so a restart polls them instead of paying twice.
- `evaluations.jsonl`: one scored record per job, in the usual schema, plus `api_model`, `effort`,
  `thinking`, `input_tokens`, `output_tokens`, `cost_usd`, `stop_reason`.
- `failures.jsonl`: jobs whose reply had no eval, move or explanation at all (as the worker drops them).

## Variants

| Tag (in records) | API model | Thinking | Effort | `max_tokens` | Why |
|---|---|---|---|---|---|
| `claude-haiku:5.5` | `claude-haiku-5-5` | default (adaptive) | default (medium) | 4,000 | Cheapest tier |
| `claude-sonnet:5.5-nothink` | `claude-sonnet-5-5` | off (`between_tools`, no tools given) | high | 3,000 | Thinking off (deferred) |
| `claude-sonnet:5.5-think` | `claude-sonnet-5-5` | adaptive | high | 8,000 | Thinking on: the C2/C3 pair (deferred) |
| `claude-opus:5.5-low` | `claude-opus-5-5` | always on | low | 4,000 | Largest tier, shortest thinking |

Thinking tokens are billed as output. `max_tokens` caps the cost of one request; a reply cut off by
it has `stop_reason: "max_tokens"` and is scored on whatever text came back (usually none).

## Steps / Logic

### 1. Jobs
For each position × variant × format, a job with the worker's fields (`job_id` from
`generate_job_id`, the position's FEN, tier, theme and engine values).

### 2. Round 1: first answers
- The prompt is exactly the worker's: `build_prompt(fen, prompt_format=...)`, no system prompt.
- Each request's custom ID is the job ID with characters outside `[A-Za-z0-9_-]` replaced, plus `-r1`.
- Only jobs with no round-1 result in `responses.jsonl` are sent.

### 3. Round 2: the move re-ask
The worker's rescue, unchanged:
1. Parse the reply (`parse_response`). If the move is legal, done.
2. Otherwise look for a legal move anywhere in the reply (`extract_move_from_text`).
3. Otherwise re-ask with `retry_move_prompt(fen, format)` (its system prompt included; plain formats
   never see a FEN) as a round-2 request, and take the first legal move in that reply.

**Difference from the local runs:** the local re-ask used temperature 0. Current Claude models
reject sampling settings, so every Claude request runs at the API default. Reported as a limitation.

### 4. Batches
- Submit, save the batch ID to `batches.json`, poll every 60 s until it ends, store every result in
  `responses.jsonl`, then remove the ID from `batches.json`.
- On start, any batch still in `batches.json` is collected first; nothing is resubmitted while one is open.
- Errored, expired or cancelled requests are logged and not stored, so the next run sends them again.

### 5. Budget guard
- **Spent** = Σ over stored results of (input tokens × input price + output tokens × output price) × 0.5.
- **Estimate** for a new batch = number of requests × mean tokens per request for that variant from
  stored results (once at least 10 exist), else the config's conservative defaults.
- If spent + estimate > `budget_usd`, nothing is submitted and the script exits with the numbers.
- `--dry-run` prints the requests' count and estimate and makes no API call.

### 6. Pilot, then the run
1. Pilot: `--positions-per-tier 2` (8 positions × 3 formats × 4 variants = 96 requests, cents).
2. Read the per-variant token use it reports; check the full-run estimate fits the budget.
3. Full run with the variants that fit, in this order: Haiku, Sonnet off, Sonnet on, Opus.
4. Then `scripts/enrich_cpl.py` for CPL, and the comparison against the criteria above.

### 7. Scoring
`score_all` and `build_result_record`, unchanged. Claude tags are listed in `API_MODEL_FAMILIES`
in `src/utils.py`: Anthropic doesn't publish parameter counts, so their size is `None`, never a guess.

## Edge cases

| Case | Behaviour |
|---|---|
| No API key anywhere | Exits before any request, saying where to put it |
| Key file readable by other users | Warning (it should be `chmod 600`) |
| Custom ID over 64 characters | Replaced by a stable hash of the job ID (the API limit is 64) |
| Request errored / expired | Not stored; resent on the next run |
| Script stopped while a batch runs | Next run collects that batch first |
| `stop_reason: "refusal"` or `"max_tokens"` | Stored and scored like any reply; counted in the summary |
| All three fields missing | Written to `failures.jsonl`, not to `evaluations.jsonl` (as the worker) |
| Budget would be exceeded | Nothing submitted; spent and estimate printed |
| Record already in `evaluations.jsonl` | Not written twice |

## Dependencies

- `anthropic` (official Python SDK, 2.x). Installing it adds packages and upgrades none of the
  existing ones (checked with `pip install --dry-run`).
- Internet access from the PC. No GPU needed.

## Reproducibility

- Seed 42 for the position sample and the bootstrap.
- Every record stores the API model ID returned by the API, effort and thinking setting.
- Outputs are not repeatable: hosted models change and sampling can't be fixed. `responses.jsonl`
  keeps every raw reply so the scoring can be re-run without new API calls.

## Tests (`tests/test_claude_batch.py`, no network, fake client)

- Sampling matches `populate_job_queue` run per tier.
- Request parameters for each variant (thinking, effort, max_tokens, no temperature, prompt identical to the worker's).
- Custom IDs: allowed characters only, ≤ 64, unique, round suffix.
- Rescue: legal first move kept; illegal but rescued from text; re-ask needed; re-ask reply used.
- Records: the usual schema plus the Claude fields; cost computed with the batch discount.
- Budget guard: refuses over budget; uses stored token means once there are enough; defaults before.
- Resume: open batch in `batches.json` is collected, not resubmitted; stored jobs aren't resent.
- Errored results aren't stored. Key loading: env first, then key file, error when neither.
