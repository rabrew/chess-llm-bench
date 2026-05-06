# Spec: Commercial Model Integration — Anthropic

**Created:** 2026-05-06
**Author:** Ryan Brew

---

## Goal

Extend the chess LLM benchmark to evaluate commercial API models alongside the existing 23 local Ollama models. The first provider is Anthropic (Claude). The benchmark infrastructure — job queue, scoring, result writer, prompt formats — is reused unchanged. Only the LLM client layer changes.

This lets us answer: **how do Claude models compare to local open-source models of similar parameter scale on chess spatial reasoning?**

---

## Inputs / Outputs

**Inputs:**
- `data/{easy,medium,hard,extreme}.json` — same 4,000 positions used for all models
- `config/config.yaml` — extended with `anthropic:` section and Claude model list
- `ANTHROPIC_API_KEY` environment variable — never hardcoded, never committed

**Outputs:**
- `results/evaluations.jsonl` — same append-only format, same record schema
- Claude model entries appear in all existing dashboard charts and metrics automatically

---

## Models

| Model | Ollama-style tag | API model ID | Size tier |
|---|---|---|---|
| Claude Haiku 4.5 | `claude-haiku-4-5` | `claude-haiku-4-5-20251001` | Small (fastest, cheapest) |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | `claude-sonnet-4-6` | Medium |
| Claude Opus 4.6 | `claude-opus-4-6` | `claude-opus-4-6` | Large (most capable) |

Start with Haiku only to validate the pipeline end-to-end before spending budget on Sonnet and Opus.

---

## Cost Estimate

**Per model: 4,000 positions × 6 prompt formats = 24,000 jobs**

Approximate token counts per job:
- Input: ~250 tokens (FEN + PGN history + 3-task question + format instruction)
- Output: ~80 tokens (Eval: N, Move: X, Explanation: ...)

| Model | Input $/1M | Output $/1M | Est. cost per model |
|---|---|---|---|
| Haiku 4.5 | $0.80 | $4.00 | ~$13 |
| Sonnet 4.6 | $3.00 | $15.00 | ~$46 |
| Opus 4.6 | $15.00 | $75.00 | ~$228 |

Run Haiku first (~$13). Confirm results look sensible. Proceed to Sonnet. Hold off on Opus unless the research narrative specifically benefits from it.

---

## Steps / Logic

### 1. New client: `src/anthropic_client.py`

Add `AnthropicClient` class with the same interface as `OllamaClient`:

```python
class AnthropicClient:
    def __init__(self, api_key: str, timeout: int = 60, max_retries: int = 3): ...
    def is_available(self) -> bool: ...          # validate API key with a cheap probe
    def chat(self, model: str, prompt: str, system_prompt: str | None = None, temperature: float | None = None) -> dict: ...
```

`chat()` returns the same dict shape as `OllamaClient.chat()`:
```python
{"response": str, "inference_ms": int, "success": bool, "model": str}
```

Use the `anthropic` Python SDK (`pip install anthropic`). The SDK handles retries and rate-limit backoff internally — configure it with `max_retries=3`.

Rate limiting: Anthropic returns HTTP 429 on quota exhaustion. The SDK raises `anthropic.RateLimitError`. Catch it, wait `retry_after` seconds (from response headers if present, else exponential backoff), and retry up to `max_retries` times.

### 2. Client routing in `src/worker.py`

Workers currently create one `OllamaClient` at init time. Extend `Worker.__init__` to detect the model name and instantiate the correct client:

```python
if job["model"].startswith("claude-"):
    self.llm_client = AnthropicClient(api_key=os.environ["ANTHROPIC_API_KEY"])
else:
    self.llm_client = OllamaClient(...)
```

No other changes to `worker.py` — prompt building, parsing, scoring, result writing are all unchanged.

### 3. Config extension (`config/config.yaml`)

Add an `anthropic:` section and Claude model entries in the `models:` list:

```yaml
anthropic:
  timeout: 60
  max_retries: 3

models:
  # ... existing Ollama models ...

  # Commercial — Anthropic
  - claude-haiku-4-5
  - claude-sonnet-4-6
  # - claude-opus-4-6  # expensive — enable explicitly
```

The `ollama:` section is unchanged. Workers detect the provider from the model name prefix.

### 4. `parse_model_info()` in `src/utils.py`

Extend the family/size map to recognise Claude models:

```python
family_map = {
    ...existing entries...,
    "claude": "claude",
}

# Size overrides for Claude (can't parse from tag alone)
SIZE_OVERRIDES = {
    "claude-haiku-4-5": 20,    # approximate
    "claude-sonnet-4-6": 70,   # approximate
    "claude-opus-4-6": 200,    # approximate
}
```

Size values are approximate — used only for the parameter-scaling chart.

### 5. `scripts/generate_jobs.py`

No changes required. Jobs are generated from `config.yaml`'s `models` list. Claude model entries produce jobs just like any other model.

### 6. `run_all.sh` — skip Ollama restart for Claude

The `restart_ollama_for_model` function should be a no-op for `claude-*` models:

```bash
restart_ollama_for_model() {
    local model="$1"
    if [[ "$model" == claude-* ]]; then
        echo "1" > /tmp/bench_workers   # single worker for API calls
        return  # no Ollama restart needed
    fi
    # ... existing logic ...
}
```

Use 1 worker for API models initially. Anthropic's default rate limit is 50 RPM — a single worker will stay well under that.

---

## Edge Cases

| Scenario | Behaviour |
|---|---|
| `ANTHROPIC_API_KEY` not set | `AnthropicClient.__init__` raises `ValueError` with clear message before any jobs run |
| API key invalid / auth error | `is_available()` returns False; pipeline aborts at Ollama-availability check (reuse same check) |
| HTTP 429 rate limit | Catch `anthropic.RateLimitError`, exponential backoff (2s, 4s, 8s), up to `max_retries` |
| HTTP 529 overloaded | Treat same as 429 |
| Response truncated (finish_reason=max_tokens) | Parse whatever was returned; missing fields recorded as None; job still completes |
| Prompt format `move_only` with legal moves list | Prompt includes legal move list as usual — no change needed |
| `is_available()` check | Make a minimal messages call with `max_tokens=1`; catches auth errors and network issues before jobs run |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `anthropic` | `>=0.50.0` | Official Anthropic Python SDK |

Add to `requirements.txt`. No other new dependencies.

---

## Security

- `ANTHROPIC_API_KEY` is **never** written to config files, logs, or result records
- `.env` file (if used locally) must be in `.gitignore`
- `AnthropicClient` reads the key from `os.environ` only — never accepts it as a positional argument to avoid accidental logging

---

## Project Structure Changes

```
src/
  anthropic_client.py   ← NEW
  llm_client.py         ← unchanged
  worker.py             ← route to correct client based on model prefix
  utils.py              ← extend parse_model_info for claude family/size

config/
  config.yaml           ← add anthropic: section, add claude models to list

requirements.txt        ← add anthropic>=0.50.0

tests/
  test_anthropic_client.py  ← NEW (mock SDK, no real API calls)
```

---

## Tests (`tests/test_anthropic_client.py`)

All tests use `unittest.mock` to patch the `anthropic` SDK — no real API calls, no API key required in CI.

| Test | What it checks |
|---|---|
| `test_chat_success` | Normal response → correct dict shape returned |
| `test_chat_retries_on_rate_limit` | `RateLimitError` → retries up to max_retries, then returns success=False |
| `test_chat_timeout` | `APITimeoutError` → returns `{"success": False}` |
| `test_is_available_returns_true` | Valid key → True |
| `test_is_available_returns_false_on_auth_error` | `AuthenticationError` → False |
| `test_api_key_not_set_raises` | Missing env var → ValueError at init |
| `test_response_shape_matches_ollama` | Return dict has same keys as OllamaClient.chat() |

---

## Acceptance Criteria

- [ ] `AnthropicClient` passes all unit tests with mocked SDK
- [ ] `ANTHROPIC_API_KEY` not present → clear error before any jobs run
- [ ] Dry run with `--dry-run --max-jobs 3 --model claude-haiku-4-5` completes without errors
- [ ] 24,000 jobs for `claude-haiku-4-5` complete and appear in `evaluations.jsonl`
- [ ] Dashboard shows Claude models alongside Ollama models
- [ ] No API key appears in any log file or result record
- [ ] All existing tests still pass
