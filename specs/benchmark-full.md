# How Well Do Local LLMs Understand Chess? A Three-Task Benchmark Against Stockfish
### Technical Specification v4

---

## 1. Research Overview

### 1.1 Research Question

How accurately do locally run open-source LLMs evaluate chess positions, identify best moves, and explain positional advantages — compared to Stockfish 17 as ground truth — and does this vary systematically across model families and parameter scales?

### 1.2 Tasks

Each position in the benchmark is evaluated using three independent tasks. Every task has a fully pre-defined scoring method locked in before any results are collected.

| Task | What is measured | Scoring method |
|---|---|---|
| **T1 — Centipawn Evaluation** | Model estimates position evaluation numerically | Absolute error: `|model_eval − stockfish_eval|` in centipawns |
| **T2 — Best Move** | Model selects the best move for the side to play | Legality check (hallucination detection) + CPL (`|stockfish_eval_before − stockfish_eval_after|`) |
| **T3 — Positional Explanation** | Model explains why the position favours White, Black, or is equal | Two-point binary score (see Section 1.3) |

### 1.3 Explanation Scoring — Option A (Pre-defined)

T3 is scored on exactly two binary criteria, both defined before any data is collected:

**Point 1 — Side identification (0 or 1):**
Does the model correctly identify who stands better? Ground truth is Stockfish's evaluation:
- Stockfish > +50cp → White
- Stockfish < −50cp → Black
- −50cp to +50cp → Equal

**Point 2 — Theme identification (0 or 1):**
Does the model's explanation mention the correct key feature of the position? Ground truth is the `theme` tag in the dataset (e.g. `fork`, `pin`, `passed_pawn`). A match is scored if the theme word or a direct synonym appears in the explanation text.

Total explanation score per position: 0, 1, or 2.

### 1.4 Models Under Test

All models run locally via **Ollama**. Models are grouped by family to enable within-family parameter scaling as a secondary analysis.

| Family | Organisation | Sizes | Ollama tags |
|---|---|---|---|
| Qwen2.5 | Alibaba | 7B, 14B | `qwen2.5:7b`, `qwen2.5:14b` |
| Llama 3.2 | Meta | 3B, 8B | `llama3.2:3b`, `llama3.2:8b` |
| Mistral | Mistral AI | 7B | `mistral:7b` |
| Phi-4 | Microsoft | 14B | `phi4:14b` |
| Gemma 3 | Google | 4B, 12B | `gemma3:4b`, `gemma3:12b` |

Primary analysis compares models of comparable parameter counts across families. Within-family size comparisons are a secondary analysis.

### 1.5 Hypotheses

These are stated before data collection and will not be revised after results are seen.

| ID | Hypothesis |
|---|---|
| **H1** | T1 absolute error will increase with position difficulty tier across all models |
| **H2** | T2 CPL will increase with position difficulty tier across all models |
| **H3** | T3 explanation score will decrease with position difficulty tier across all models |
| **H4** | Within a model family, larger parameter counts will produce lower T1 error, lower T2 CPL, and higher T3 scores |
| **H5** | Models will perform relatively better on T3 (explanation) than T2 (move) on opening positions, and worse on T3 than T2 on endgame positions |

### 1.6 Secondary Experiment — Correction Loop

If T2 CPL exceeds the configured threshold, the model receives corrective feedback and is tested on a second position of the same theme. The learning delta (CPL attempt 1 − CPL attempt 2) measures whether feedback improves subsequent performance. A matched control condition (second position presented without feedback) isolates the effect of the correction itself. This is a secondary experiment and does not affect the primary three-task results.

---

## 2. Dataset

### 2.1 Three Position Sources

Positions are drawn from three sources to cover a range of familiarity and structure:

| Source | Description | Approximate count |
|---|---|---|
| **Lichess puzzles** | Tactical positions with known themes and difficulty ratings | 500 per tier |
| **Real game positions** | Positions sampled from Lichess game database across opening, middlegame, endgame phases | 500 per tier |
| **Generated positions** | Positions created programmatically using `python-chess` | 500 per tier |

### 2.2 Generated Positions

`python-chess` is used to generate positions that provably do not appear in any published game database, providing a strong test of reasoning without memorisation. Generation methods:

- **Random legal play** — play random legal moves from the starting position until a target move count is reached
- **Theme-seeded generation** — start from a known theme position and apply random perturbations to pieces
- **Endgame tablebases** — generate positions from Syzygy endgame tablebases for clean, engine-verifiable endgame positions

All generated positions are validated: legal FEN, not in check unless intentional, not an immediate game-over state.

### 2.3 Position Schema

Each dataset file is a **JSON array**:

```json
[
  {
    "id": 421,
    "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 2 3",
    "pgn_moves": "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6",
    "theme": "pin",
    "difficulty": "medium",
    "phase": "opening",
    "source": "lichess_puzzles",
    "stockfish_eval": 85,
    "stockfish_best_move": "Nxe5"
  }
]
```

| Field | Type | Description |
|---|---|---|
| `id` | integer | Unique position identifier |
| `fen` | string | FEN string |
| `pgn_moves` | string | Move history in PGN notation |
| `theme` | string | Key tactical/strategic feature |
| `difficulty` | string | `easy`, `medium`, `hard`, `extreme` |
| `phase` | string | `opening`, `middlegame`, `endgame` |
| `source` | string | `lichess_puzzles`, `real_game`, `generated` |
| `stockfish_eval` | integer | Stockfish centipawn evaluation (pre-computed) |
| `stockfish_best_move` | string | Stockfish best move in SAN (pre-computed) |

> **Note:** `stockfish_eval` and `stockfish_best_move` are pre-computed at dataset creation time at depth 22. This avoids repeated Stockfish calls during benchmarking and ensures consistent ground truth across all runs.

### 2.4 Difficulty Tiers

| Tier | Lichess puzzle rating equivalent | Description |
|---|---|---|
| `easy` | < 1200 | Simple one-move tactics, clear material advantage |
| `medium` | 1200–1800 | Two-move combinations, mild positional features |
| `hard` | 1800–2400 | Multi-step tactics, non-obvious positional play |
| `extreme` | > 2400 | Complex combinations, deep strategic concepts |

---

## 3. Directory Structure

```
chess-llm-bench/
│
├── data/
│   ├── easy.json
│   ├── medium.json
│   ├── hard.json
│   └── extreme.json
│
├── jobs/
│   └── jobs.db                    # SQLite job queue (auto-created)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Load, filter, sample dataset
│   ├── dataset_builder.py         # Build datasets from Lichess, PGN, and generated positions
│   ├── position_generator.py      # python-chess position generation
│   ├── job_generator.py           # Create standard + correction + control jobs
│   ├── worker.py                  # Worker process loop
│   ├── job_queue.py               # SQLite queue interface
│   ├── engine_wrapper.py          # Stockfish subprocess management
│   ├── llm_client.py              # Ollama HTTP client
│   ├── evaluator.py               # T1/T2/T3 scoring logic
│   ├── feedback_loop.py           # Correction and control condition
│   ├── result_writer.py           # File-locked JSONL writer
│   ├── metrics.py                 # Aggregate stats and analysis
│   └── utils.py                   # Config, hashing, logging, directory setup
│
├── scripts/
│   ├── build_dataset.py           # CLI: build all dataset files
│   ├── precompute_stockfish.py    # CLI: pre-compute Stockfish evals for all positions
│   ├── generate_jobs.py           # CLI: populate job queue
│   ├── pull_models.py             # CLI: pull Ollama models
│   ├── run_workers.py             # CLI: launch worker pool
│   └── generate_plots.py          # CLI: produce visualisations
│
├── results/
│   ├── evaluations.jsonl          # Append-only result records
│   ├── logs/                      # Per-run metadata snapshots
│   ├── plots/                     # Generated visualisations
│   └── metrics/                   # Aggregated summaries
│
├── config/
│   └── config.yaml
│
├── requirements.txt
└── README.md
```

---

## 4. Configuration

```yaml
stockfish:
  path: /usr/games/stockfish
  depth: 22
  threads: 1

ollama:
  base_url: http://localhost:11434   # change to ngrok URL for Colab
  timeout: 180
  max_retries: 3

models:
  - qwen2.5:7b
  - qwen2.5:14b
  - llama3.2:3b
  - llama3.2:8b
  - mistral:7b
  - phi4:14b
  - gemma3:4b
  - gemma3:12b

benchmark:
  max_positions_per_tier: 500
  random_seed: 42

evaluation:
  cpl_threshold: 50
  centipawn_eval_range: [-2000, 2000]   # clamp model responses to this range

workers:
  count: 4

correction_loop:
  enabled: true
  control_group: true
  max_attempts: 2
```

Environment variables override config values using `CHESS_<SECTION>_<KEY>`.

---

## 5. Prompts

All three tasks are sent in a **single prompt per position** to minimise API calls and keep context consistent.

### Standard prompt (`pgn+fen` format)

```
You are analysing a chess position.

Moves played so far:
{pgn_moves}

Current position (FEN):
{fen}

Answer all three questions below.

1. What is the centipawn evaluation of this position from White's perspective?
   A positive number means White is better. A negative number means Black is better.
   0 means equal. Give only a number, no explanation.

2. What is the best move for the side to play? Give only the move in SAN notation.

3. Who stands better in this position — White, Black, or is it equal?
   Give a one-sentence explanation of the key reason.

Respond using this exact format:
Eval: <integer>
Move: <SAN move>
Explanation: <White / Black / Equal> — <one sentence reason>
```

### `fen_only` format

Same as above but without the `Moves played so far` section.

### `cot` format

Same as above but with this added before the output format instruction:

```
Think step by step before answering:
- What are the key features of this position?
- Who controls more space? Who has more active pieces?
- What is the most forcing move available?
```

---

## 6. Job Definition

```json
{
  "job_id": "job_00421",
  "job_type": "standard",
  "position_id": 421,
  "fen": "...",
  "pgn_moves": "...",
  "model": "qwen2.5:7b",
  "prompt_format": "pgn+fen",
  "difficulty": "medium",
  "phase": "opening",
  "source": "lichess_puzzles",
  "theme": "pin",
  "trial": 1,
  "status": "pending",
  "paired_control_job_id": null,
  "hash": "<sha256>"
}
```

**Job types:** `standard`, `correction`, `control`

**Status values:** `pending` → `in_progress` → `done` / `failed`

---

## 7. Worker Execution

```bash
python scripts/run_workers.py --workers 4
```

Each worker loop:

1. Claims next `pending` job atomically from SQLite
2. Builds prompt from position + prompt format
3. Sends to Ollama via HTTP POST `/api/chat`
4. Parses `Eval`, `Move`, and `Explanation` fields
5. Scores T1 (absolute centipawn error)
6. Scores T2 (legality + CPL against pre-computed Stockfish best move)
7. Scores T3 (Option A: two binary points)
8. If correction loop enabled and T2 CPL > threshold → triggers correction job
9. Appends result to `evaluations.jsonl`
10. Marks job `done`

---

## 8. Scoring Implementation

### T1 — Centipawn Evaluation

```python
# Clamp model response to configured range before scoring
model_eval = max(-2000, min(2000, int(parsed_eval)))
t1_absolute_error = abs(model_eval - position["stockfish_eval"])

# Directional accuracy (secondary)
def direction(val):
    if val > 50: return "White"
    if val < -50: return "Black"
    return "Equal"

t1_direction_correct = direction(model_eval) == direction(position["stockfish_eval"])
```

### T2 — Best Move

```python
import chess

board = chess.Board(position["fen"])
try:
    move = board.parse_san(parsed_move)
    if move in board.legal_moves:
        t2_legal = True
        board_copy = board.copy()
        board_copy.push(move)
        # CPL uses pre-computed stockfish eval
        t2_cpl = abs(position["stockfish_eval"] - get_eval_after_move(board_copy))
    else:
        t2_legal = False
        t2_cpl = None
except Exception:
    t2_legal = False
    t2_cpl = None
```

### T3 — Positional Explanation (Option A)

```python
THEME_SYNONYMS = {
    "fork": ["fork", "double attack"],
    "pin": ["pin", "pinned", "pinning"],
    "skewer": ["skewer"],
    "passed_pawn": ["passed pawn", "passer"],
    "discovery": ["discovered", "discovery"],
    # extend as needed
}

def score_explanation(explanation_text, stockfish_eval, theme, model_side_claim):
    # Point 1: side identification
    ground_truth_side = direction(stockfish_eval)
    p1 = 1 if model_side_claim == ground_truth_side else 0

    # Point 2: theme identification
    keywords = THEME_SYNONYMS.get(theme, [theme])
    p2 = 1 if any(kw in explanation_text.lower() for kw in keywords) else 0

    return p1, p2, p1 + p2
```

---

## 9. Result Record

```json
{
  "job_id": "job_00421",
  "job_type": "standard",
  "attempt": 1,
  "parent_job_id": null,
  "position_id": 421,
  "model": "qwen2.5:7b",
  "model_family": "qwen",
  "model_size_b": 7,
  "prompt_format": "pgn+fen",
  "difficulty": "medium",
  "phase": "opening",
  "source": "lichess_puzzles",
  "theme": "pin",

  "t1_model_eval": 120,
  "t1_stockfish_eval": 85,
  "t1_absolute_error": 35,
  "t1_direction_correct": true,

  "t2_move": "Nxe5",
  "t2_best_move": "Nxe5",
  "t2_legal": true,
  "t2_cpl": 0,

  "t3_explanation": "White is better due to the pinned knight on c6.",
  "t3_side_claimed": "White",
  "t3_p1_side_correct": 1,
  "t3_p2_theme_correct": 1,
  "t3_score": 2,

  "inference_ms": 3840,
  "timestamp": "2026-03-11T10:00:00Z"
}
```

---

## 10. Correction Loop (Secondary Experiment)

If `t2_cpl` exceeds `evaluation.cpl_threshold`:

**Feedback condition — correction prompt:**

```
Your move was not the best.

Position: {fen}
Your move: {move}
Best move: {best_move}

Explain why {best_move} is stronger. Then answer the same three questions
for the following new position of the same type.
```

**Control condition:** the follow-up position is presented with no correction — the standard three-task prompt only.

**Follow-up position selection:**
1. Filter dataset for same `theme` and `difficulty`
2. Exclude original and any previously used position in this trial
3. Shuffle using `random.Random(random_seed + job_id_int)`
4. Take first result; fall back to same difficulty tier if no same-theme position available

**Learning metric:**

```
Learning Delta = t2_cpl_attempt_1 − t2_cpl_attempt_2
Net feedback effect = Learning Delta (correction) − Learning Delta (control)
```

---

## 11. Visualisations

All saved to `results/plots/`.

| Plot | Filename | Description |
|---|---|---|
| **T1 Error by Difficulty** | `t1_error_by_difficulty.png` | Mean absolute centipawn error per difficulty tier per model |
| **T1 Error by Phase** | `t1_error_by_phase.png` | Mean absolute centipawn error across opening / middlegame / endgame |
| **T2 CPL by Difficulty** | `t2_cpl_by_difficulty.png` | Mean CPL per difficulty tier per model |
| **T2 Hallucination Rate** | `t2_hallucination.png` | % illegal moves per difficulty tier per model |
| **T3 Score by Difficulty** | `t3_score_by_difficulty.png` | Mean explanation score per difficulty tier per model |
| **T3 Score by Phase** | `t3_score_by_phase.png` | Explanation score across game phases |
| **Task Profile Radar** | `task_profile_radar.png` | Radar/spider chart showing T1/T2/T3 relative strengths per model |
| **Parameter Scaling** | `parameter_scaling.png` | T1/T2/T3 vs parameter count within each model family |
| **Source Comparison** | `source_comparison.png` | Performance on Lichess vs real game vs generated positions |
| **Correction Delta** | `correction_delta.png` | Net feedback effect per model (secondary experiment) |

---

## 12. Duplicate Protection

```
hash = SHA256(fen + model + prompt_format + job_type + str(trial))
```

Two-layer protection:
1. At job generation: duplicate hashes skipped on SQLite insert
2. At worker execution: hash checked against existing results before querying Ollama

---

## 13. Error Handling

| Error | Behaviour |
|---|---|
| Ollama not running | Abort immediately with clear message |
| Ollama timeout | Retry once; mark `failed` if second attempt times out |
| Missing/unparseable `Eval` field | `t1_absolute_error = null`; continue scoring T2 and T3 |
| Missing/unparseable `Move` field | `t2_legal = false`; continue scoring T1 and T3 |
| Missing `Explanation` field | `t3_score = null`; continue scoring T1 and T2 |
| All three fields missing | Mark job `failed`, log full raw response |
| Invalid FEN | Mark job `failed` immediately |

Partial results are always written — a job that produces T1 and T2 but fails T3 parsing is still recorded and counted.

---

## 14. Module Responsibilities

| Module | Responsibility |
|---|---|
| `dataset_builder.py` | Fetch Lichess puzzles, sample real game PGNs, call position generator |
| `position_generator.py` | Generate legal positions via python-chess; validate and export |
| `data_loader.py` | Load JSON datasets; filter/sample by difficulty, phase, source, theme |
| `job_generator.py` | Create standard + correction + control jobs; compute hashes; insert into SQLite |
| `job_queue.py` | Atomic claim, complete, fail, count jobs in SQLite |
| `worker.py` | Main loop: claim → prompt → parse → score T1/T2/T3 → write |
| `engine_wrapper.py` | Stockfish: start, evaluate, restart on crash |
| `llm_client.py` | Ollama HTTP requests with retry and timeout |
| `evaluator.py` | T1 scoring, T2 legality + CPL, T3 Option A scoring |
| `feedback_loop.py` | Correction prompt, follow-up selection, control condition |
| `result_writer.py` | File-locked append to `evaluations.jsonl` |
| `metrics.py` | Aggregate T1/T2/T3 stats, parameter scaling analysis, learning delta |
| `utils.py` | Config loading, hashing, logging, directory creation |

---

## 15. Dependencies

**Python 3.10+**

| Library | Purpose |
|---|---|
| `python-chess` | Move validation, board logic, position generation |
| `requests` | Ollama HTTP client |
| `pandas` | Result analysis |
| `numpy` | Numerical computation |
| `pyyaml` | Config parsing |
| `matplotlib` | Plot generation |
| `seaborn` | Plot styling |
| `tqdm` | Progress bars |
| `filelock` | Concurrent-safe JSONL writes |

---

## 16. Performance

| Requirement | Detail |
|---|---|
| **Stockfish evals** | Pre-computed at dataset build time — no Stockfish calls during benchmarking |
| **Worker parallelism** | `multiprocessing.Pool`, default 4 workers |
| **Ollama parallelism** | Set `OLLAMA_NUM_PARALLEL=4` in environment |
| **Estimated throughput** | 3–8s per job at 7B; 8–25s at 14B on RTX 5080 |
| **Colab fallback** | Change `ollama.base_url` to ngrok tunnel URL; no code changes needed |

---

## 17. Reproducibility

Every run writes `results/logs/run_<timestamp>.json`:

```json
{
  "timestamp": "2026-03-11T10:00:00Z",
  "models": ["qwen2.5:7b", "llama3.2:8b", "mistral:7b"],
  "prompt_formats": ["fen_only", "pgn+fen", "cot"],
  "hypotheses": ["H1", "H2", "H3", "H4", "H5"],
  "scoring_method": "Option A",
  "config_snapshot": {},
  "dataset_files": ["easy.json", "medium.json", "hard.json", "extreme.json"],
  "random_seed": 42,
  "worker_count": 4,
  "total_jobs": 0,
  "ollama_version": "",
  "stockfish_version": "17"
}
```

---

## 18. Implementation Notes

### SQLite atomic job claiming

```sql
UPDATE jobs
SET status = 'in_progress', worker_id = ?, claimed_at = ?
WHERE job_id = (
  SELECT job_id FROM jobs
  WHERE status = 'pending'
  LIMIT 1
)
RETURNING *;
```

Enable WAL mode at startup: `conn.execute("PRAGMA journal_mode=WAL;")`

### Multiprocessing, not threading

Use `multiprocessing.Pool`. Each worker owns its Stockfish instance. Stockfish evals are only needed for T2 CPL during benchmarking — ground truth evals are pre-computed in the dataset.

### File-locked writes

```python
from filelock import FileLock
lock = FileLock("results/evaluations.jsonl.lock")
with lock:
    with open("results/evaluations.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
```

### Dry-run mode

`run_workers.py --dry-run` processes the first 10 jobs and prints to stdout without writing results. Always run this before a full benchmark run.

### Pre-computing Stockfish evaluations

Run `scripts/precompute_stockfish.py` after building the dataset and before generating jobs. This populates `stockfish_eval` and `stockfish_best_move` in every position record. During benchmarking, T2 CPL is computed by comparing the model's move against the pre-computed best move — no live Stockfish calls are needed unless T2 CPL requires re-evaluation after the model's move.

---

## 19. Acceptance Criteria

- [ ] Dataset contains positions from all three sources: Lichess puzzles, real games, generated
- [ ] All positions have pre-computed `stockfish_eval` and `stockfish_best_move`
- [ ] All configured Ollama models pull successfully via `pull_models.py`
- [ ] Single prompt correctly elicits all three task responses
- [ ] T1 absolute error computed correctly against pre-computed ground truth
- [ ] T2 legality check and CPL computed correctly
- [ ] T3 scored using Option A as defined in Section 1.3 — no changes after data collection begins
- [ ] Partial results recorded when only some tasks parse successfully
- [ ] Correction and control jobs correctly linked via `paired_control_job_id`
- [ ] All ten plots generated from `evaluations.jsonl`
- [ ] Parameter scaling plot correctly groups results by model family and size
- [ ] Source comparison plot correctly separates Lichess / real game / generated results
- [ ] Dry-run mode works end-to-end
- [ ] Experiments resume correctly after interruption
- [ ] Reproducibility snapshot written at start of every run
