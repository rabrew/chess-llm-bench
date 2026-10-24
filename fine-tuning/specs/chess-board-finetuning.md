# Spec: Chess Board Fine-tuning Curriculum Experiment

## Goal

Benchmark results show a bimodal failure pattern: models flip between perfect moves and catastrophic blunders. The hypothesis is that models have fragile FEN-to-board-state representations — they sometimes misread the board entirely.

This experiment fine-tunes each of the 22 benchmarked models on a Stage 1 curriculum task: given a FEN string, list every piece on the board and its exact square. The expectation is that reliable board-state grounding reduces downstream blunder rate in the existing T1/T2/T3 benchmark tasks.

---

## Training Task

**Stage 1 (this experiment):** FEN → explicit piece location list

Example output:
```
White: King e1, Queen d1, Rooks a1 h1, Bishops c4 f1, Knights f3 b1, Pawns a2 b2 c2 d2 e4 f2 g2 h2
Black: King e8, Queen d8, Rooks a8 h8, Bishops c8 f8, Knights c6 g8, Pawns a7 b7 c7 d7 e5 f7 g7 h7
Side to move: White
```

**Stage 2 (existing benchmark):** FEN → move / eval / explanation (unchanged)

---

## Inputs / Outputs

**Training data generation (`generate_board_dataset.py`)**
- Input: `data/{easy,medium,hard,extreme}.json` (5,000 records per tier = 20,000 total)
- Output: `data/board_dataset.jsonl` — Alpaca-format JSONL with keys `instruction`, `input`, `output`
- Deduplication: by FEN before writing

**Fine-tuning (`finetune_model.py`)**
- Input: `data/board_dataset.jsonl`, HuggingFace model weights, `fine-tuning/config/finetune_config.yaml`
- Output: GGUF file at `~/finetuned_gguf/<ollama_ft_tag>/model-q4_k_m.gguf`

**Ollama import (`import_finetuned_to_ollama.py`)**
- Input: GGUF file
- Output: new Ollama model tag (e.g. `llama3.2:3b-board-ft`)

**Benchmark re-run (`ft_bench_config.yaml` + existing pipeline)**
- Input: fine-tuned Ollama models
- Output: `results/finetuning_experiment/evaluations.jsonl`

**Comparison (`compare_finetuning_results.py`)**
- Input: `results/evaluations.jsonl` (base) + `results/finetuning_experiment/evaluations.jsonl` (ft)
- Output: `results/finetuning_experiment/comparison.csv`

---

## Curriculum Stages

### Stage 1 — Board Comprehension (this spec)
Train model to explicitly enumerate every piece and square from a FEN string.
Dataset: 20,000 FEN positions drawn from existing benchmark data.

### Stage 2 — Downstream Benchmark (existing)
Run the existing 6-prompt-format benchmark against fine-tuned models.
Compare T1/T2/T3 metrics against base model results.

---

## Board State Utility (`fine-tuning/src/board_state.py`)

Central module used by training data generation, fine-tuning evaluation, and comparison.

### `fen_to_board_description(fen) -> str`
Uses python-chess to extract all pieces and return a canonical description string.
Format: `White: King e1, Queen d1, Rooks a1 h1, ...\nBlack: ...\nSide to move: White`
Pieces within each type are sorted by file then rank.

### `parse_board_description(text) -> dict`
Best-effort parser to convert a model's text output back into `{piece_label: [squares]}`.
Handles partial/slightly malformed output.

### `verify_board_description(fen, model_text) -> dict`
Compare model output against python-chess ground truth. Returns:
- `correct`: bool
- `errors`: list of discrepancy strings
- `missing_pieces`: pieces in FEN but absent from model output
- `extra_pieces`: pieces in model output not in FEN
- `wrong_squares`: pieces mentioned on wrong square
- `accuracy`: float 0.0–1.0

### `build_correction(fen, model_text) -> str`
Produce a correction string for wrong model output.
Format: `Correction — <list of fixes>. Full board: <ground truth>`

---

## Fine-tuning Pipeline

1. `python fine-tuning/scripts/generate_board_dataset.py` — create `data/board_dataset.jsonl`
2. `python fine-tuning/scripts/finetune_model.py <ollama_base>` — one model per invocation
3. `python fine-tuning/scripts/import_finetuned_to_ollama.py <ollama_base>` — register in Ollama
4. `python scripts/generate_jobs.py --config fine-tuning/config/ft_bench_config.yaml` — generate jobs
5. `python scripts/run_workers.py --config fine-tuning/config/ft_bench_config.yaml` — run benchmark
6. `python fine-tuning/scripts/compare_finetuning_results.py` — compare results

---

## Held-Out Split (F-2, recorded 2026-10-24)

`split_held_out(records, seed=42, n=200)` shuffles with `random.Random(seed)` and takes the first 200
records as the held-out set. The held-out positions are used for per-epoch board accuracy, by
`verify_export.py` and by the F-9 board-reading test. They must never be trained on.

- **Rule:** the held-out and training sets share no FEN. The split is positional, so this holds only
  if no FEN appears twice in the dataset. `split_held_out` therefore **raises `ValueError` if any
  `input` (FEN) is duplicated**, rather than letting a duplicate leak into both sides.
- **Checked 2026-10-24:** `data/board_dataset.jsonl` has 20,000 records and 20,000 unique FENs. None
  are benchmark FENs (F-1).
- **Tests** (`fine-tuning/tests/test_held_out_split.py`):
  - the sizes, determinism and FEN-disjointness of the split
  - the duplicate guard
  - the same checks on the real dataset (skipped when the file is absent), including that no held-out
    FEN is a benchmark FEN

---

## Experiment Tracking (F-7, recorded 2026-10-24)

Each training run writes one JSON file to `runs/` at the repo root (tracked in git; small). Model
weights, adapters and GGUFs stay out of git.

**Record fields:**

| Field | Meaning |
|---|---|
| `run_id` | `<UTC stamp>_<ft tag with ':' → '_'>` |
| `created_at` | UTC ISO time |
| `model`, `ft_tag`, `hf_id` | base Ollama tag, fine-tuned tag, HuggingFace id |
| `git_sha`, `git_dirty` | from `src.provenance.git_state`; `None` when unknown |
| `seed`, `config_path`, `config` | the training seed, plus a snapshot of the whole fine-tuning config |
| `dataset_path`, `dataset_sha256`, `dataset_records` | |
| `metrics` | `global_step`, `epoch`, `final_train_loss`, `loss_history` (step, loss), and `held_out_accuracy` per epoch where recorded |
| `outputs` | checkpoint dir and GGUF path |
| `notes` | free text, e.g. why a record was backfilled |

**Writing:**
- `finetune_model.py` writes the record after a successful GGUF export.
- `write_run_record` refuses to overwrite an existing file.

**Backfill:**
- `python fine-tuning/scripts/run_record.py backfill` creates records for batch 1 from each checkpoint
  dir's latest `trainer_state.json`.
- The git SHA at training time is unknown, so it is left `None` and stated in `notes`.
- The dataset hash is of the current `data/board_dataset.jsonl`. It was not changed after batch 1 was
  generated; this is stated as an assumption in `notes`.

**Tests:** `fine-tuning/tests/test_run_record.py`.

**Training script structure (2026-10-24):**
- `run_finetuning` is split into small steps: `plan_run`, then loading and splitting the data, loading
  the model, training, exporting the GGUF, and writing the run record.
- The model size used for the gradient-checkpointing and CPU-offload thresholds comes from
  `MODEL_REGISTRY` (`parse_model_info`), not a parse of the tag name. That parse duplicated the
  registry and read `mixtral:8x7b` as 56B.
- `EpochAccuracyCallback` keeps `history` (`epoch`, `held_out_accuracy`) for the run record. A failed
  check adds no entry.
- **Tests:** `fine-tuning/tests/test_finetune_run_steps.py`.

---

## Model Selection (F-5, recorded 2026-10-23)

The fine-tuned models are the benchmarked models that meet three conditions:

- **They can be run.** They fit the available hardware for both inference and training.
- **They are open source and trainable.** The weights are downloadable, and a LoRA adapter can be
  trained and merged.
- **They span many model families.** Llama, Gemma, Qwen, Mistral, DeepSeek-R1 and others, so a result
  is not one family's quirk.

Every fine-tuned model keeps its base model in the same comparison table.

**Batch 1 (done):**
- llama3.2:3b, gemma3:4b, qwen2.5:7b, mistral:7b and deepseek-r1:7b
- gemma4:e2b, which is excluded from the benchmark: see `specs/finetuned-benchmark-run.md`, "Gemma 4
  exclusion"

**Batch 2 (decided 2026-10-23):** the remaining models are trained on a **cloud GPU**, not the local PC.
- The local PC (RTX 5080, 16 GB VRAM, 32 GB RAM) cannot QLoRA-train the 26–35B models, mixtral:8x7b or
  llama3.3:70b at the current settings.
- The 70B merge-and-export would not fit on its disk either: a 16-bit merge needs about 140 GB, before
  the BF16 GGUF.
- Training, merging and GGUF export run in the cloud. The Q4_K_M GGUF and the LoRA adapter come back to
  the PC, where they are imported into Ollama, checked with `verify_export.py`, and benchmarked
  locally. This keeps inference settings identical to the base runs.

---

## Benchmark Runs

- The fine-tuned models run with the same `max_positions_per_tier: 1000` and `random_seed: 42` as the
  base models.
- The results go to `results/finetuning_experiment/evaluations_ft.jsonl` (`config/config_finetuned.yaml`).
- **The base models are re-run on the current code** (`config/config_base_rerun.yaml`, writing
  `results/finetuning_experiment/evaluations_base_rerun.jsonl`). The original base records predate
  the move rescue, so they are not comparable (issues-log Issue 18).

---

## Comparison Methodology

`fine-tuning/scripts/compare_finetuning_results.py` (spec: `specs/t1-zero-baseline-and-ft-comparison.md`).
It pairs first-attempt records on `(base model, position_id, prompt_format)` and reports base,
fine-tuned and delta values for:

- **T1:** Spearman ρ, Cohen's κ at ±50 cp, and MAE relative to always answering 0. Raw direction
  accuracy is not used, because it rewards guessing near 0 (Issue 14).
- **T2:** legal-move rate over move-asking prompts, and clamped CPL.
- **T3:** mean whole-word T3 score.

T0 (board-description accuracy) is measured separately by `board_reading_test.py`.

---

## Hardware

**Local PC:** RTX 5080 (16 GB VRAM), 32 GB RAM. This is the hardware batch 1 was trained on; the
earlier "RTX 4090, 24 GB" table here was wrong.

| Size | Where | Strategy |
|---|---|---|
| 3–8B | Local | 4-bit QLoRA (batch 1 took about 1.5–3.5 h per model) |
| 10–14B | Local, if VRAM allows; otherwise cloud | 4-bit QLoRA + gradient checkpointing |
| 26–35B, mixtral:8x7b, llama3.3:70b | Cloud GPU (batch 2) | 4-bit QLoRA |

---

## Dependencies

- `python-chess>=1.10.0` (already in requirements.txt)
- `unsloth[colab-new]` — 4–8× faster QLoRA fine-tuning
- `trl>=0.8.0` — SFTTrainer
- `datasets>=2.18.0` — HuggingFace datasets
- `huggingface-hub>=0.22.0` — model downloads
- `zstandard>=0.21.0` — zstd support

---

## Edge Cases

- FEN with promoted pieces (second queen, etc.) — must appear in output
- Empty board squares — must NOT appear in output
- Side to move annotation — must always be present
- FEN deduplication — same position from different tiers is included once
- Model output partial/malformed — `parse_board_description` is best-effort
- 70B model on 24GB VRAM — requires CPU offload, very slow

---

## Reproducibility

- Random seed: `42` everywhere (dataset sampling, benchmark positions)
- Training: `seed=42` passed to SFTTrainer
- Dataset: 5,000 samples per tier, drawn from existing benchmark JSON files with `random.seed(42)`
- LoRA rank: 16, alpha: 32
- Quantization: 4-bit NF4 (Unsloth default)
- GGUF export: `q4_k_m`
- Python-chess version pinned in requirements.txt
