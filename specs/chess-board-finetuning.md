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
- Input: `data/board_dataset.jsonl`, HuggingFace model weights, `config/finetune_config.yaml`
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

## Board State Utility (`src/board_state.py`)

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

1. `python scripts/generate_board_dataset.py` — create `data/board_dataset.jsonl`
2. `python scripts/finetune_model.py <ollama_base>` — one model per invocation
3. `python scripts/import_finetuned_to_ollama.py <ollama_base>` — register in Ollama
4. `python scripts/generate_jobs.py --config config/ft_bench_config.yaml` — generate jobs
5. `python scripts/run_workers.py --config config/ft_bench_config.yaml` — run benchmark
6. `python scripts/compare_finetuning_results.py` — compare results

---

## Benchmark Re-run (ft-only)

- Base model results already exist in `results/evaluations.jsonl` — NOT re-run
- Fine-tuned models run with same `max_positions_per_tier: 1000` and `random_seed: 42`
- Results stored separately: `results/finetuning_experiment/evaluations.jsonl`
- Comparison joins on `(position_id, prompt_format)`

---

## Comparison Methodology

For each (model_base, position_id, prompt_format):
- Delta T1: direction accuracy improvement
- Delta T2: move legality rate, clamped CPL change
- Delta T3: explanation score change
- T0 (new): board description accuracy from `verify_board_description()`

---

## VRAM Constraints (RTX 4090, 24GB)

| Size | Strategy | Est. time |
|---|---|---|
| 3–14B | Standard 4-bit QLoRA | 30 min – 2 hr |
| 30–35B | 4-bit + gradient checkpointing | 2–5 hr |
| mixtral:8x7b | 4-bit QLoRA (~12B active) | 1–2 hr |
| llama3.3:70b | 4-bit + CPU offload | Run overnight |

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
