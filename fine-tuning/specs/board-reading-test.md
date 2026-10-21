# Spec: Board-Reading Test, Before and After Fine-Tuning (F-8, F-9)

**Closes:** F-8 and F-9 in [../../specs/exhibition-readiness.md](../../specs/exhibition-readiness.md)
**Created:** 2026-10-21

## Goal

Measure how well each model lists the pieces on a board **before** and **after** fine-tuning,
on the same 200 held-out positions, using the models as they actually run in Ollama.

- **F-8:** the current score, `verify_board_description(...)["accuracy"]`, is recall: real pieces
  found ÷ real pieces. Invented pieces don't lower it. Add precision, counts of missing and extra
  pieces, and side-to-move correctness. The existing keys keep their meaning, so nothing that
  already uses them changes.
- **F-9:** the in-training figures were measured on the full-precision model, and only the
  averages were kept. This test runs the compressed (q4_k_m) Ollama models, the ones the
  benchmark uses, and saves every answer so it can be re-marked later.

## Inputs / Outputs

- **Inputs:**
  - `data/board_dataset.jsonl`
  - Ollama at `$OLLAMA_HOST` (default `http://localhost:11434`)
  - model pairs: every base model in `fine-tuning/config/finetune_config.yaml` whose `ollama_ft_tag` exists in Ollama
- **Outputs:**
  - `results/finetuning_experiment/board_t0.jsonl`: one row per (model, position), holding the
    prompt id, the raw response, every score field and the time taken
  - `results/finetuning_experiment/board_t0_summary.csv`: one row per model

## Steps / Logic

### F-8: new fields from `verify_board_description`

| Field | Meaning |
|---|---|
| `accuracy` (unchanged) | recall: correct pieces ÷ real pieces |
| `recall` | same value, under a clearer name |
| `precision` | correct pieces ÷ pieces the model listed (0 if it listed none) |
| `total_pieces` | number of real pieces |
| `missing_count` | real pieces not listed on their square |
| `extra_count` | listed pieces that are not on that square |
| `correct` (unchanged) | no missing and no extra pieces: a **perfect board** |
| `side_to_move_correct` | `True`/`False` from a "Side to move: White/Black" line, or `None` if there isn't one |

### F-9: the test

1. Held-out positions: `split_held_out(load_dataset_records(...), seed=42, n=200)`. These are the
   same 200 positions used during training, and they were never trained on.
2. Prompt: the training instruction, a fixed description of the answer format, then the FEN,
   wrapped in the Alpaca layout used in training. **Every model gets exactly the same prompt text.**
3. Call `/api/generate` with `raw: true`, which skips each model's chat template. The fine-tuned
   models' Ollama templates are inconsistent: some pass the prompt through and Gemma 3 wraps it
   in chat turns, so raw mode is the only way to send identical text.
   - Options: `temperature 0`, `seed 42`, `num_ctx 1024`, `num_predict 300`, `stop ["\n###"]`.
4. Mark each answer with `verify_board_description`, and append the row to the JSONL straight away.
5. Resume: (model, position index) pairs already in the JSONL are skipped.
6. Summary per model:
   - n
   - mean recall and mean precision
   - perfect-board rate, with a Wilson 95% interval
   - mean extra and mean missing pieces
   - side-to-move accuracy (over answers that state it)
   - unparsed rate (answers where no pieces could be read)

## Edge cases

| Case | Behaviour |
|---|---|
| Base model answers in prose, not the format | It parses as few or no pieces. This is counted, and shown as the unparsed rate, not hidden. |
| DeepSeek-R1 writes `<think>` text first | Kept in the raw response. The parser only reads the lines that use the format. |
| Ollama error or timeout for one position | Retry twice. If it still fails, stop with the error, keeping the rows already saved, and rerun to resume. |
| A fine-tuned model is missing from Ollama | Its pair is skipped, with a warning naming it |
| Output file already has a row | Skipped (resume) |

## Dependencies

`requests` (already in requirements) and a running Ollama.

## Reproducibility

Seed 42 for the held-out split and for Ollama sampling, temperature 0, and fixed prompt text.
The Ollama model digest is recorded in each row.

## Limitations

- Raw mode means base models don't get their usual chat wrapping. They are still told the exact answer format.
- One run per model at temperature 0, so there are no run-to-run error bars, only intervals over positions.
