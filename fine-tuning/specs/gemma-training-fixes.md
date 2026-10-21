# Spec: Gemma Fine-Tuning Fixes and Benchmark Exclusion Test

**Closes:** the gemma3:4b and gemma4:e2b skips in the 2026-10-20 training batch, and F-1 in [../../specs/exhibition-readiness.md](../../specs/exhibition-readiness.md)
**Created:** 2026-10-21

## Goal

Two models were skipped in the 2026-10-20 batch:

1. **gemma4:e2b and the other Gemma 4 entries:** the `hf_id` values were guesses marked `# VERIFY`. `google/gemma-4e-2b-it` does not exist, and HuggingFace returns 401.
2. **gemma3:4b:** training ran fine up to the end of epoch 1 (step 1238). It then crashed inside the
   held-out accuracy check, in `model.generate`. The traceback is
   `IndexKernel.cu: index out of bounds`, raised in `flex_attention_forward`. Unsloth uses
   PyTorch flex attention for Gemma 3, and its cached-generation path fails partway through training.

Also, the benchmark-exclusion change (`--exclude-benchmark`, `load_benchmark_fens`) was used for the
training data, but it had no test.

## Inputs / Outputs

- `fine-tuning/config/finetune_config.yaml`, with each model entry gaining an optional `attn_implementation`
- `build_load_kwargs(entry, training, use_grad_ckpt, use_cpu_offload) -> dict`: the keyword arguments for
  `FastLanguageModel.from_pretrained`

## Steps / Logic

1. Replace the Gemma 4 IDs with the ones listed on HuggingFace (checked 2026-10-21):

   | Ollama tag | hf_id |
   |---|---|
   | gemma4:e2b | google/gemma-4-E2B-it |
   | gemma4:e4b | google/gemma-4-E4B-it |
   | gemma4:26b | google/gemma-4-26B-A4B-it (MoE: 26B total, about 4B active) |
   | gemma4:31b | google/gemma-4-31B-it |

2. Give every Gemma entry (gemma3 and gemma4) `attn_implementation: "eager"`. This is plain attention,
   with no flex or compiled kernel, so the crashing code path is never used. Gemma models are small
   enough that eager attention fits easily at a 512-token limit.
3. Move the from_pretrained kwargs into `build_load_kwargs`. It adds `attn_implementation` only when the
   entry sets it, so the models that already trained get exactly the same arguments as before.

## Edge cases

| Case | Behaviour |
|---|---|
| Entry has no `attn_implementation` | Key is not passed, so the load is identical to the 2026-10-20 runs |
| `attn_implementation` is not one of `eager`, `sdpa`, `flash_attention_2` | `ValueError` naming the model |
| An hf_id still marked as a guess | The config test fails |
| The old gemma3:4b checkpoint folder from the crashed run | Delete it before retraining, so `--resume` is never used on a crashed run by accident |

## Dependencies

None new. Needs HuggingFace access to the Gemma repos: they are gated, so the licence has to be accepted on the HF account.

## Verification

- `tests/test_gemma_training_fixes.py`:
  - every hf_id in the config has no `VERIFY` marker and is in `Org/Name` form
  - the four Gemma 4 IDs match the table above
  - every gemma entry has `attn_implementation: eager`
  - `build_load_kwargs` leaves out the key for non-Gemma models and rejects bad values
- `tests/test_board_dataset.py` (exclusion):
  - `generate_records` skips excluded FENs
  - `load_benchmark_fens` reads the `fen` field, skips blank lines, and returns an empty set for a missing file
  - when the real `data/board_dataset.jsonl` and `results/evaluations.jsonl` exist, no training FEN is a benchmark FEN (otherwise skipped)
- Live check: gemma3:4b gets past the epoch-1 accuracy check and is loaded into Ollama, and gemma4:e2b loads and trains

---

## Part 2 (2026-10-21, after the second batch)

The second batch found four more problems:

| # | What happened | Cause |
|---|---|---|
| 5 | gemma3:4b trained fine (held-out 97.7% then 99.0%), but Ollama got an 851 MB model | Unsloth writes two GGUFs for multimodal models: the language model `*.Q4_K_M.gguf` and the vision projector `*-mmproj.gguf`. The import took the alphabetically first one, which was the projector. |
| 6 | The good gemma3:4b output was deleted at 14:26 | A later run with the same tag failed straight away, and `_cleanup_dirs` removed the checkpoint and GGUF folders. **Those folders were from the earlier, successful run.** |
| 7 | gemma4:e2b crashed at the end of epoch 1, and its whole hour of training was deleted | Gemma 4's tokenizer is a multimodal *processor*. `tokenizer(prompt)` passes the prompt as the first positional argument, which a processor treats as `images`. |
| 8 | A crash in the per-epoch accuracy check throws away the training | The check runs inside the trainer, so its exception ends training, and cleanup then deletes the checkpoints. |

### Steps / Logic

5. `find_model_gguf(dirs, quantization)` returns the language-model GGUF:
   - drop any file with `mmproj` in its name
   - if one file is left, use it
   - if several are left, use the one whose name contains the quantization (`q4_k_m`, case-insensitive)
   - otherwise raise `FileNotFoundError` naming the folder (none left, or no unique match)
6. `ensure_outputs_free(dirs, resume)` runs **before** loading the model. If any output folder for this tag already exists and `--resume` is not set, it raises `FileExistsError` naming the folders, and nothing is deleted. Cleanup on failure can therefore only remove folders this run created.
7. `evaluate_held_out` calls `tokenizer(text=prompt, return_tensors="pt")`. The keyword works for plain tokenizers and processors.
8. `EpochAccuracyCallback.on_epoch_end` catches an exception from `evaluate_held_out`, logs it at ERROR level with the traceback, and lets training continue. The accuracy check is a progress indicator only. The real board-reading measurement is F-9, which runs on the Ollama models.

### Edge cases

| Case | Behaviour |
|---|---|
| Only an mmproj file exists | `FileNotFoundError`, and nothing is imported |
| `model.Q4_K_M.gguf` and `model.BF16.gguf` both exist | Q4_K_M is chosen |
| Output folder exists, `--resume` set | Allowed (resume needs the checkpoints) |
| Output folder exists from an old crashed run | User deletes it by hand, a deliberate step |

### Verification (`tests/test_gemma_training_fixes.py`)

- mmproj is never chosen, and the quantization match wins among several files
- `ensure_outputs_free` raises when a folder exists and passes with `--resume` or when no folders exist
- `evaluate_held_out` passes the prompt as `text=` (a fake processor fails on positional input)
- the callback logs an error and does not raise when evaluation fails
