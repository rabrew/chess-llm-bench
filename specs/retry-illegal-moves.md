# Spec: Retry Illegal Moves

## Goal
After a benchmark run completes, re-prompt any model that gave an illegal or missing
move, this time providing the full list of legal moves so the model must pick one.
Results are written to a separate JSONL file so the original data is untouched.

## Inputs / Outputs
- **Input:** `results/evaluations.jsonl` (existing benchmark results)
- **Input:** Ollama running with the relevant models available
- **Output:** `results/evaluations_retried.jsonl` — one record per retried job,
  containing the original job_id, the original illegal move, the retried move,
  whether the retried move was legal, and its CPL

## Steps / Logic
1. Load all records from `evaluations.jsonl`
2. Filter to records where `t2_legal == False`
3. For each record:
   a. Get the FEN and model
   b. Generate all legal SAN moves with `chess.Board(fen).legal_moves`
   c. Build a short prompt: tell the model its move was illegal (or missing),
      show the legal move list, ask it to pick the best one — move only
   d. Call Ollama with the same model
   e. Parse the first token of the response as the move
   f. Validate legality with python-chess
   g. Compute CPL if legal (use pre-computed stockfish_eval from original record)
   h. Write result record to `evaluations_retried.jsonl`
4. Print a summary: how many retried, how many became legal, legal% improvement

## Result Record Fields
- `job_id` — original job_id
- `model`
- `fen`
- `original_move` — what the model gave first time (may be None)
- `retried_move` — what the model gave after seeing legal moves
- `retried_legal` — bool
- `retried_cpl` — int or None (None if not legal or not best move)
- `stockfish_best_move`
- `inference_ms`

## Edge Cases
- Model still returns an illegal move after seeing the list — record as retried_legal=False
- Model not available in Ollama — skip with warning
- FEN unparseable — skip with warning
- Record already in `evaluations_retried.jsonl` — skip (idempotent on re-run)

## Dependencies
- `python-chess` (already in requirements)
- `requests` / Ollama client (already in src/)
- `results/evaluations.jsonl` must exist and have illegal-move records
