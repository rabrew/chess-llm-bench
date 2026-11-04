# OpenAI commercial benchmark: operation and budget
This is an experiment-operation note, not submission prose.

## Frozen default comparison
- 200 positions, 50 per difficulty tier; seed 42; sampled from the primary saved benchmark positions.
- GPT-6 Luna and GPT-6.1 Sol, one combined fen_only prompt per position: 400 calls.
- Eight pilot calls (one position per tier per model) are part of the same main cohort and are reused.
- Low reasoning, default service tier, no tools, no rescue, no automatic retries; 2048 output tokens including reasoning.
- Proprietary parameter counts remain unknown. Historical reference provenance remains unverified; fresh common T2 reference scoring is still required.
- PGN histories were not retained in primary results, so PGN-based prompts are not silently reconstructed.

## Cost controls
The working ceiling is USD20, with a cumulative USD1 pilot sublimit inside it. This is deliberately below the user's EUR40 credit allocation, not a currency conversion or account-balance guarantee.
Verified standard rates on 2026-11-04 (USD per million input/output tokens): Luna 0.10/0.50, Sol 2/10. Source: https://developers.openai.com/api/docs/pricing
The real-data planning estimates were approximately USD0.18 for the pilot and USD8.87 for the full cohort, including its pilot. These are conservative allowances, not actual usage charges.

One external ledger controls both phases. It reserves conservative input plus maximum output charges before sending each request, retains uncertain charges, and resumes without buying already saved responses. Do not delete/reset the ledger or start a different ledger to bypass a safe stop.
The ledger limits this runner's campaign, not unrelated account activity. No API balance or credit access has been verified.

## Plan without spending
From the project root:
```bash
venv/bin/python scripts/run_openai_benchmark.py \
  --output-dir "$HOME/chess-openai-20261104/run" \
  --budget-ledger "$HOME/chess-openai-20261104/budget.json" \
  --phase pilot
```
Planning requires no key, makes no API calls, and creates no run files. Use --phase main to inspect the complete cohort allowance.

## Execute after the key is available
The runner reads OPENAI_API_KEY from the current process environment. Never put it in config, Git, command-line arguments or shared chat. A key exported in one terminal does not automatically appear in another SSH session.

In the same Linux Bash terminal, enter it without echoing or adding it to shell history:
```bash
read -rs -p 'OpenAI API key: ' OPENAI_API_KEY
export OPENAI_API_KEY
```
Run the planning command with --execute appended to collect the pilot. Inspect pilot status, usage, first parses, refusals and truncation before starting --phase main --execute with the same paths and configuration. The main phase requires a successful pilot. Frozen settings/source/prompts must remain unchanged on resume.

If a call times out, returns unreliable usage or hits another safe stop, inspect the durable record and ledger. Do not blindly repeat it: a failed connection can still have incurred a charge.

## Outputs and analysis
The external run directory contains its manifest, per-request records/raw responses and phase summary. These are separate from protected historical results and must be backed up separately from Git.
The existing full local-model audit exporter assumes known local model metadata; do not feed commercial records into it unchanged. Use saved first-attempt observations and matched positions for the subsequent commercial comparison, and rescore selected moves with a documented common reference before reporting T2 quality.
