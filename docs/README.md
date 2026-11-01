# Docs

| Document | What it is | Read it if… |
|---|---|---|
| [SIMPLE_SUMMARY.md](SIMPLE_SUMMARY.md) | The whole project in plain language | you want the short version |
| [FINDINGS.md](FINDINGS.md) | Every result, limitations, and how the audit changed the numbers | you want the evidence |
| [CONCLUSIONS.md](CONCLUSIONS.md) | Post-fix conclusions, fine-tuning effects, and unfinished experiment limits | you want the current interpretation |
| [PROJECT_RECORD.md](PROJECT_RECORD.md) | Experiments, decisions, fixes, verification and remaining work | you want the consolidated project audit trail |
| [issues-log.md](issues-log.md) | Every measurement problem found, numbered, with its fix | you want to check how the numbers were corrected |
| [RESEARCH.md](RESEARCH.md) | Research framing and background notes | you want the context and motivation |
| [CHANGES.md](CHANGES.md) | Change history of the project | you want to know what changed when |
| [RUNBOOK.md](RUNBOOK.md) | Step-by-step commands for running the benchmark | you want to run it |
| [SCRIPTS.md](SCRIPTS.md) | What every script in `scripts/` and `bin/` does | you're reading the code |

The tables in FINDINGS and SIMPLE_SUMMARY are generated from `results/metrics/` by
`scripts/render_findings.py`. Running it with `--check` fails if any table no longer matches the data.
