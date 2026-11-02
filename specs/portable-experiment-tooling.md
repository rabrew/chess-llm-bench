# Portable experiment tooling
## Goal
Remove machine-specific repository paths and prevent model tags becoming executable SQL helper code. Keep historic analysis labels explicit and parser tests discoverable.
## Inputs / Outputs
CHESS_BENCH_ROOT optionally overrides repository root; otherwise derive from the script path. OLLAMA_MODELS optionally overrides model storage.
## Steps / Logic
Resolve repository before relative operations; pass model names as Python argv and SQL parameters. Check shell syntax without executing live scripts. Detect duplicate test method definitions with AST.
## Edge cases
Repository paths containing spaces, arbitrary working directories, unknown tags, missing overrides, duplicate test names. Never execute lifecycle scripts during validation.
## Dependencies
Bash, Python standard library, pytest.
