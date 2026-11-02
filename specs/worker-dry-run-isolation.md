# Worker dry-run isolation
## Goal
Prevent dry runs completing or changing real jobs.
## Inputs / Outputs
Existing SQLite queue; dry_run boolean. Outputs only an in-memory queue snapshot and returned records.
## Steps / Logic
Open source DB read-only and copy into an in-memory JobQueue. Claims/completions and stale resets operate on the snapshot. Do not construct result directories or trigger correction jobs. CLI uses the same snapshot before resetting jobs.
## Edge cases
Missing source fails without creating it. SQLite errors propagate. Multiple dry workers may see the same snapshot; this mode is diagnostic rather than throughput measurement.
## Dependencies
Python sqlite3; temporary fixtures, mocked model calls.
