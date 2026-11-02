# Development and multi-agent workflow

## Local CPU checks

Create a separate environment if needed; do not replace an environment used by
active training:

```bash
python3 -m venv /tmp/chess-bench-dev
/tmp/chess-bench-dev/bin/python -m pip install -r requirements-dev.txt
make check PYTHON=/tmp/chess-bench-dev/bin/python
```

With existing development dependencies installed, `make check` uses
`venv/bin/python`. `make test` and `make lint` run either check separately.
Pytest collects only `tests/`, whose engines/network are mocked and whose file
fixtures use temporary paths. The commands suppress bytecode and pytest/Ruff
caches. Live scripts under `scripts/` are not part of test collection.

`requirements-dev.txt` contains the CPU dependencies for this suite, including
Flask for the dashboard. It deliberately does not include the GPU ONNX runtime,
Unsloth, or fine-tuning stack in `requirements.txt`. Ruff initially enforces
fatal syntax/name checks (`E9`, `F63`, `F7`, `F82`) without mass reformatting or
unrelated style cleanup. GitHub Actions runs the same checks on Python 3.11 and
3.12 for pushes and pull requests with read-only repository permissions. No
model service, research dataset, engine binary, GPU, or API credentials are needed.

## Codex roles

Configured for installed Codex CLI **0.166.0**, using project-scoped standalone
custom agent TOML files under `.codex/agents/`. This is the current supported
format, rather than older `[agents.<role>]` configuration tables. See the
[official OpenAI subagent documentation](https://learn.chatgpt.com/docs/agent-configuration/subagents).

Start a fresh session from this trusted repository:

```bash
codex -C /home/rabrew/Desktop/chess-llm-bench
```

Accept project trust only after inspecting its instructions/configuration;
untrusted projects may not load local configuration. No user-global config or
trust settings are changed by this setup. Restart existing sessions to discover
the new project agents. Model and reasoning settings inherit your session defaults.

The main session acts as lead using `AGENTS.md` and the lead role instructions.
Example task:

> Fix the described bug. Act as lead, delegate implementation to implementation,
> independent review to reviewer, and baseline/final checks to validation. Wait
> for all reports, resolve introduced issues, and summarize the final diff and checks.

The named `lead` agent is also available for delegating coordination within a
larger session; normally keep the main session as lead to preserve the three
available subagent slots. `.codex/config.toml` enables agents and caps concurrent
subagents at three. Agents share files: the lead assigns one writer per file,
then requests independent review after edits finish. Review/validation are
instructed not to edit; this is not a separate read-only sandbox. Every agent
inherits the parent session's approval and sandbox controls.

Use `/subagents` in the CLI to inspect agent threads. `codex resume --last`
continues a previous session. The configuration and instructions persist in
the repository; agents run when a session delegates work, not as background
daemons. If delegation is unavailable, the lead reports that and performs the
same implementation/review/validation phases serially.

Validate local config without running model inference or starting a service:

```bash
codex --strict-config app-server --print-startup-config
```

Do not bypass the sandbox if it cannot start. Correct the host configuration or
approve narrowly scoped commands as appropriate. Agent work uses your existing
Codex access and can consume additional tokens; CI itself runs only CPU tools.

## Setup baseline

Before these changes, the existing environment (Python 3.14) passed all **543**
tests and fatal Ruff checks. One pre-existing python-chess warning references
deprecated `asyncio.DefaultEventLoopPolicy`. No CI, pytest configuration, or lint
configuration existed. Existing dependency requirements use lower bounds, not a
complete reproducible lock file. Preserve ongoing research files and processes
as directed in `AGENTS.md`; validation must not run the benchmark quick start.
