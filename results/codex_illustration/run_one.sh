#!/bin/bash
# Usage: run_one.sh <prompt-file> <out-prefix>. Codex with every tool off, from an empty folder.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR/empty"
~/.npm-global/bin/codex exec --sandbox read-only --skip-git-repo-check --ephemeral --ignore-user-config \
  --disable shell_tool --disable unified_exec --disable browser_use --disable browser_use_external \
  --disable in_app_browser --disable computer_use --disable apps --disable plugins --disable sleep_tool \
  --disable tool_suggest --disable skill_search -c web_search='"disabled"' \
  --json -o "$2.txt" - < "$1" > "$2.jsonl" 2> "$2.err"
