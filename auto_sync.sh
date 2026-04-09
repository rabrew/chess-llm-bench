#!/bin/bash
# Auto-commit and push any code changes every 30 minutes.
# Data files, logs, and results are excluded via .gitignore.

REPO="/home/rabrew/Desktop/chess-llm-bench"
cd "$REPO" || exit 1

# Only proceed if there are staged/unstaged changes to tracked files
if git diff --quiet && git diff --cached --quiet; then
    exit 0
fi

git add -u
git commit -m "chore: auto-sync $(date '+%Y-%m-%d %H:%M')"
git push
