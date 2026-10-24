#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
mkdir -p results/logs
LOG="results/logs/run_all_$(date +%Y%m%d_%H%M%S).log"
bash bin/run_all.sh 2>&1 | tee "$LOG"
