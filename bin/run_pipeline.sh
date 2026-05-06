#!/bin/bash
cd /home/rabrew/Desktop/chess-llm-bench
mkdir -p results/logs
LOG="results/logs/run_all_$(date +%Y%m%d_%H%M%S).log"
bash bin/run_all.sh 2>&1 | tee "$LOG"
