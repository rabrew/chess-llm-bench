#!/bin/bash
# Live progress monitor — refreshes every 3 seconds
# Usage: bash monitor_progress.sh

while true; do
    printf '\033[H\033[J'
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║              CHESS LLM BENCHMARK — LIVE PROGRESS                    ║"
    printf "║  %-68s║\n" "$(date)"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""

    python3 - <<'EOF'
import sqlite3, sys

DB = "/mnt/shared/chess-llm-bench/jobs/jobs.db"
ORDER = [
    ("llama3.2:3b",     "3B"),  ("gemma3:4b",        "4B"),
    ("qwen2.5:7b",      "7B"),  ("mistral:7b",        "7B"),
    ("deepseek-r1:7b",  "7B"),  ("wizardlm2:7b",      "7B"),
    ("llama3.1:8b",     "8B"),  ("gemma3:12b",       "12B"),
    ("qwen2.5:14b",    "14B"),  ("phi4:14b",         "14B"),
    ("deepseek-r1:14b","14B"),  ("solar:10.7b",      "11B"),
    ("qwen2.5:32b",    "32B"),  ("codellama:34b",    "34B"),
    ("yi:34b",         "34B"),  ("command-r:35b",    "35B"),
    ("mixtral:8x7b",   "47B"),  ("llama3.3:70b",     "70B"),
    ("qwen2.5:72b",    "72B"),
]

try:
    conn = sqlite3.connect(DB, timeout=5)
    cur = conn.cursor()
    cur.execute("SELECT model, status, COUNT(*) FROM jobs GROUP BY model, status")
    rows = cur.fetchall()
    cur.execute("SELECT model, position_id FROM jobs WHERE status = 'in_progress' LIMIT 1")
    active_row = cur.fetchone()
    conn.close()

    def tier_from_pos(pos_id):
        if pos_id is None: return "?"
        if pos_id <= 2080883: return "easy"
        if pos_id <= 4117552: return "medium"
        if pos_id <= 5454662: return "hard"
        return "extreme"

    active_tier = tier_from_pos(active_row[1] if active_row else None)
except Exception as e:
    print(f"  DB unavailable: {e}")
    sys.exit()

from collections import defaultdict
data = defaultdict(lambda: defaultdict(int))
for model, status, count in rows:
    data[model][status] = count

BAR_WIDTH = 24

print(f"  {'Model':<22} {'Size':>4}  {'Progress':<28} {'Done':>5} {'Fail':>5} {'Active':>6}")
print("  " + "─" * 72)

total_done = total_jobs = 0
cur_model = None

for model, size in ORDER:
    d = data.get(model)
    if not d:
        print(f"  {model:<22} {size:>4}  {'— waiting —'}")
        continue

    done   = d.get("done", 0)
    fail   = d.get("failed", 0)
    active = d.get("in_progress", 0)
    pend   = d.get("pending", 0)
    total  = done + fail + active + pend

    total_done += done
    total_jobs += total

    filled = int(BAR_WIDTH * done / total) if total else 0
    bar    = "█" * filled + "░" * (BAR_WIDTH - filled)
    pct    = int(100 * done / total) if total else 0

    if active > 0 and cur_model is None:
        cur_model = model
        tag = f"  ◀ RUNNING ({active_tier})"
    elif pend == 0 and active == 0 and done > 0:
        tag = "  ✓"
    else:
        tag = ""

    print(f"  {model:<22} {size:>4}  [{bar}] {pct:>3}%  {done:>5}/{total:<5} {fail:>5} {active:>6}{tag}")

print()
if cur_model:
    print(f"  Currently running : {cur_model} ({active_tier})")
overall_pct = int(100 * total_done / total_jobs) if total_jobs else 0
print(f"  Overall progress  : {total_done}/{total_jobs} jobs ({overall_pct}%)")
EOF

    sleep 3
done
