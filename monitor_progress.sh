#!/bin/bash
# Live progress monitor — refreshes every 5 seconds
# Source of truth: results/evaluations.jsonl (DB is wiped per tier, so unreliable for done counts)

cd /home/rabrew/Desktop/chess-llm-bench
source venv/bin/activate 2>/dev/null || true

while true; do
    NOW=$(date "+%a %d %b %H:%M:%S")
    TITLE="CHESS LLM BENCHMARK — LIVE PROGRESS"
    PADDING=$(( 70 - ${#TITLE} - ${#NOW} ))

    BODY=$(python3 - <<'PYEOF'
import json, sqlite3, sys, os
from collections import defaultdict
from datetime import datetime, timezone

RESULTS   = "results/evaluations.jsonl"
DB        = "jobs/db/jobs.db"
TIERS     = ["easy", "medium", "hard", "extreme"]
JOBS_TIER = 6000
TOTAL_PER_MODEL = JOBS_TIER * len(TIERS)  # 24000
BAR_WIDTH = 20

# Model list with display sizes
MODELS = [
    ("llama3.2:3b",      "3B"),
    ("gemma3:4b",        "4B"),
    ("gemma4:e2b",       "2B"),
    ("qwen2.5:7b",       "7B"),
    ("mistral:7b",       "7B"),
    ("wizardlm2:7b",     "7B"),
    ("deepseek-r1:7b",   "7B"),
    ("llama3.1:8b",      "8B"),
    ("gemma3:12b",       "12B"),
    ("gemma4:e4b",       "4B"),
    ("qwen2.5:14b",      "14B"),
    ("phi4:14b",         "14B"),
    ("deepseek-r1:14b",  "14B"),
    ("solar:10.7b",      "11B"),
    ("qwen2.5:32b",      "32B"),
    ("codellama:34b",    "34B"),
    ("yi:34b",           "34B"),
    ("command-r:35b",    "35B"),
    ("mixtral:8x7b",     "47B"),
    ("gemma4:26b",       "26B"),
    ("gemma4:31b",       "31B"),
    ("llama3.3:70b",     "70B"),
    ("qwen2.5:72b",      "72B"),
]

# ── 1. Count done jobs per model+tier from evaluations.jsonl ──────────────
done_counts  = defaultdict(lambda: defaultdict(int))  # model -> tier -> n
model_latest = {}  # model -> latest timestamp (for throughput)

if os.path.exists(RESULTS):
    with open(RESULTS, "rb") as f:
        f.seek(0, 2)
        fsize = f.tell()
        # Read last 2MB for throughput calc (avoid scanning full file twice)
        sample_start = max(0, fsize - 2_000_000)
        f.seek(0)
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                m = r.get("model", "")
                t = r.get("difficulty", "")
                done_counts[m][t] += 1
                ts = r.get("timestamp")
                if ts and (m not in model_latest or ts > model_latest[m]):
                    model_latest[m] = ts
            except Exception:
                continue

# ── 2. Throughput: jobs/sec for the currently active model ────────────────
# Scan last 500 lines of results file for recent timestamps
recent_ts = []
if os.path.exists(RESULTS):
    with open(RESULTS, "rb") as f:
        # tail-like: read last chunk
        f.seek(0, 2)
        size = f.tell()
        chunk = min(size, 200_000)
        f.seek(size - chunk)
        tail_data = f.read().split(b"\n")
    for raw in tail_data[-600:]:
        raw = raw.strip()
        if not raw:
            continue
        try:
            r = json.loads(raw)
            ts = r.get("timestamp")
            if ts:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                recent_ts.append(dt)
        except Exception:
            continue

jobs_per_sec = 0.0
if len(recent_ts) >= 10:
    recent_ts.sort()
    span = (recent_ts[-1] - recent_ts[0]).total_seconds()
    if span > 0:
        jobs_per_sec = len(recent_ts) / span

# ── 3. Active model/tier from jobs DB ────────────────────────────────────
active_model = None
active_tier  = None
db_pending   = {}  # model -> pending count
db_active    = {}  # model -> in_progress count

if os.path.exists(DB):
    try:
        conn = sqlite3.connect(DB, timeout=3)
        cur  = conn.cursor()
        cur.execute("SELECT model, status, COUNT(*) FROM jobs GROUP BY model, status")
        for model, status, cnt in cur.fetchall():
            if status == "in_progress":
                db_active[model] = cnt
                if active_model is None:
                    active_model = model
            elif status == "pending":
                db_pending[model] = cnt
        # Determine active tier from position_id of a sample in_progress job
        if active_model:
            cur.execute(
                "SELECT position_id FROM jobs WHERE model=? AND status='in_progress' LIMIT 1",
                (active_model,)
            )
            row = cur.fetchone()
            if row:
                pos = row[0]
                if pos <= 2080883:   active_tier = "easy"
                elif pos <= 4117552: active_tier = "medium"
                elif pos <= 5454662: active_tier = "hard"
                else:                active_tier = "extreme"
        conn.close()
    except Exception:
        pass

# Fallback: infer active model/tier from results (most recently updated model
# that isn't 24000 complete)
if active_model is None and model_latest:
    candidates = [
        (ts, m) for m, ts in model_latest.items()
        if sum(done_counts[m].values()) < TOTAL_PER_MODEL
    ]
    if candidates:
        candidates.sort(reverse=True)
        active_model = candidates[0][1]
        # active tier = first incomplete tier
        for t in TIERS:
            if done_counts[active_model].get(t, 0) < JOBS_TIER:
                active_tier = t
                break

# ── 4. ETA ────────────────────────────────────────────────────────────────
remaining_active = 0
if active_model:
    remaining_active = TOTAL_PER_MODEL - sum(done_counts[active_model].values())

eta_str = ""
if jobs_per_sec > 0 and remaining_active > 0:
    secs = remaining_active / jobs_per_sec
    h, rem = divmod(int(secs), 3600)
    m2, s  = divmod(rem, 60)
    if h > 0:
        eta_str = f"  ETA for {active_model}: ~{h}h {m2}m  ({jobs_per_sec:.1f} jobs/s)"
    else:
        eta_str = f"  ETA for {active_model}: ~{m2}m {s}s  ({jobs_per_sec:.1f} jobs/s)"

# ── 5. Render ─────────────────────────────────────────────────────────────
BAR_WIDTH = 22
print(f"  {'Model':<22} {'Size':>4}  {'Progress':<24}  {'Pct':>3}  {'Done':>7}/{TOTAL_PER_MODEL:<5}  E M H X  Status")
print("  " + "─" * 84)

grand_done  = 0
grand_total = 0

for model, size in MODELS:
    tier_done  = done_counts.get(model, {})
    total_done = sum(tier_done.values())
    grand_done  += total_done
    grand_total += TOTAL_PER_MODEL

    pct    = int(100 * total_done / TOTAL_PER_MODEL)
    filled = int(BAR_WIDTH * total_done / TOTAL_PER_MODEL)
    bar    = "█" * filled + "░" * (BAR_WIDTH - filled)

    # Per-tier indicators
    tier_icons = []
    for t in TIERS:
        n = tier_done.get(t, 0)
        if n >= JOBS_TIER:
            tier_icons.append("✓")
        elif model == active_model and t == active_tier:
            tier_icons.append("▶")
        elif n > 0:
            tier_icons.append("~")
        else:
            tier_icons.append("·")
    tier_str = " ".join(tier_icons)

    if model == active_model:
        pend = db_pending.get(model, 0)
        act  = db_active.get(model, 0)
        status = f"◀ RUNNING ({active_tier})  p={pend} a={act}"
    elif total_done >= TOTAL_PER_MODEL:
        status = "✓ complete"
    elif total_done == 0:
        status = "· waiting"
    else:
        status = "~ partial"

    print(f"  {model:<22} {size:>4}  [{bar}]  {pct:>3}%  {total_done:>7}/24000  {tier_str}  {status}")

print()
if eta_str:
    print(eta_str)
overall_pct = int(100 * grand_done / grand_total) if grand_total else 0
print(f"  Overall: {grand_done:,}/{grand_total:,} jobs  ({overall_pct}%)")
PYEOF
)
    # Clear then print header + body atomically
    printf '\033[H\033[J'
    printf "╔%s╗\n" "$(printf '═%.0s' $(seq 1 72))"
    printf "║ %s%*s%s ║\n" "$TITLE" "$PADDING" "" "$NOW"
    printf "╚%s╝\n" "$(printf '═%.0s' $(seq 1 72))"
    echo ""
    printf "%s\n" "$BODY"

    sleep 5
done
