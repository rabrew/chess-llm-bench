# Project Changes & Updates

All modifications made to the project from initial commit.

---

## 1. Config — Model Name Fix (`config/config.yaml`)

**Problem:** `llama3.2:8b` does not exist on Ollama.
**Fix:** Changed to the correct tag.

```yaml
# Before
- llama3.2:8b

# After
- llama3.1:8b
```

---

## 2. Job Generator — Batch Insertion to Fix OOM (`src/job_generator.py`)

**Problem:** `populate_job_queue` called `generate_standard_jobs` which built the entire job list (330 million dicts) in memory before inserting anything, causing the process to be OOM-killed.

**Fix:** Replaced the full in-memory generation with a streaming batch approach — generates and inserts 10,000 jobs at a time, keeping memory usage flat regardless of dataset size.

```python
# Before — builds all 330M jobs in RAM first
jobs = generate_standard_jobs(positions, models, prompt_formats)
inserted = job_queue.insert_jobs(jobs)

# After — generates and inserts in batches of 10k
BATCH_SIZE = 10000
batch = []
for pos in positions:
    for model in models:
        for prompt_format in prompt_formats:
            batch.append({ ... })
            if len(batch) >= BATCH_SIZE:
                inserted += job_queue.insert_jobs(batch)
                batch = []
if batch:
    inserted += job_queue.insert_jobs(batch)
```

---

## 3. Job Queue — Single Connection Per Batch (`src/job_queue.py`)

**Problem:** `insert_jobs` called `insert_job` in a loop, which opened and closed a new SQLite connection for every single job. With batches of 10,000 jobs this eventually hit the OS file descriptor limit, causing `sqlite3.OperationalError: unable to open database file`.

**Fix:** Rewrote `insert_jobs` to use a single connection for the entire batch, wrapping all inserts in one transaction and committing at the end. Much faster and eliminates the file descriptor exhaustion.

```python
# Before — new connection per job
def insert_jobs(self, jobs):
    inserted = 0
    for job in jobs:
        if self.insert_job(job):  # opens/closes connection each time
            inserted += 1
    return inserted

# After — one connection, one transaction per batch
def insert_jobs(self, jobs):
    inserted = 0
    with self._connect() as conn:
        for job in jobs:
            try:
                conn.execute("INSERT INTO jobs ...", (...))
                inserted += 1
            except sqlite3.IntegrityError:
                pass  # duplicate, skip
        conn.commit()
    return inserted
```

---

## 4. Ollama Model Storage — Symlink Architecture

**Problem:** All 19 models (~250GB total) couldn't fit in one location:
- Main partition (`/dev/nvme0n1p5`): 295GB total, was running low
- Shared NTFS partition (`/dev/nvme0n1p6`): 339GB total, 107GB used by Games

**Solution:**
1. Moved existing model blobs to the shared partition at `/mnt/shared/ollama_models/`
2. Created a real models directory on main at `/usr/share/ollama/.ollama/models/`
3. Symlinked each individual blob from shared into main's blobs directory — no duplicates, no copies
4. New model downloads (llama3.1:8b, qwen2.5:72b) go directly to main

This way ollama sees all 19 models from one directory while the actual data is split across both drives with zero duplication.

**Final layout:**
```
/usr/share/ollama/.ollama/models/blobs/
  sha256-xxxx -> /mnt/shared/ollama_models/blobs/sha256-xxxx  (17 models)
  sha256-yyyy  (llama3.1:8b — stored directly on main)
  sha256-zzzz  (qwen2.5:72b — stored directly on main)
```

---

## 5. Models Pulled

Full list of 19 models downloaded and verified:

| Model | Size | Tier |
|-------|------|------|
| llama3.2:3b | 2.0 GB | Small |
| gemma3:4b | 3.3 GB | Small |
| qwen2.5:7b | 4.7 GB | Small |
| mistral:7b | 4.4 GB | Small |
| deepseek-r1:7b | 4.7 GB | Small |
| wizardlm2:7b | 4.1 GB | Small |
| llama3.1:8b | 4.9 GB | Medium |
| solar:10.7b | 6.1 GB | Medium |
| gemma3:12b | 8.1 GB | Medium |
| qwen2.5:14b | 9.0 GB | Medium |
| phi4:14b | 9.1 GB | Medium |
| deepseek-r1:14b | 9.0 GB | Medium |
| qwen2.5:32b | 19 GB | Large |
| codellama:34b | 19 GB | Large |
| yi:34b | 19 GB | Large |
| command-r:35b | 18 GB | Large |
| mixtral:8x7b | 26 GB | MoE |
| llama3.3:70b | 42 GB | XL |
| qwen2.5:72b | 47 GB | XL |

---

## 6. New Files Added

| File | Purpose |
|------|---------|
| `RESEARCH.md` | Research question framing and notes for Stripe Young Scientist 2027 |
| `CHANGES.md` | This file — project change log |
