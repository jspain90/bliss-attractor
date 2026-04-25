# Attractor Experiment Harness

AI-to-AI conversation harness for replicating Kyle Fish's attractor state findings.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create the database

Connect to your Postgres instance and run:

```bash
psql -U postgres -f setup_db.sql
```

This creates the `attractor` database with the `runs` and `turns` tables.

### 3. Set environment variables

```bash
# Postgres connection string
export ATTRACTOR_DB_DSN="postgresql://user:password@host:5432/attractor"

# Anthropic API key (only needed for frontier runs)
export ANTHROPIC_API_KEY="sk-ant-..."
```

Add these to your `.bashrc` or `.env` file for persistence.

---

## Running an experiment

```bash
python attractor.py configs/local_v1_1.yaml
```

Press `Ctrl+C` at any time to interrupt gracefully. All completed turns are preserved.

---

## Config files

| File | Description |
|------|-------------|
| `configs/local_v1_1.yaml` | Ollama/Ollama — harness validation |
| `configs/opus_v1_1.yaml` | Claude Opus / Opus — primary replication (v1.1) |
| `configs/opus_v2_1.yaml` | Claude Opus / Opus — Fish interference condition (v2.1, /stop) |

Update `endpoint` in local configs to your Ollama LAN address.

---

## Build sequence

- [x] 1. Database schema (`setup_db.sql`)
- [x] 2. YAML configs + model abstraction (`attractor.py`)
- [x] 3. Ollama/Ollama loop with streaming stdout and Ctrl+C
- [x] 4. Postgres logging
- [ ] 5. Local test run — verify mechanics
- [ ] 6. Set `ANTHROPIC_API_KEY`, verify Anthropic provider
- [ ] 7. Cheap Anthropic model test run (e.g. `claude-haiku-4-5-20251001`)
- [ ] 8. Full Claude Opus replication run

---

## Updating rater assessment

After a run, update the rater fields directly:

```sql
UPDATE runs
SET attractor_observed = true,
    rater_assessment = '{"notes": "Entered stable register at turn 23", "rater": "blind"}'
WHERE run_id = '<run_id>';
```

---

## Querying results

```sql
-- All completed runs with attractor observation
SELECT run_id, model_a, model_b, system_prompt_version,
       attractor_observed, terminated_by
FROM runs
WHERE completed_at IS NOT NULL
ORDER BY created_at DESC;

-- All turns for a specific run
SELECT turn_number, speaker, content->>'message' AS message, token_count
FROM turns
WHERE run_id = '<run_id>'
ORDER BY turn_number;
```
