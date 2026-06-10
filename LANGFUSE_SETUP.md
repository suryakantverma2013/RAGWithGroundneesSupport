# Langfuse v2 — Local Docker Deployment & Haystack Integration

Langfuse is an open-source LLM observability platform.  
In this project it tracks every OpenAI embedding call made by `ingest_pdf.py`,
showing token counts, cost in USD, and latency per pipeline run.

---

## Prerequisites

- [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/) installed and running
- The `docker-compose.langfuse.yml` file (already present in this directory)

---

## 1. Deploy Langfuse with Docker

### 1a. Create the volume directory

Langfuse v2 uses one volume — the PostgreSQL database.  
Create the directory before starting the containers:

```powershell
New-Item -ItemType Directory -Force -Path "E:\LangfuseVolumes\postgres"
```

### 1b. Generate secure secrets

The `docker-compose.langfuse.yml` ships with placeholder secrets.  
Replace them before first start:

```powershell
# Run twice — use one value for NEXTAUTH_SECRET, another for SALT
python -c "import secrets; print(secrets.token_hex(32))"
```

Open `docker-compose.langfuse.yml` and replace both placeholder lines:

```yaml
NEXTAUTH_SECRET: replace-this-with-a-random-32-char-secret-string   # ← paste here
SALT: replace-this-with-a-different-random-salt-string               # ← paste here
```

> **Important:** Use two *different* values. These never need to be remembered —
> just make them random. Changing them after first start invalidates all sessions.

### 1c. Start the containers

```powershell
docker compose -f docker-compose.langfuse.yml up -d
```

Docker will pull the images on first run (~400 MB total). When done:

```
✔ Container langfuse-db      Started
✔ Container langfuse-server  Started
```

Langfuse UI is now available at **http://localhost:3000**

---

## 2. Configure Langfuse (first-time setup)

### Step 1 — Create an admin account

1. Open **http://localhost:3000**
2. Click **Sign Up**
3. Enter any email + password (this is local-only, no verification email)
4. You are now logged in as the admin

### Step 2 — Create a project

1. Click **+ New Project** on the home screen
2. Name it (e.g. `milvus-rag`)
3. Click **Create**

### Step 3 — Generate API keys

1. Inside the project, go to **Settings → API Keys**
2. Click **+ Create new API key**
3. Copy both values shown:
   - **Secret Key** → starts with `sk-lf-`
   - **Public Key** → starts with `pk-lf-`

> These are shown only once. Store them in a `.env` file (see section 3).

---

## 3. Changes made to `ingest_pdf.py`

Langfuse tracing is **opt-in** and **zero-code-change** — it activates
automatically when the three environment variables below are present.
No modifications to pipeline logic are needed.

The following block was added near the top of `ingest_pdf.py`,
after the Tesseract OCR section:

```python
# ── optional: Langfuse tracing ────────────────────────────────────────────────
_LANGFUSE_ENABLED = False
_lf_key = os.environ.get("LANGFUSE_SECRET_KEY")
_lf_pub = os.environ.get("LANGFUSE_PUBLIC_KEY")
if _lf_key and _lf_pub:
    try:
        from langfuse.haystack import LangfuseTracer
        from haystack.tracing import enable_tracing
        enable_tracing(LangfuseTracer())
        _LANGFUSE_ENABLED = True
        log.info("Langfuse tracing enabled → %s",
                 os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"))
    except ImportError:
        log.warning("langfuse package not installed — run: uv add langfuse")
```

### Install the package

```powershell
uv add langfuse
```

### Set environment variables

Create a `.env` file in the project root (never commit this file):

```dotenv
OPENAI_API_KEY=sk-proj-...

LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

Or set them in PowerShell for the current session:

```powershell
$env:LANGFUSE_HOST       = "http://localhost:3000"
$env:LANGFUSE_PUBLIC_KEY = "pk-lf-..."
$env:LANGFUSE_SECRET_KEY = "sk-lf-..."
$env:OPENAI_API_KEY      = "sk-proj-..."
```

### Run ingestion with tracing active

```powershell
uv run python ingest_pdf.py "C:\path\to\document.pdf" my_collection
```

The log will confirm tracing is on:

```
[INFO] Langfuse tracing enabled → http://localhost:3000
```

Every `OpenAIDocumentEmbedder` call is now captured as a trace in Langfuse.

### What you see in the dashboard

```
Trace: ingest_pdf  — $0.0193  — 4.1s
  └── OpenAIDocumentEmbedder
        model:         text-embedding-3-large
        input_tokens:  14 832
        output_tokens: 0
        cost:          $0.0193
        latency:       3.8s
        batch_size:    32
```

---

## 4. Start, Stop, and Status commands

| Action | Command |
|--------|---------|
| Start (background) | `docker compose -f docker-compose.langfuse.yml up -d` |
| Stop (keep data) | `docker compose -f docker-compose.langfuse.yml down` |
| Stop + wipe all data | `docker compose -f docker-compose.langfuse.yml down -v` |
| View live logs | `docker compose -f docker-compose.langfuse.yml logs -f` |
| View server logs only | `docker compose -f docker-compose.langfuse.yml logs -f langfuse-server` |
| Check running status | `docker compose -f docker-compose.langfuse.yml ps` |
| Restart server only | `docker compose -f docker-compose.langfuse.yml restart langfuse-server` |

> Run all commands from the directory that contains `docker-compose.langfuse.yml`.

---

## Volume reference

| What | Host path | Container path |
|------|-----------|----------------|
| PostgreSQL data | `E:\LangfuseVolumes\postgres` | `/var/lib/postgresql/data` |

Data persists across `down` / `up` cycles.  
Only `down -v` or manually deleting `E:\LangfuseVolumes\postgres` wipes it.

---

## Port reference

| Service | Host port | Notes |
|---------|-----------|-------|
| Langfuse UI + API | `3000` | http://localhost:3000 |
| PostgreSQL | not exposed | internal container network only |

---

## Upgrading Langfuse

Langfuse v2 applies database migrations automatically on startup.  
To upgrade to a newer v2 patch:

```powershell
docker compose -f docker-compose.langfuse.yml pull
docker compose -f docker-compose.langfuse.yml up -d
```

---

## Troubleshooting

**UI shows "Application error" on first open**  
The server starts before the database is fully ready.  
Wait 10 seconds and refresh — the healthcheck retries handle this.

**Traces not appearing in the dashboard**  
1. Confirm env vars are set: `echo $env:LANGFUSE_SECRET_KEY`
2. Confirm the log line `Langfuse tracing enabled` appeared when running `ingest_pdf.py`
3. Traces are flushed asynchronously — wait ~5 seconds after the run completes before checking the UI
4. Check server logs: `docker compose -f docker-compose.langfuse.yml logs langfuse-server`

**Port 3000 conflict**  
Change the host port in `docker-compose.langfuse.yml`:
```yaml
ports:
  - "3001:3000"   # host:container
```
Then update `LANGFUSE_HOST=http://localhost:3001` in your env vars.
