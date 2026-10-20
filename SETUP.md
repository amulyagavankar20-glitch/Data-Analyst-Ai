# Project setup and architecture

This guide explains how the application is structured and how to run it locally on Windows with Conda, PostgreSQL, Groq, and Docker Desktop. For a production deployment, follow [the deployment guide](docs/DEPLOYMENT.md); the local executor setup below uses standard Docker `runc` and is for development only.

## 1. How the project works

```mermaid
flowchart LR
    Browser[Browser] --> App[Streamlit app<br/>app.py]
    App --> Auth[PostgreSQL auth and quotas<br/>src/auth.py]
    App --> Workflow[LangGraph workflow<br/>src/agents/graph.py]
    Workflow --> Groq[Groq API<br/>intent, planning, code, explanation]
    Workflow -->|Parquet data + generated code| Executor[Executor API<br/>executor_service/server.py]
    Executor -->|short-lived bounded container| Worker[Analysis worker<br/>src/agents/executor_worker.py]
    Worker --> Toolkit[Approved helpers<br/>src/agents/toolkit.py]
    Worker -->|result, evidence, Plotly figures| Executor
    Executor --> Workflow
    Workflow --> App
```

### Main request lifecycle

1. The Streamlit app checks the user's PostgreSQL-backed login session and receives a question and optional tabular uploads.
2. Upload readers validate file types, byte size, shape, and format before creating a DataFrame. Uploaded tables and conversation state live in the Streamlit session, not in PostgreSQL.
3. LangGraph classifies the request. Chat/advice can return directly; analysis or modification requests are planned and code is generated through Groq.
4. Generated code is checked, then sent with a serialized DataFrame to the separate executor API. The API starts a short-lived worker container with bounded CPU, memory, processes, time, input/output, and no network access. The worker exposes pandas, Plotly, and vetted statistics/churn helpers.
5. The app displays the computed result, execution evidence, and interactive charts. Analysis requests leave the dataset unchanged; only a successful modification request commits a new DataFrame version.

The executor is a security boundary and must remain a separate service. Never run generated code in the Streamlit process or give the app access to the Docker socket. On a production executor host, use gVisor `runsc` or rootless Docker, private networking, and TLS. The local Windows instructions use a development-only `runc` override and do not provide equivalent isolation.

## 2. Prerequisites

- Windows 10/11 with PowerShell.
- Miniconda or Anaconda.
- Python 3.12, installed through Conda below.
- PostgreSQL 15+ (local installation or a reachable PostgreSQL server).
- A Groq API key and a model available to that account.
- Docker Desktop with the Linux engine running. The executor needs Docker; the Streamlit app can run directly in Conda.

## 3. Create the environment and install packages

Run these commands from the repository root:

```powershell
conda create -n agentic python=3.12 -y
conda activate agentic
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

If the `agentic` environment already exists, just activate it and install the requirements. `requirements-dev.txt` includes the application dependencies and development/test tools.

## 4. Prepare PostgreSQL

Create an empty database (for example, `analyst_auth`) and a database role that the application can use. The app needs its auth and usage tables; do not use a PostgreSQL superuser as the app's runtime account.

Apply the versioned migration from the repository root. Supply a migration-owner connection string in the temporary `MIGRATION_DATABASE_URL` environment variable:

```powershell
$env:MIGRATION_DATABASE_URL = "postgresql://migration_owner:URL_ENCODED_PASSWORD@localhost:5432/analyst_auth"
python migrate.py
Remove-Item Env:MIGRATION_DATABASE_URL
```

Use URL encoding for reserved characters in a database password; for example, `@` becomes `%40`. Then grant the application role the required table and sequence access, as described in [PostgreSQL setup](docs/DEPLOYMENT.md#2-postgresql-setup). The application connection goes in `DATABASE_URL`; migrations use the separate migration-owner connection. `schema.sql` is a reference snapshot; `migrations/001_auth.sql` is the migration applied by `migrate.py`.

## 5. Configure the app

Copy the example file, then edit `.env` with your own values. Never commit this file:

```powershell
Copy-Item .env.example .env
```

Set at least:

```dotenv
GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-20b
DATABASE_URL=postgresql://analyst_app:URL_ENCODED_PASSWORD@localhost:5432/analyst_auth
ANALYSIS_EXECUTOR_URL=http://127.0.0.1:8765
ANALYSIS_EXECUTOR_TOKEN=use_the_same_random_secret_in_both_terminals_at_least_32_characters
DAILY_ANALYSIS_LIMIT=100
```

Generate a different, random executor token for your own environment; the value above is only a placeholder. `SIGNUP_INVITE_CODE` can be set if account registration should require an invite. See [.env.example](.env.example) for all app settings. Do not put PostgreSQL superuser credentials or production secrets in source files.

## 6. Build the executor worker image

Start Docker Desktop and confirm its Linux engine is available:

```powershell
docker info
```

Then build the image from the repository root:

```powershell
docker build -f Dockerfile.executor -t ai-data-analyst-executor:latest .
```

The image is only the worker. It does not start the executor HTTP service and it is not used as the Streamlit app image. The executor service is started separately in the next step.

## 7. Start the local executor and app

Use **two PowerShell terminals**, both opened at the repository root and both in the `agentic` Conda environment.

### Terminal A: executor API

Set the same random token configured in `.env`. The server process reads its token from the environment; it does not read `.env` itself.

```powershell
conda activate agentic
$env:ANALYSIS_EXECUTOR_TOKEN = "PASTE_YOUR_RANDOM_SECRET_HERE"
$env:ANALYSIS_EXECUTOR_IMAGE = "ai-data-analyst-executor:latest"
$env:EXECUTOR_RUNTIME = "runc"
$env:EXECUTOR_CI_ALLOW_RUNC = "1"
$env:EXECUTOR_BIND = "127.0.0.1"
python -m executor_service.server
```

Keep this terminal open. The `runc` setting is only for controlled local development on a disposable machine. Do not expose this endpoint to your LAN or the public internet.

### Terminal B: Streamlit app

Ensure `.env` contains the matching executor URL and token, then start Streamlit:

```powershell
conda activate agentic
streamlit run app.py
```

Open the local URL printed in Terminal B (normally `http://localhost:8501`). Create an account, sign in, upload a small non-sensitive supported table, and try a simple analysis. Stop each foreground service with **Ctrl+C** in its own terminal.

## 8. Confirm the services are reachable

In another PowerShell terminal, check the app and executor health endpoints:

```powershell
Invoke-WebRequest http://127.0.0.1:8501/_stcore/health
Invoke-WebRequest http://127.0.0.1:8765/health
```

Both should return HTTP 200 when their service is running. Also verify PostgreSQL connectivity with the app role and complete one small end-to-end analysis; an HTTP health check alone does not prove that database login, Groq, and execution all work.

## 9. Supported uploads and local data handling

The upload layer accepts CSV, TSV, XLS, XLSX, JSON row objects or equal-length column arrays, JSONL/NDJSON row objects, and Parquet. Application limits currently cap combined uploads at 20 MiB, each table at 500,000 rows and 200 columns, and normalized Parquet data at 16 MiB by default. Excel reads the active sheet. These bounds are configurable only where documented; increasing them requires matching memory and request-size capacity across the app and executor.

DataFrames are held in Streamlit session state, not persisted to PostgreSQL. A browser refresh or app restart may end the active dataset session. Notebook exports contain code/metadata, not the uploaded data or execution output; download the dataset separately if needed.

## 10. Troubleshooting

| Symptom | Check |
|---|---|
| `docker info` cannot connect | Start Docker Desktop and wait for the Linux engine to become ready. |
| Executor fails on startup | Confirm Docker is available, the worker image was built, the token is set, and the local-only `EXECUTOR_CI_ALLOW_RUNC=1` override is present for this dev setup. |
| Analysis says executor is unavailable | Check that Terminal A is still running, `/health` responds, and `.env` has the same token and `http://127.0.0.1:8765` URL. |
| Login or registration fails | Check `DATABASE_URL`, PostgreSQL is running, the migration was applied, and the runtime role has the needed grants. URL-encode reserved password characters. |
| Language service unavailable | Check `GROQ_API_KEY`, `GROQ_MODEL`, outbound HTTPS access, model availability, and provider quota. Never paste the key into diagnostics. |
| Upload rejected | Check the supported format and file/row/column limits listed above. |

For production deployment, upgrades, health checks, and executor hardening, continue with [Deployment](docs/DEPLOYMENT.md), [Security and privacy](docs/SECURITY.md), and [Operations](docs/OPERATIONS.md).
