# Development guide

## Prerequisites

- Windows, macOS, or Linux with Python 3.12.
- PostgreSQL for authentication-related behavior.
- A Groq API key for manual end-to-end model use.
- Docker for executor integration tests. Production worker isolation also needs `runsc` or rootless Docker.

## Environment setup

Create and activate the Conda environment, install pinned project dependencies, copy the example environment file, and fill in local values:

```powershell
conda create -n agentic python=3.12 -y
conda activate agentic
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Apply the database migration as documented in [Deployment](DEPLOYMENT.md#postgresql-setup). Do not use production credentials in local development. Keep `.env` untracked.

## Run locally

```powershell
streamlit run app.py
```

The application requires a reachable PostgreSQL database and executor service for the corresponding features. Model calls require network access to Groq. If Docker Desktop is unavailable, unit tests can still run; Docker-backed integration tests cannot.

## Checks

```powershell
python -m pytest -q
python -m ruff check app.py src executor_service tests
python -m compileall -q app.py src executor_service
```

Integration tests require the `ai-data-analyst-executor:latest` image and an available Docker daemon. They execute worker code inside a container. Use a disposable development environment and never enable `EXECUTOR_CI_ALLOW_RUNC=1` on a production executor.

Tests should mock external Groq requests and use synthetic data. Tests must not require a real provider key, include real dataset values in assertion output, or depend on unstable network behavior.

## Code organization

| Path | Responsibility |
|---|---|
| `app.py` | Streamlit UI, upload validation, session state, workflow invocation, result display |
| `src/auth.py` | PostgreSQL-backed password authentication, sessions, and quota helpers |
| `src/agents/graph.py` | Intent routing and analysis workflow |
| `src/agents/executor.py` | Authenticated client for the isolated execution API |
| `src/agents/executor_worker.py` | Worker-side request handling and generated-code output capture |
| `src/agents/toolkit.py` | Vetted statistics and visualization helpers exposed to generated code |
| `executor_service/server.py` | Bounded HTTP service that starts restricted worker containers |
| `migrations/` | Ordered database migrations |
| `tests/` | Unit and integration tests |

## Change guidelines

- Preserve fail-closed behavior if isolated execution is unavailable.
- Keep secrets out of source, test output, logs, and notebook exports.
- Avoid sending sample values to Groq unless the selected privacy tier permits it.
- Add or update tests when changing behavior; mock provider and executor boundaries.
- Update deployment/security documentation when changing environment variables, network exposure, migrations, worker limits, or data sharing.
- Keep app and worker releases compatible; deploy the pair together when the execution protocol changes.
