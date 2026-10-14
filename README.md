# AI Data Analyst

Streamlit application for asking questions about uploaded CSV and Excel datasets. It uses PostgreSQL for user accounts and sessions, a LangGraph workflow with Groq for analysis planning and code generation, and a disposable Docker worker to execute generated Python.

## Security and execution requirements

The analysis worker is a separate container. It receives only the requested code and a JSON copy of the current DataFrame. It has no network, no host filesystem mounts, a read-only root filesystem, a non-root user, dropped Linux capabilities, and limits of 512 MB RAM, one CPU, 64 processes, and 25 seconds. Output is capped. A failed analysis does not commit DataFrame changes.

The app requires access to PostgreSQL, a Docker daemon, and the `ai-data-analyst-executor:latest` image. The worker image must be built before starting the app:

```bash
docker build -f Dockerfile.executor -t ai-data-analyst-executor:latest .
```

For the app itself, use Python 3.10 or 3.12 and install pinned dependencies:

```bash
conda create -n agentic python=3.10 -y
conda activate agentic
pip install -r requirements.txt
```

### Start local PostgreSQL

Create a `postgres.env` file (git-ignored) with local database credentials:

```text
POSTGRES_DB=analyst_auth
POSTGRES_USER=analyst
POSTGRES_PASSWORD=chooseAlongLocalPassword123
```

You can copy `postgres.env.example` as a starting point, then replace the placeholder password with a long alphanumeric value.

Start PostgreSQL in Docker with a persistent volume:

```powershell
docker run --name analyst-auth-db --restart unless-stopped -p 127.0.0.1:5432:5432 `
  --env-file postgres.env -v analyst-auth-data:/var/lib/postgresql/data -d postgres:16
```

Copy `.env.example` to `.env` and set the app settings:

```text
GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-20b
DATABASE_URL=postgresql://analyst:chooseAlongLocalPassword123@localhost:5432/analyst_auth
```

The app applies [`schema.sql`](schema.sql) at startup to create `auth_users` and `auth_sessions`. The configured database role needs permission to create tables and indexes in the target schema. For a remote database, use a TLS-verified connection URL and keep port 5432 private.

### Run the app

Then start Streamlit:

```bash
streamlit run app.py
```

When running the app in Docker, the app container needs access to the Docker daemon socket so it can launch the isolated worker. This grants the app process control of that daemon; generated analysis code is never given the socket. Mount the host socket only in a deployment where the app is trusted, and never mount it into the executor container.

The worker image is independent of the application image. Build it on the same Docker daemon that the app uses. Set `ANALYSIS_EXECUTOR_IMAGE` if you use a different image tag.

For a containerized app, build both images and pass the Docker socket to the trusted app container:

```bash
docker build -f Dockerfile.executor -t ai-data-analyst-executor:latest .
docker build -t ai-data-analyst-app .
docker run --rm -p 8501:8501 --env-file .env.docker \
  -v /var/run/docker.sock:/var/run/docker.sock ai-data-analyst-app
```

Create `.env.docker` from `.env` and change the database host in `DATABASE_URL` from `localhost` to `host.docker.internal`; the local Streamlit process should continue using `localhost`.

## Data handling

- Only authenticated users can upload and process files.
- CSV and XLSX uploads are limited to 20 MB, 500,000 rows, and 200 columns.
- Repeated Streamlit reruns do not reload an unchanged file over the working DataFrame.
- Successful worker results commit the returned DataFrame; failures discard the worker's isolated copy.
- Generated charts are interactive Plotly visuals with hover details, zoom, pan, responsive sizing, and legend controls. Distribution requests are guided to include both counts and percentages.
- Column names, types, shape, and the user's query are sent to Groq for analysis. Dataset values, sample rows, and computed result values stay in the app by default. The sidebar checkbox explicitly opts into sharing up to three rows and computed results for model-generated summaries.
- Passwords are stored as Argon2 hashes. Random session tokens are held server-side, hashed in PostgreSQL, expire after 12 hours, and are checked against the database on reruns.

## Workflow

1. Classify the request as analysis, general chat, or unsupported.
2. Plan the analysis and generate code.
3. Perform syntax and unsafe-construct checks, then execute in the isolated worker.
4. Retry failed code up to three attempts, then return a terminal failure message.
5. Explain successful output and display interactive Plotly charts returned by that execution.

The code checks are defense in depth. Container isolation is still required. If Docker or the worker image is unavailable, analysis fails closed.

## Development and verification

Install the application and test dependencies, build the executor image, then run:

```bash
pip install -r requirements-dev.txt
```

```bash
pytest
python verify_graph_step2.py
```

The verification script uses a mocked LLM and does not require a Groq key. The Docker worker tests require Docker. `verify_real_agents.py` is an optional manual check that makes real Groq requests.

Firebase accounts are not migrated automatically. Users need to register new accounts. Password reset and email verification are not implemented in this app; plan account recovery before exposing registration to a wider audience.
