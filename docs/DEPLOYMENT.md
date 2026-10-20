# Deployment guide

This guide describes a deployment with PostgreSQL, the Streamlit app, Groq, and a separately hosted analysis executor. Keep the app and worker on separate trust boundaries. The app must not have access to the host Docker socket.

## 1. Deployment topology

Deploy these components:

1. PostgreSQL, reachable only by the app and migration operator.
2. Streamlit app, reachable through an HTTPS reverse proxy.
3. Executor API on a dedicated host, reachable only from the app over TLS or a private network with TLS termination.
4. Worker image on the executor host, executed using gVisor `runsc` or rootless Docker.

The executor listens on loopback by default (`EXECUTOR_BIND=127.0.0.1`). Put a TLS reverse proxy in front of it or run it behind a private service network. Do not publish its HTTP port directly to the public Internet.

## 2. PostgreSQL setup

Create a database and a temporary migration owner. Apply the versioned migration using that owner:

```powershell
$env:MIGRATION_DATABASE_URL = "postgresql://migration_owner:URL_ENCODED_PASSWORD@db-host:5432/analyst_auth?sslmode=verify-full"
python migrate.py
Remove-Item Env:MIGRATION_DATABASE_URL
```

The migration is `migrations/001_auth.sql`. `schema.sql` is a reference snapshot and is not the migration runner. Migrations should be applied once per environment, before deploying an app version that depends on them.

Create a separate runtime role with only the required data access. Run as a database administrator and replace role and database names:

```sql
CREATE ROLE analyst_app LOGIN PASSWORD 'GENERATE_A_UNIQUE_SECRET';
GRANT CONNECT ON DATABASE analyst_auth TO analyst_app;
GRANT USAGE ON SCHEMA public TO analyst_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO analyst_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO analyst_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO analyst_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO analyst_app;
```

Set `DATABASE_URL` to the runtime role connection. URL-encode reserved characters in its password (`@` becomes `%40`, for example). For managed PostgreSQL, require certificate verification with the provider's recommended SSL mode and CA configuration. Restrict inbound database access to app and migration hosts.

## 3. Application secrets and configuration

Use a secret manager or the platform's protected environment settings. Do not bake secrets into an image or commit them. Configure:

```text
GROQ_API_KEY=<secret>
GROQ_MODEL=openai/gpt-oss-20b
DATABASE_URL=<least-privilege PostgreSQL URL>
ANALYSIS_EXECUTOR_URL=https://<private-executor-name>
ANALYSIS_EXECUTOR_TOKEN=<random secret, 32+ characters>
DAILY_ANALYSIS_LIMIT=100
SIGNUP_INVITE_CODE=<optional; set to enable invite-only registration>
```

Generate separate high-entropy secrets for the database password, executor token, and invite code. If open account registration is not desired, set `SIGNUP_INVITE_CODE`. `MAX_DATASET_PARQUET_BYTES` may be set to adjust the normalized dataset cap; the default is 16 MiB. The upload widget and application currently cap the combined upload at 20 MiB.

The app uses PostgreSQL to validate sessions on reruns. Streamlit session state contains the uploaded dataset and conversation state, so use a single app replica unless you have configured session affinity. A browser refresh or process restart may end the user's dataset session. Uploaded datasets are not stored in PostgreSQL.

## 4. Executor service

### Host requirements

- Dedicated Linux host or isolated VM with a supported Docker Engine.
- Production runtime configured as `runsc` (gVisor) or rootless Docker.
- Firewall allowing executor traffic only from the app network/reverse proxy.
- TLS between the app and executor, either directly or through a trusted private proxy.
- No shared writable host folders with the app; no Docker socket mounted into the app container.

The executor checks at startup that the configured runtime exists and that the daemon is rootless or the runtime is `runsc`. Do not set `EXECUTOR_CI_ALLOW_RUNC=1` in production; that flag exists only for CI/test environments.

### Build and start

Build the worker image on the executor host or in a trusted image pipeline, then publish it to a private registry or load it on that host:

```bash
docker build -f Dockerfile.executor -t ai-data-analyst-executor:latest .
```

Set the following environment for the executor API process:

```text
ANALYSIS_EXECUTOR_TOKEN=<same secret configured in the app>
ANALYSIS_EXECUTOR_IMAGE=ai-data-analyst-executor:latest
EXECUTOR_RUNTIME=runsc
EXECUTOR_BIND=127.0.0.1
EXECUTOR_PORT=8765
EXECUTOR_CONCURRENCY=2
```

Start it under a service manager such as systemd, with restart-on-failure and the environment injected from a protected secret source:

```bash
python -m executor_service.server
```

The HTTP service exposes unauthenticated `GET /health` for readiness and bearer-token-protected `POST /v1/execute`. Place an authenticated TLS proxy in front of `/v1/execute`; scope the bearer token to the app and rotate it if exposed. The worker itself is run with no network, a read-only root filesystem, dropped capabilities, no-new-privileges, a non-root uid, CPU/memory/PID limits, bounded temporary storage, time limits, and bounded request/output sizes. Review `executor_service/server.py` when changing deployment policy.

### Local development executor

Local use with the standard `runc` runtime is only appropriate for development on a disposable machine. Configure `EXECUTOR_CI_ALLOW_RUNC=1` only for this controlled test case, use a random token, bind to loopback, and do not expose the service to a network. It is not equivalent to gVisor or rootless Docker isolation.

For Windows development with Docker Desktop, build the image, then open two PowerShell terminals. In terminal A, set the same randomly generated token that you will use in terminal B, and start the executor:

```powershell
conda activate agentic
$env:ANALYSIS_EXECUTOR_TOKEN = "PASTE_A_RANDOM_SECRET_OF_32_OR_MORE_CHARACTERS"
$env:ANALYSIS_EXECUTOR_IMAGE = "ai-data-analyst-executor:latest"
$env:EXECUTOR_RUNTIME = "runc"
$env:EXECUTOR_CI_ALLOW_RUNC = "1"
$env:EXECUTOR_BIND = "127.0.0.1"
python -m executor_service.server
```

In terminal B, configure the app with that exact token and the loopback endpoint before launching Streamlit:

```powershell
conda activate agentic
$env:ANALYSIS_EXECUTOR_TOKEN = "PASTE_THE_SAME_RANDOM_SECRET"
$env:ANALYSIS_EXECUTOR_URL = "http://127.0.0.1:8765"
streamlit run app.py
```

For an app deployed in a container, `127.0.0.1` refers to the app container itself. Use a private network hostname for the executor instead. Keep the development-only `runc` configuration off production hosts.

## 5. Build and run the app

Build the app image from the repository root:

```bash
docker build -f Dockerfile -t ai-data-analyst:local .
```

Run one app instance with secrets injected by the deployment platform. Example for a local test only (do not put real secrets directly in shell history):

```bash
docker run --rm -p 8501:8501 --env-file .env ai-data-analyst:local
```

Production should use the platform's secret injection, TLS reverse proxy, restart policy, resource limits, and non-root container controls. The image's default entrypoint starts Streamlit on port 8501. Configure the reverse proxy to forward WebSocket traffic; Streamlit uses WebSockets for its session connection. Do not mount `/var/run/docker.sock` into this container.

The Dockerfile does not bundle `.env`; verify `.dockerignore` remains in place when changing build context rules.

## 6. Health checks and first-run verification

- App: `GET /_stcore/health` on the app's port; Dockerfile also defines this health check.
- Executor: `GET /health` on the private executor endpoint.
- PostgreSQL: run a connection check from the app network using the runtime role.
- Verify the app can register/sign in, upload a small non-sensitive CSV, and complete an analysis.
- Confirm the app logs do not include secrets or raw uploaded rows.
- Confirm worker containers use the expected runtime and exit after each request.

Do not count a successful app container start as proof that PostgreSQL, Groq, or the executor is operational. Test each dependency and one end-to-end analysis before directing users to the deployment.

## 7. Upgrade and rollback

1. Review release changes and migration files.
2. Back up PostgreSQL and validate restoration in a non-production environment.
3. Apply new migrations with the temporary migration owner.
4. Deploy the app image and executor worker image compatible with the same code revision.
5. Run health and end-to-end checks.
6. If rollback is needed, redeploy the prior app/worker images. Database migrations are not automatically rolled back; use a reviewed forward-fix or a tested database restore.

Tag images with immutable release identifiers in production instead of relying on `latest` or `local`.
