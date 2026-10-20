# Operations guide

## Start and stop

For local Conda development:

```powershell
conda activate agentic
streamlit run app.py
```

Stop the foreground app with Ctrl+C. In deployment, use the platform's process manager or container lifecycle controls. The app and executor are separate services; restarting one does not restart the other.

## Health checks

| Component | Check | Expected result |
|---|---|---|
| Streamlit app | `GET /_stcore/health` on port 8501 | HTTP 200 and `ok` |
| Executor | `GET /health` on its private URL | HTTP 200 JSON status |
| PostgreSQL | Connect from app host with runtime role | Successful TLS-authenticated connection |
| Groq | Submit a small non-sensitive prompt in the app | Valid model response |

Health checks are dependency-specific. An app health response does not confirm that its database, model provider, or executor is reachable.

## Logs and diagnostics

Keep application, proxy, executor, database, and container logs in the platform's centralized logging system with access controls and retention limits. Do not log bearer tokens, database URLs, passwords, prompt sample rows, or uploaded contents. The executor intentionally returns generic infrastructure errors; correlate them using deployment timestamps and service-level logs.

## Backups

Back up PostgreSQL regularly and test restore procedures. The database contains account/session and usage data, not user-uploaded datasets. Dataset state is held in a Streamlit session and may be lost when the browser session or process ends. Users should download an updated dataset if they need a durable copy.

Back up deployment configuration and migration history separately from secrets. Store secrets only in the secret manager and verify that backups are encrypted and access-controlled.

## Common problems

### “Isolated analysis service is unavailable”

1. Check that `ANALYSIS_EXECUTOR_URL` is set in the app environment and points to the reachable executor base URL.
2. Check `GET /health` from the app's network namespace.
3. Confirm `ANALYSIS_EXECUTOR_TOKEN` matches on app and executor and is at least 32 characters.
4. Confirm the executor service is running and its firewall/TLS proxy allows the app.
5. Check executor startup logs for missing `runsc` or rootless Docker configuration.
6. Confirm `ANALYSIS_EXECUTOR_IMAGE` exists on the executor Docker host and matches the deployed code revision.

### Groq classification or planning unavailable

Verify `GROQ_API_KEY`, `GROQ_MODEL`, outbound DNS/HTTPS access from the app, provider status, quotas, and model availability. Never display or paste the key in diagnostics. The app may report a generic language-service message for provider/network failures.

### Authentication/database errors

Check `DATABASE_URL`, URL-encode reserved password characters, confirm the PostgreSQL server is reachable, verify TLS settings, and check that migrations were applied. Ensure the runtime role has only the required grants and that `MIGRATION_DATABASE_URL` is not being used as the app connection.

### Dataset upload rejected

Confirm the file is CSV, TSV, XLS/XLSX, JSON records/column arrays, JSONL/NDJSON, or Parquet; combined upload size is limited to 20 MiB, each table to 500,000 rows and 200 columns, and normalized Parquet to 16 MiB by default. Excel currently reads the active sheet. Increase caps only after reviewing app memory, proxy limits, and worker request sizing.

## Deployment maintenance

- Pin app and worker images to release tags or digests.
- Apply database migrations before app versions that depend on them.
- Rotate database and executor secrets on a schedule and after exposure.
- Patch the executor host/runtime and rebuild worker images when dependencies change.
- Monitor CPU, memory, disk, container restarts, database capacity, provider errors, and executor saturation.
- Keep the executor private and verify its isolation runtime after host maintenance.
