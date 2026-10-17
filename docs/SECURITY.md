# Security and privacy

## Trust boundaries

- The Streamlit app handles credentials, session metadata, uploaded data, and LLM requests.
- PostgreSQL stores account and session state. Uploaded dataset contents are kept in Streamlit session state, not in PostgreSQL.
- Groq is an external processor. The app sends question and dataset context according to the selected privacy tier.
- Generated Python is untrusted. It must execute only through the separate executor API and isolated worker boundary.

## Generated-code execution

Never replace the executor with in-process `exec()` or a plain subprocess and treat it as a sandbox. The worker service runs each request inside a constrained container and requires a production `runsc` runtime or rootless daemon. Its invocation disables networking, uses a read-only root, drops Linux capabilities, prevents privilege gain, uses a non-root UID, and applies CPU, memory, PID, temporary filesystem, execution-time, request-size, response-size, and concurrency limits.

The executor service still depends on host configuration. Operators must:

- Run it on a dedicated host or VM and keep Docker's API/socket private to that host.
- Configure and verify gVisor `runsc` or rootless Docker.
- Keep the executor API private and protect it with TLS and network access controls.
- Use a random shared bearer token with at least 32 characters, store it in a secret manager, and rotate it after exposure.
- Patch the host kernel, Docker Engine, runtime, and worker dependencies.
- Avoid mounting application secrets, host directories, or credentials into worker containers.

The worker image is not itself the security boundary. A default `runc` deployment with the test override is for controlled development/CI only.

## Data sent to Groq

The sidebar provides privacy tiers. Tier 0 sends schema, column types, shape, and the user's question. Tier 1 also sends an aggregate profile. Tier 2 can include up to three sample rows and computed results. Column names can themselves contain sensitive information. Select the minimum tier needed and do not upload data unless your organization permits the configured external processing.

The app does not promise that data is anonymized. Review Groq's current terms and your organization's data-processing rules independently before using real personal or regulated data. The app's local computation and worker do not automatically imply that no data leaves the machine: prompts and selected context are sent to the configured model provider.

## Accounts and sessions

- Passwords are stored as Argon2 hashes; plaintext passwords are not stored.
- PostgreSQL sessions are validated against server-side records and expire. The app also enforces login attempt controls and a daily analysis quota.
- Use a least-privilege database role for the app and a separate migration owner.
- Set `SIGNUP_INVITE_CODE` where public registration is not intended.
- Email verification and password recovery are not implemented. Account recovery requires an operator-approved database process.
- Keep database and executor traffic on private networks and use TLS for remote connections.

## Upload and resource controls

Current application checks cap each file at 20 MiB, datasets at 500,000 rows and 200 columns, and normalized Parquet data at 16 MiB by default. The executor separately limits worker resource use. Reverse proxies and hosting platforms should also set request body, connection, CPU, memory, and concurrency limits appropriate to the deployment.

## Incident response

If an app/provider/database/executor secret is exposed, revoke or rotate it at the issuing service, update the secret store and both sides of any shared credential, restart affected services, and review access logs. Do not include the secret or dataset contents in incident tickets. Preserve only the diagnostic metadata needed by your incident process.
