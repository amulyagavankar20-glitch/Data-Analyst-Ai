# Data-Analyst-Ai: implementation plan

Sources: **[A]** your audit, **[B]** my two reviews. Each item has a "done when" check so it can be closed objectively.

---

## 1. Backlog

### P0: before any public link

| ID | Issue | Fix | Done when |
|---|---|---|---|
| P0-1 [A,B] | App container mounts `/var/run/docker.sock` (host-root equivalent). A socket proxy alone does not fix it: `containers/create` bodies (binds, privileged) can't be filtered by path rules. | Split into a **separate executor service**: small HTTP API on its own VM/host that accepts only `{code, dataset, limits}`, builds the `docker run` itself (fixed image, fixed flags, no caller-supplied args) and returns JSON. Run Docker rootless or use gVisor/Firecracker there. Web app holds no Docker access, only a service token. | App image has no `docker` binary and no socket mount; `grep -r docker.sock` is empty; a compromised app can only submit jobs. |
| P0-2 [B] | Plain questions can silently overwrite the dataset (`df = df.groupby(...)`) because every success commits the returned frame. | Classifier returns `ANALYZE` / `MODIFY` / `CHAT` / `UNSUPPORTED`. Commit the frame only for `MODIFY`; for `ANALYZE` discard it. Add **Reset to original** and keep the last N dataset versions (undo). | Test: "average age?" with generated code that reassigns `df` leaves the session frame unchanged. |
| P0-3 [B] | Open signup, no verification, no throttling, no quotas; each query launches a 512 MB / 1 CPU container. | Global concurrency semaphore (and queue with a visible "waiting" state), per-user rate limit and daily quota, login lockout/backoff, email verification (or invite codes) before first analysis. | Load test with 20 concurrent sessions stays within host limits; 10 bad logins trigger lockout. |
| P0-4 [B] | Infra failures (Docker down, image missing, timeout, worker crash) are retried 3x through the LLM. | Typed errors: `CodeError` (retry with feedback) vs `InfraError` / `LimitError` (no retry, clear user message). | Test: fake worker raising `InfraError` makes exactly 1 attempt and 0 extra LLM calls. |

### P1: correctness and data integrity

| ID | Issue | Fix | Done when |
|---|---|---|---|
| P1-1 [B] | JSON round-trip corrupts dtypes: `"00501"` becomes `501`, `"007"` becomes `7`, datetimes become strings. Every committed edit repeats it. | Switch to **Parquet** (`pyarrow` in both images; gzip/zstd) or pickle only inside the trusted boundary. Never use `read_json` type inference. | Round-trip test: leading-zero strings, tz-aware datetimes, categoricals, nullable ints, big ints all survive unchanged. |
| P1-2 [A] | Request cap (20 MB JSON) is below what a valid 20 MB upload serializes to; user only sees a failure later. | Serialize **once at upload time** with the real format, measure, and reject with a precise message (rows x cols budget, bytes). Align `MAX_INPUT_BYTES` to measured worst cases (build 5 representative datasets: wide, tall, text-heavy, numeric, mixed). | A 20 MB CSV either loads and analyzes, or is rejected at upload with "X MB after conversion; limit is Y". Never fails mid-analysis for size. |
| P1-3 [A,B] | Duplicate headers: pandas renames CSV/Excel duplicates (`a`, `a.1`), so `columns.is_unique` is always true after parsing. | Read the raw header row first (`csv.reader` / openpyxl first row), reject duplicates or apply a documented rename policy and show the mapping to the user. | Test: CSV with `id,name,id` is rejected or visibly renamed. |
| P1-4 [B] | `exec(code, globals, locals)` with separate dicts: any `lambda`/nested function using `pd`/`df`/`np` raises `NameError` (verified). | Use one namespace dict for globals and locals. | Test: `df.apply(lambda r: pd.isna(r).sum(), axis=1)` runs. |
| P1-5 [B] | Retry prompt omits the failing code, and `safe_error` hides useful detail (which column, which line). | Pass previous code plus a **sanitized** error (exception type, offending column name, line number; never cell values) to codegen. Column names are already shared, so naming the missing column is safe. | Test: KeyError retry prompt contains the missing column name and prior code, and no cell values. |
| P1-6 [B] | Worker catches `Exception` only, so `SystemExit` yields "check deployment". `read_json` of a literal string is deprecated. | `except BaseException`; return `UserCodeExit`. Remove `read_json` as per P1-1. | Test passes. |
| P1-7 [B] | `validate_code` rejects chart-only code (no `print`/`result`), wasting retries. | Accept code that defines a `BaseFigure`, or auto-set `result` to a summary. | Chart-only generation succeeds first try. |

### P2: privacy, UX, observability

| ID | Issue | Fix | Done when |
|---|---|---|---|
| P2-1 [A] | Consent wording is wrong: opt-in also sends sample rows to the **planner and codegen** calls, not just summaries. | Replace with a tiered, accurate disclosure (see §3.1). Always disclose that column names, types, shape and your question go to Groq. | Copy reviewed; tests assert prompts contain no values at tier 0. |
| P2-2 [B] | Default answer is always "Analysis completed", and the result is hidden in a collapsed expander. | Show result inline. At tier 0, build a **template-based summary in code** (tables, headline numbers) instead of an LLM call. | "What is the mean age?" shows the number without a click. |
| P2-3 [B] | "View Analysis Steps" replays chat history; nodes emit no messages. | Store `plan`, `code`, `retry_history` per answer; show them (code in a collapsible block, with copy and download). | Expander shows the real plan and code. |
| P2-4 [B] | No conversation context (planner/codegen see the latest question only). | Pass the last N turns plus a compact **analysis state** (named intermediate tables, prior questions, dataset version). | "Now split that by region" works. |
| P2-5 [B] | With opt-in, up to 1 MB of output is sent to Groq. | Truncate/summarize output to a token budget (e.g. 6k chars) before the LLM call. | Test with a 1 MB print stays under budget. |
| P2-6 [B] | No observability. | Structured logs per run: intent, attempts, error category, latency, worker exit status, tokens. **No dataset values in logs.** | Dashboard or log query shows error rate and p95 latency. |

### P3: hygiene

- Delete `Data_Analyst_Sql.sql` (byte-identical to `schema.sql`). [B]
- Run migrations separately; give the app DB role **no DDL rights**. Stop splitting `schema.sql` on `;`. [B]
- Remove the example password from `.env.example` / README; generate one in the setup instructions. [B]
- Run `docker rm` only on failure paths, not after every success. [B]
- Pin base images by digest, and lock transitive deps (`pip-compile --generate-hashes`). [B]
- Cap `analysis_artifacts` in `session_state` (keep last N, drop older figures). [B]
- Session survives a refresh? Currently the token lives in `session_state` only. Either document it or use a signed cookie. [B]
- Add `ruff`, `mypy` (light), and GitHub Actions CI. [B]

---

## 2. Test plan: prove isolation, don't assume it

Rules:
- **CI must fail, not skip,** when Docker or the image is missing (`if os.getenv("CI"): fail`). Locally, skip is fine.
- Every negative test needs a **positive control** (proof the thing exists and works when allowed), and must assert **why** it failed (error category), not just that it failed. [A]

| Test | How | Assert |
|---|---|---|
| Effective config | Start the worker container detached with a sleeping entrypoint using the **same flags** from `docker_command`, then `docker inspect` | `NetworkMode=none`, `ReadonlyRootfs=true`, `CapDrop=[ALL]`, `Privileged=false`, `Memory=512m`, `PidsLimit=64`, `User=65534`, `Binds` empty, `no-new-privileges`, tmpfs `noexec` |
| Secrets | Put a sentinel in the parent env (`SECRET_SENTINEL=xyz`) and as a file in the app image | Worker output never contains it; `os.environ` has only what the image sets |
| Host/app files | Positive control: the file exists in the app image. In worker: path absent | Error category is "file not found", and `/proc/self/mountinfo` shows no host bind mounts |
| Filesystem writes | Write to `/`, `/worker`, `/tmp` | `/` and `/worker` raise `PermissionError`/`ReadOnly`; `/tmp` works up to 64 MB then `ENOSPC`; executing a binary from `/tmp` fails (noexec) |
| Network | Positive control: same code on a test container with network, succeeds. In worker: `socket.connect(('1.1.1.1',53))` and DNS | `OSError` (unreachable/no route), not a timeout |
| Timeout | `while True: pass`, and `time.sleep(999)` | Returns within limit + slack; `docker ps -a --filter name=analysis-` is empty afterwards |
| Memory | Allocate 2 GB | Worker OOM-killed (exit 137), typed `LimitError`, host unaffected |
| Fork bomb | `os.fork` loop (reach it via the `exec` namespace) | Capped by PIDs limit, no host impact |
| Output flood | Print 100 MB | `LimitError`, process stopped, memory in parent bounded |
| Validator bypass attempts | Corpus of 30+ tricks: `getattr` via `df.pipe`, `pd.io.common.os`, f-string attribute names, `__builtins__` lookups | Either rejected by `validate_code`, **or contained** by the container (test the second property, since the validator is only defense in depth) |
| Data integrity | Round-trip suite from P1-1 | Frames identical |
| Retry logic | Fake worker returning `InfraError` / `CodeError` | Attempt counts and prompt contents as specified in P0-4 / P1-5 |
| Auth | Lockout, expiry, revoke, session fixation attempt, SQL injection strings in email | As expected; no raw DB errors shown |
| Concurrency | 20 parallel `run_isolated_code` calls | Semaphore holds; all containers cleaned up |

Also add the unit tests that currently cannot be collected in a bare environment: put `psycopg` and `langchain_core` in `requirements-dev.txt`, pin the Python version, and run in CI.

---

## 3. Making it a real analyst, not "text in, code out"

For the record: this app has no retrieval, so it isn't RAG. It's a **text-to-code agent**, which is harder to make reliable. The risk is that it looks like a thin wrapper around an LLM. What separates a strong product from a wrapper is **verification, vetted tools, and context**. In priority order:

### 3.1 Tiered data context (privacy and quality)
- **Tier 0:** names, types, shape (default).
- **Tier 1:** a locally computed **profile** (null %, cardinality, min/max/quantiles, top categories, detected semantic types such as id / date / currency / category). Aggregates only; no rows. The model writes much better code with this, and it is far less sensitive than samples.
- **Tier 2:** plus sample rows and results, with an explicit warning that this goes to Groq during planning, code generation and summaries.
- Let users add a **data dictionary** (column meanings, units, metric definitions such as "revenue = qty x price"). Persist per dataset. This is the single biggest quality gain for real data.

### 3.2 Verification loop (the main "better than a wrapper" feature)
1. **Self-consistency:** generate 2 independent solutions; if numeric outputs disagree, rerun a reconciliation step or tell the user.
2. **Programmatic sanity checks:** row counts before/after, NaN handling reported, denominators non-zero, percentages sum to ~100, result shape matches the question type.
3. **Faithfulness check in code:** every number in the final explanation must appear in the execution output (regex). Reject and regenerate otherwise.
4. **Assumption log:** the answer states how nulls, duplicates and outliers were handled.

### 3.3 Vetted analysis toolkit
Free-form pandas code fails in predictable ways. Provide tested helpers in the worker (`stats.compare_groups`, `stats.correlate`, `stats.regress`, `stats.detect_outliers`, `ts.decompose`, `ts.forecast_baseline`, `viz.distribution`, `viz.class_balance`) and tell the model to prefer them. Fewer lines of generated code means fewer bugs, and the statistics are done correctly (effect sizes, confidence intervals, test assumptions, multiple-comparison warnings).

### 3.4 Auto-EDA on upload
Run a deterministic report (distributions, missingness map, correlations, duplicates, constant columns, suspicious outliers, mixed types) with no LLM. Show "Suggested questions" derived from it. This makes the first minute feel intelligent and is free of LLM cost.

### 3.5 Provenance and reproducibility
- Show "How this was computed" (plan + code + data version) on every answer.
- **Export** the session as a runnable notebook/script and download the cleaned dataset.
- Dataset version history with diff ("dropped 12 rows, 2 columns changed type"), and undo.

### 3.6 Interaction quality
- **Clarifying questions** when the request is ambiguous ("revenue" matches 3 columns).
- Honest **refusals/limits**: "n=14, too small for a reliable test", or "causal claims aren't supported by this analysis".
- Streaming progress ("planning, running, checking") rather than one long spinner.
- Multi-table support: let users upload several files and join via DuckDB; this also helps with big data.

### 3.7 Model use
- Cheap model for classification; a stronger model for codegen (e.g. a 120B-class model on Groq) with structured output (JSON schema: `{plan, code, assumptions}`) instead of regex-extracting Markdown.
- Make model names config, and record the model in the eval report.

---

## 4. Evaluation and LLM-as-judge ("is the response better?")

Build this **before** the larger changes in §3 so each change can be proven to help.

### 4.1 Eval set
- 80-150 cases across ~10 datasets (clean, messy, wide, time series, imbalanced, with ID-like strings, with nulls).
- Categories: simple aggregate, group-by, filtering, ranking, time series, correlation/stats, charting, data cleaning (`MODIFY`), multi-step, ambiguous question, unanswerable question, adversarial/prompt-injection in cell values, chit-chat.
- Each case: dataset, question, **reference answer** (value, table or chart spec), expected intent, tolerance, and `must_not` conditions (e.g. must not modify df).

### 4.2 Layer 1: programmatic checks (primary signal, cheap, no judge needed)
Numeric match within tolerance; table equality (order-insensitive); chart spec check (trace type, x/y columns, title and axis labels present); dataset unchanged for `ANALYZE`; intent label correct; no values leaked at tier 0; attempts <= 3; latency; token cost.

### 4.3 Layer 2: LLM judge (for what code can't check)

**Rubric (each 1-5, with written anchors; judge outputs reasoning first, then scores):**

| Dimension | 5 means | 1 means |
|---|---|---|
| Correctness | Matches reference (or is verifiably right if no reference) | Wrong or contradicts output |
| Faithfulness | Every claim is supported by the execution output | Invents numbers or conclusions |
| Relevance and completeness | Answers what was asked, nothing key missing | Off-topic or partial |
| Method soundness | Appropriate technique, handles nulls/outliers/groups properly, no leakage | Misapplied method |
| Insight and clarity | Concise, states the "so what", plain language | Dumps output, vague |
| Visualization quality | Right chart, labeled, readable, uses counts and percentages where relevant | Wrong or cluttered |
| Calibration and caveats | States assumptions and limits; no overclaiming (causation, small n) | Overconfident |

Hard-fail flags (auto zero): fabricated number, ignores the question, destructive change without request, values leaked while opt-out.

**Protocol (the answer to "is it better?"):**
1. **Pairwise comparison** of old vs new pipeline (A/B) on the same cases: the judge sees question, schema, reference, output of A and B, and picks `A`, `B` or `tie` with the rubric. **Run each pair twice with positions swapped**; count only consistent verdicts, others are ties. This controls position bias.
2. Hide system identity, and **normalize length/format** or explicitly instruct "do not prefer longer answers" (verbosity bias).
3. Use a **different model family** for the judge than the generator (e.g. judge with Claude or a 70B Llama while generating with gpt-oss) to reduce self-preference. Temperature 0, JSON-schema output.
4. **Reference-guided**: always give the judge the gold answer and the real execution output so it checks facts instead of guessing.
5. **Calibrate the judge:** hand-label ~40 cases yourself; report judge-vs-human agreement (Cohen's kappa or Spearman) and fix the rubric until agreement is acceptable (kappa >= 0.6). Re-check when the judge model changes.
6. **Statistics:** report win/tie/loss rate with a bootstrap 95% CI and a paired test on rubric scores; run each case 3x to measure run-to-run variance (generation is stochastic even at temperature 0 on hosted models). Use the paired-comparison and sequential-testing ideas from your agent-eval work here.
7. **Shipping gate (CI/nightly):** a change merges only if (a) programmatic pass rate does not drop beyond the CI, (b) judge win-rate vs main is >= 50% with CI lower bound above ~40%, (c) no new hard-fail flags, (d) p95 latency and cost stay within budget.

**Judge prompt skeleton:**
```
You are a strict reviewer of data-analysis answers. Do not prefer longer answers.
Inputs: QUESTION, DATASET SCHEMA, REFERENCE ANSWER, EXECUTION OUTPUT (ground truth of what the code printed), RESPONSE A, RESPONSE B.
1. For each response, list every factual claim and mark it SUPPORTED / UNSUPPORTED by EXECUTION OUTPUT.
2. Compare against REFERENCE for correctness.
3. Score each rubric dimension 1-5 using the anchors; set hard_fail flags.
4. Choose winner: A | B | tie. Explain in <= 3 sentences.
Return JSON: {claims:[...], scores:{A:{...},B:{...}}, hard_fail:{A:[...],B:[...]}, winner, reason}
```

### 4.4 Runtime judge (cheap, always on)
Before showing an answer: run the programmatic faithfulness check (§3.2.3). Optionally a small judge call (only when tier 2 is on) with the question + output + answer asking "does this answer follow from the output? yes/no". On "no", regenerate once, else show the raw result with a note.

### 4.5 User feedback loop
Thumbs up/down plus a reason on each answer, stored without dataset values. Failures become new eval cases.

---

## 5. Suggested order

1. **Week 1: safety and integrity:** P0-2, P0-4, P1-1, P1-2, P1-3, P1-4, P1-5, P1-6; adversarial test suite (§2); CI that fails without Docker.
2. **Week 2: deployability:** P0-1 (executor service), P0-3 (limits, verification, quotas), P2-1 consent rewrite, P3 hygiene.
3. **Week 3: measure:** eval set, programmatic checks, judge + calibration, baseline numbers recorded for the current build.
4. **Week 4+: differentiators**, each gated by the eval: 3.1 profile tier and data dictionary, 3.2 verification loop, 3.3 toolkit, 3.4 auto-EDA, 3.5 provenance/versioning, 3.6 clarifying questions.

Each §3 feature ships only if the judge and programmatic scores improve. That is also the strongest portfolio story: *"I measured that verification raised correctness from X% to Y%."*
