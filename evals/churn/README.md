# Churn evaluation starter set

This is the first small benchmark for a churn-analysis pack. It evaluates the existing workflow on a deterministic, locally generated synthetic subscription snapshot. It does not use a third-party dataset, so there are no external dataset license terms. The fixture is intentionally small and does not represent real customer behavior.

## Dataset

`fixture.py` creates 60 synthetic customers plus one exact duplicate row. The fixture includes plan, region, payment method, auto-renewal, tenure, support tickets, signup cohort, churn status, cancellation date, and deliberately missing values. Its data dictionary is defined next to the generator. It has one fixed as-of date, not plan-change events or monthly billing history.

## Questions and holdout

`cases.json` contains 20 prompts with expected intent and a human-review rubric:

- `dev`: 14 prompts for debugging and prompt/tool iteration.
- `holdout`: 6 prompts (30%) for one final check. Do not inspect or tune against holdout results while iterating on the dev set. Replacing a prompt or changing implementation based on a holdout result means it is no longer an untouched holdout; create a new versioned holdout set for subsequent comparisons.

The set covers overall and segmented churn, class balance, tenure, missingness, duplicates, cohort rates, unsupported-history handling, model advice, target leakage, chart output, and one explicit dataset modification.

## Churn helper semantics

The isolated worker exposes `churn.churn_rate`, `churn.retention_by_cohort`, and `churn.segment_lift` as deterministic descriptive helpers. They accept common true/false, 1/0, yes/no, churned/active, and retained encodings; missing outcomes are excluded and counted, while unexpected non-missing labels raise an error. The denominator is valid rows in the supplied snapshot. These functions do not estimate survival, time-to-event hazard, causal effects, or a forecast. Segment lift returns both a ratio against the overall snapshot rate and a percentage-point difference; small segment counts remain visible in the result.

## Run a diagnostic baseline

The runner makes real Groq and executor calls. Confirm that `.env` contains `GROQ_API_KEY`, `ANALYSIS_EXECUTOR_URL`, and `ANALYSIS_EXECUTOR_TOKEN`, and that the isolated executor is available before running it. These calls can consume provider quota. Use only this synthetic fixture; do not add user data to benchmark results.

From the repository root in the project environment:

```powershell
python -m evals.churn.run --split dev
```

Run all cases (including the held-out cases) only when you intend to record a complete diagnostic snapshot:

```powershell
python -m evals.churn.run --split all
```

Each run writes JSON traces and a blank `.review.csv` under `evals/churn/results/`. The trace records model intent, execution failure, latency, generated code, output, chart type/title/axes, and worker-captured DataFrame shape, schema, code hash, code-referenced columns, and deterministic verification-helper results. Referenced columns come from code syntax and are not guaranteed to have been accessed at runtime. Deterministic helper results confirm the stated operation on input data, not that selected filters or interpretation match the question. It contains only synthetic dataset content, but generated code and provider responses should still be reviewed before sharing.

## Review and score

Review each response against its rubric. Enter `yes` or `no` for `answer_correct`, `evidence_supported`, and (for chart cases) `chart_correct`. Do not mark a response correct just because execution succeeded or the explanation sounds plausible. Check values against independent calculations from the fixture and inspect chart type, fields, title, and labels.

Then summarize:

```powershell
python -m evals.churn.score evals/churn/results/churn-dev-TIMESTAMP.json evals/churn/results/churn-dev-TIMESTAMP.review.csv
```

The report separates intent routing and workflow completion from human-reviewed answer correctness. Manual binary rates include Wilson 95% confidence intervals. With only 14 dev examples or 6 holdout examples, intervals will be wide; these are debugging signals, not a statistically strong product claim. Do not report an accuracy percentage until every included case has been reviewed, and describe the split and sample size alongside the percentage.

This initial set is a harness scaffold, not a validated benchmark. Before making resume claims, add cases from multiple independent datasets, independently verified numeric gold answers, explicit chart-quality checks, and a written protocol for raters. Keep prompts, model ID, dependency versions, fixture version, and run timestamp with each result for reproducibility.
