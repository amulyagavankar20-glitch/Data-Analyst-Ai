"""Small, documented analysis helpers exposed to generated code in the worker."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import plotly.express as px


def compare_groups(frame, value, group, first, second):
    """Welch two-sample t test, Hedges' g, and 95% CI for the mean difference."""
    a = pd.to_numeric(frame.loc[frame[group] == first, value], errors="coerce").dropna().to_numpy(float)
    b = pd.to_numeric(frame.loc[frame[group] == second, value], errors="coerce").dropna().to_numpy(float)
    if min(len(a), len(b)) < 2:
        raise ValueError("Each group needs at least two numeric observations.")
    result = scipy_stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    se = float(np.sqrt(va / len(a) + vb / len(b)))
    df = (va / len(a) + vb / len(b)) ** 2 / ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
    critical = float(scipy_stats.t.ppf(.975, df))
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    d = (float(np.mean(a) - np.mean(b)) / pooled) if pooled else float("nan")
    correction = 1 - 3 / (4 * (len(a) + len(b)) - 9)
    difference = float(np.mean(a) - np.mean(b))
    return {"group_1": str(first), "group_2": str(second), "n_1": len(a), "n_2": len(b),
            "mean_1": float(np.mean(a)), "mean_2": float(np.mean(b)),
            "mean_difference": difference, "ci_95": [difference - critical * se, difference + critical * se],
            "welch_t": float(result.statistic), "p_value": float(result.pvalue),
            "hedges_g": float(d * correction), "assumptions": "Independent observations; Welch test does not assume equal variances. Check distributions and sampling design."}


def grouped_mean(frame, value, group):
    """Compute per-group means without mutating data; report excluded null values."""
    if group not in frame or value not in frame:
        raise KeyError(group if group not in frame else value)
    numeric = _numeric_column(frame, value)
    working = pd.DataFrame({"group": frame[group].to_numpy(), "value": numeric.to_numpy()})
    rows = []
    for name, subset in working.groupby("group", dropna=False, sort=True, observed=True):
        valid = subset["value"].dropna()
        rows.append({"group": "(missing)" if pd.isna(name) else str(name),
                     "rows": int(len(subset)), "n": int(len(valid)),
                     "excluded_missing": int(subset["value"].isna().sum()),
                     "mean": float(valid.mean()) if len(valid) else None})
    return {"group_column": str(group), "value_column": str(value), "groups": rows,
            "definition": "Arithmetic mean; missing values excluded within each group."}


def correlate(frame, columns, method="pearson"):
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    return {"correlation": numeric.corr(method=method).round(4).to_dict(),
            "method": method, "n": int(numeric.dropna().shape[0]),
            "caveat": "Correlation is not evidence of causation."}


def regress(frame, target, predictors):
    data = frame[[target, *predictors]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) <= len(predictors) + 2:
        raise ValueError("Regression needs more complete observations than predictors plus two.")
    x = np.column_stack([np.ones(len(data)), data[predictors].to_numpy(float)])
    y = data[target].to_numpy(float)
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank < x.shape[1]:
        raise ValueError("Predictors are linearly dependent; remove redundant columns.")
    residual = y - x @ beta
    df = len(y) - x.shape[1]
    covariance = (residual @ residual / df) * np.linalg.inv(x.T @ x)
    se = np.sqrt(np.diag(covariance))
    t_values = beta / se
    return {"n": len(y), "r_squared": float(1 - (residual @ residual) / np.sum((y - y.mean()) ** 2)),
            "coefficients": {"intercept": float(beta[0]), **dict(zip(predictors, beta[1:].astype(float)))},
            "p_values": dict(zip(["intercept", *predictors], scipy_stats.t.sf(np.abs(t_values), df).astype(float) * 2)),
            "caveat": "Associational model; check linearity, residuals, independence, and influential points."}


def detect_outliers(series, multiplier=1.5):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0, "lower_bound": None, "upper_bound": None, "method": "IQR"}
    q1, q3 = values.quantile([.25, .75])
    spread = q3 - q1
    lower, upper = float(q1 - multiplier * spread), float(q3 + multiplier * spread)
    return {"count": int(((values < lower) | (values > upper)).sum()),
            "lower_bound": lower, "upper_bound": upper, "method": "IQR rule; flag only, not automatic removal"}


def decompose(series, period):
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    if period < 2 or len(values) < 2 * period:
        raise ValueError("Seasonal decomposition needs at least two complete periods.")
    trend = pd.Series(values).rolling(period, center=True, min_periods=1).mean().to_numpy()
    residual = values - trend
    phase = np.arange(len(values)) % period
    seasonal_pattern = np.array([np.mean(residual[phase == i]) for i in range(period)])
    seasonal = seasonal_pattern[phase]
    return {"trend": trend.tolist(), "seasonal": seasonal.tolist(),
            "residual": (values - trend - seasonal).tolist(), "period": int(period),
            "method": "Additive moving-average decomposition; assumes a fixed seasonal period."}


def forecast_baseline(series, horizon, period=None):
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    if not len(values) or not 1 <= horizon <= 365:
        raise ValueError("Provide numeric observations and a horizon from 1 to 365.")
    if period and 1 <= period <= len(values):
        forecast = np.resize(values[-period:], horizon)
        method = f"Seasonal naive (period={period})"
    else:
        forecast = np.repeat(values[-1], horizon)
        method = "Naive last-value baseline"
    return {"forecast": forecast.tolist(), "horizon": horizon, "method": method,
            "caveat": "Baseline only; no prediction intervals or covariates."}


def distribution(frame, column):
    return px.histogram(frame, x=column, title=f"Distribution of {column}", marginal="box")


def class_balance(frame, column):
    counts = frame[column].value_counts(dropna=False).rename_axis("class").reset_index(name="count")
    counts["percent"] = (counts["count"] / max(len(frame), 1) * 100).round(2)
    return px.bar(counts, x="class", y="count", color="class", text="percent",
                  title=f"Class balance: {column}", hover_data={"percent": True})


def _binary_outcome(frame, column, positive=None):
    """Normalize common binary churn encodings; exclude null outcomes."""
    if column not in frame:
        raise KeyError(column)
    values = frame[column]
    normalized = values.astype("string").str.strip().str.casefold()
    mapping = {"true": True, "false": False, "1": True, "0": False,
               "yes": True, "no": False, "y": True, "n": False,
               "churned": True, "active": False, "retained": False}
    if positive is not None:
        positive_label = str(positive).strip().casefold()
        observed = set(normalized.dropna().unique().tolist())
        if len(observed) != 2 or positive_label not in observed:
            raise ValueError("The selected positive churn value must identify one of exactly two observed labels.")
        result = normalized.eq(positive_label).where(normalized.notna())
    else:
        result = normalized.map(mapping)
    invalid = values.notna() & result.isna()
    if invalid.any():
        examples = values.loc[invalid].astype(str).drop_duplicates().head(3).tolist()
        raise ValueError(f"Column {column!r} must contain a recognized binary churn outcome; unexpected values: {examples}.")
    return result.astype("boolean")


def churn_rate(frame, outcome="churned", positive=None):
    """Return a snapshot churn rate with explicit valid and excluded counts.

    The denominator is rows with a non-missing, recognized binary outcome.
    It is a snapshot proportion, not a time-based churn hazard.
    """
    labels = _binary_outcome(frame, outcome, positive)
    valid = labels.dropna()
    total = int(len(valid))
    churned = int(valid.sum())
    return {"outcome_column": outcome, "churned": churned,
            "retained": total - churned, "n": total,
            "excluded_missing": int(labels.isna().sum()),
            "churn_rate": float(churned / total) if total else None,
            "retention_rate": float((total - churned) / total) if total else None,
            "definition": "Snapshot proportion among rows with a valid binary outcome; not a time-based rate.",
            "positive_value": None if positive is None else str(positive)}


def churn_class_balance(frame, outcome="churned", positive=None):
    """Return both binary classes as counts and percentages of valid outcomes."""
    labels = _binary_outcome(frame, outcome, positive).dropna()
    total = int(len(labels))
    churned = int(labels.sum())
    retained = total - churned
    return {"outcome_column": str(outcome), "n": total,
            "excluded_missing": int(frame[outcome].isna().sum()),
            "classes": [
                {"class": "Churned", "count": churned,
                 "percent": round(100 * churned / total, 2) if total else None},
                {"class": "Active", "count": retained,
                 "percent": round(100 * retained / total, 2) if total else None},
            ],
            "definition": "Percentages use rows with a valid binary outcome."}


def rate_by_bins(frame, column, bins, labels, outcome="churned", positive=None):
    """Calculate binary outcome rates in explicit left-closed numeric bands."""
    if column not in frame:
        raise KeyError(column)
    if len(bins) != len(labels) + 1 or any(left >= right for left, right in zip(bins, bins[1:])):
        raise ValueError("Provide one more increasing bin edge than labels.")
    values = _numeric_column(frame, column)
    band = pd.cut(values, bins=bins, labels=labels, right=False, include_lowest=True)
    outcome_labels = _binary_outcome(frame, outcome, positive)
    working = pd.DataFrame({"band": band, "churned": outcome_labels.to_numpy()})
    working = working.dropna(subset=["band", "churned"])
    grouped = working.groupby("band", observed=False, sort=True)
    records = []
    for name, subset in grouped:
        count = int(len(subset))
        churned = int(subset["churned"].sum())
        records.append({"band": str(name), "n": count, "churned": churned,
                        "churn_rate": churned / count if count else None})
    return {"column": str(column), "outcome_column": str(outcome),
            "excluded_missing_or_out_of_range": int(len(frame) - len(working)),
            "interval_convention": "left-closed, right-open [lower, upper)",
            "bands": records}


def exact_duplicate_summary(frame):
    """Count extra exact duplicate rows across all columns, without counting originals."""
    duplicate_mask = frame.duplicated(keep="first")
    group_mask = frame.duplicated(keep=False)
    return {"operation": "exact_duplicates", "duplicate_rows": int(duplicate_mask.sum()),
            "rows_in_duplicate_groups": int(group_mask.sum()),
            "definition": "Exact full-row matches across all columns; each repeated row after its first copy counts once."}


def retention_by_cohort(frame, cohort, outcome="churned", positive=None):
    """Summarize cohort counts, churn, and retention using valid outcomes only."""
    if cohort not in frame:
        raise KeyError(cohort)
    labels = _binary_outcome(frame, outcome, positive)
    working = pd.DataFrame({"cohort": frame[cohort].to_numpy(),
                            "churned": labels.to_numpy()})
    working = working.dropna(subset=["churned"])
    result = []
    for name, group in working.groupby("cohort", dropna=False, sort=True, observed=True):
        count = int(len(group))
        churned = int(group["churned"].sum())
        result.append({"cohort": "(missing)" if pd.isna(name) else str(name),
                       "n": count, "churned": churned, "retained": count - churned,
                       "churn_rate": churned / count,
                       "retention_rate": (count - churned) / count})
    summary = {"cohort_column": cohort, "outcome_column": outcome,
               "excluded_missing_outcome": int(labels.isna().sum()), "cohorts": result,
               "definition": "Snapshot churn and retention proportions by cohort; not a survival estimate.",
               "positive_value": None if positive is None else str(positive)}
    if cohort not in summary:
        summary[cohort] = result
    return summary


def segment_lift(frame, segment, outcome="churned", positive=None):
    """Compare segment snapshot churn proportions with the overall proportion.

    ``lift_ratio`` is segment rate / overall rate; ``difference_pp`` is the
    absolute percentage-point difference. Small segment counts are exposed.
    """
    if segment not in frame:
        raise KeyError(segment)
    labels = _binary_outcome(frame, outcome, positive)
    working = pd.DataFrame({"segment": frame[segment].to_numpy(),
                            "churned": labels.to_numpy()})
    working = working.dropna(subset=["churned"])
    total = int(len(working))
    overall_churned = int(working["churned"].sum())
    baseline = overall_churned / total if total else None
    result = []
    for name, group in working.groupby("segment", dropna=False, sort=True, observed=True):
        count = int(len(group))
        churned = int(group["churned"].sum())
        rate = churned / count
        result.append({"segment": "(missing)" if pd.isna(name) else str(name),
                       "n": count, "churned": churned, "churn_rate": rate,
                       "lift_ratio": rate / baseline if baseline else None,
                       "difference_pp": (rate - baseline) * 100 if baseline is not None else None})
    summary = {"segment_column": segment, "outcome_column": outcome,
               "overall_churn_rate": baseline,
               "excluded_missing_outcome": int(labels.isna().sum()), "segments": result,
               "definition": "Descriptive snapshot comparison; lift is not causal and may be unstable for small groups.",
               "positive_value": None if positive is None else str(positive)}
    if segment not in summary:
        summary[segment] = result
    return summary


def _numeric_column(frame, column):
    if column not in frame:
        raise KeyError(column)
    source = frame[column]
    numeric = pd.to_numeric(source, errors="coerce")
    if (source.notna() & numeric.isna()).any():
        raise ValueError(f"Column {column!r} contains non-numeric values.")
    return numeric


def verify_row_count(frame):
    """Deterministically report the input row count."""
    return {"operation": "row_count", "value": int(len(frame)), "rows": int(len(frame))}


def verify_aggregate(frame, column, operation="mean"):
    """Deterministically compute a column sum, mean, or non-missing count."""
    if column not in frame:
        raise KeyError(column)
    if operation == "count":
        values = frame[column]
        return {"operation": "count", "column": str(column),
                "value": int(values.count()), "excluded_missing": int(values.isna().sum())}
    if operation not in {"sum", "mean"}:
        raise ValueError("Supported aggregate operations are sum, mean, and count.")
    values = _numeric_column(frame, column)
    count = int(values.count())
    value = float(values.sum()) if operation == "sum" else (float(values.mean()) if count else None)
    return {"operation": operation, "column": str(column), "value": value,
            "n": count, "excluded_missing": int(values.isna().sum())}


def verify_grouped(frame, group, column=None, operation="count"):
    """Deterministically compute grouped counts, sums, or means."""
    if group not in frame:
        raise KeyError(group)
    if operation not in {"count", "sum", "mean"}:
        raise ValueError("Supported grouped operations are count, sum, and mean.")
    if operation == "count":
        values = frame[group] if column is None else frame[column]
        if column is not None and column not in frame:
            raise KeyError(column)
        present = pd.Series(values.notna().to_numpy(dtype=bool), dtype="float64").replace({0.0: np.nan})
        working = pd.DataFrame({"group": frame[group].to_numpy(), "value": present.to_numpy()})
    else:
        if column is None:
            raise ValueError("A value column is required for grouped sum or mean.")
        values = _numeric_column(frame, column)
        working = pd.DataFrame({"group": frame[group].to_numpy(), "value": values.to_numpy()})
    grouped = working.groupby("group", dropna=False, sort=True, observed=True)["value"]
    if operation == "count":
        result = grouped.count()
    elif operation == "sum":
        result = grouped.sum(min_count=1)
    else:
        result = grouped.mean()
    records = []
    for name, value in result.items():
        group_value = None if pd.isna(name) else name.item() if hasattr(name, "item") else name
        numeric_value = None if pd.isna(value) else float(value) if operation != "count" else int(value)
        records.append({"group": group_value, "value": numeric_value})
    return {"operation": f"grouped_{operation}", "group_column": str(group),
            "column": None if column is None else str(column), "groups": records}


def verify_ratio(numerator, denominator):
    """Deterministically calculate a ratio from two explicit numbers."""
    numerator, denominator = float(numerator), float(denominator)
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        raise ValueError("Ratio inputs must be finite and the denominator must be nonzero.")
    return {"operation": "ratio", "numerator": numerator,
            "denominator": denominator, "value": numerator / denominator}


stats = SimpleNamespace(compare_groups=compare_groups, correlate=correlate,
                        regress=regress, detect_outliers=detect_outliers,
                        grouped_mean=grouped_mean)
ts = SimpleNamespace(decompose=decompose, forecast_baseline=forecast_baseline)
viz = SimpleNamespace(distribution=distribution, class_balance=class_balance)
churn = SimpleNamespace(churn_rate=churn_rate,
                        retention_by_cohort=retention_by_cohort,
                        segment_lift=segment_lift,
                        class_balance=churn_class_balance,
                        rate_by_bins=rate_by_bins)


def make_verifier(records):
    """Create recording wrappers for deterministic calculations in one run."""
    def record(function):
        def wrapped(*args, **kwargs):
            result = function(*args, **kwargs)
            records.append(result)
            return result
        return wrapped
    return SimpleNamespace(row_count=record(verify_row_count),
                           aggregate=record(verify_aggregate),
                           grouped=record(verify_grouped),
                           duplicates=record(exact_duplicate_summary),
                           ratio=record(verify_ratio))
