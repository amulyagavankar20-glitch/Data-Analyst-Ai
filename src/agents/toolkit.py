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


stats = SimpleNamespace(compare_groups=compare_groups, correlate=correlate,
                        regress=regress, detect_outliers=detect_outliers)
ts = SimpleNamespace(decompose=decompose, forecast_baseline=forecast_baseline)
viz = SimpleNamespace(distribution=distribution, class_balance=class_balance)
