import numpy as np
import pandas as pd

from src.agents.toolkit import (class_balance, compare_groups, correlate,
                                churn_rate, make_verifier, grouped_mean,
                                churn_class_balance,
                                detect_outliers, decompose, distribution,
                                forecast_baseline, regress, retention_by_cohort,
                                rate_by_bins, segment_lift, exact_duplicate_summary)


def test_compare_groups_returns_welch_effect_and_interval():
    frame = pd.DataFrame({"score": [1, 2, 3, 5, 6, 7], "group": ["a", "a", "a", "b", "b", "b"]})
    result = compare_groups(frame, "score", "group", "a", "b")
    assert result["n_1"] == result["n_2"] == 3
    assert result["ci_95"][0] < result["mean_difference"] < result["ci_95"][1]
    assert result["p_value"] < 0.1
    assert "Welch" in result["assumptions"]


def test_correlation_regression_and_outlier_helpers():
    frame = pd.DataFrame({"x": [1., 2., 3., 4., 5., 6.], "y": [3., 5., 7., 9., 11., 13.]})
    assert correlate(frame, ["x", "y"])["correlation"]["x"]["y"] == 1.0
    model = regress(frame, "y", ["x"])
    assert np.isclose(model["coefficients"]["x"], 2.0)
    assert detect_outliers(pd.Series([1, 1, 1, 1, 100]))["count"] == 1


def test_time_series_helpers_state_their_limits():
    decomposition = decompose(pd.Series(np.arange(24)), 12)
    forecast = forecast_baseline(pd.Series([1, 2, 3, 4]), 3)
    assert len(decomposition["trend"]) == 24
    assert forecast["forecast"] == [4., 4., 4.]
    assert "baseline" in forecast["method"].lower()


def test_viz_helpers_return_plotly_figures():
    frame = pd.DataFrame({"class": ["a", "a", "b"], "value": [1, 2, 3]})
    assert len(class_balance(frame, "class").data) == 2
    assert len(distribution(frame, "value").data) == 2


def test_churn_helpers_honor_explicit_positive_class_and_report_denominators():
    frame = pd.DataFrame({"status": ["Active", "Closed", None, "Active"],
                          "cohort": ["Q1", "Q1", "Q2", "Q2"],
                          "plan": ["monthly", "annual", "monthly", "annual"]})
    overall = churn_rate(frame, "status", positive="Closed")
    assert overall["churned"] == 1
    assert overall["n"] == 3
    assert overall["excluded_missing"] == 1
    by_cohort = retention_by_cohort(frame, "cohort", "status", positive="Closed")
    assert [group["n"] for group in by_cohort["cohorts"]] == [2, 1]
    by_plan = segment_lift(frame, "plan", "status", positive="Closed")
    assert sum(group["n"] for group in by_plan["segments"]) == 3
    assert by_plan["plan"] == by_plan["segments"]
    assert "not causal" in by_plan["definition"]


def test_deterministic_verifier_supports_simple_and_grouped_calculations():
    frame = pd.DataFrame({"group": ["a", "a", "b"], "value": [1, None, 5]})
    checks = []
    verify = make_verifier(checks)
    assert verify.row_count(frame)["value"] == 3
    assert verify.aggregate(frame, "value", "mean")["value"] == 3.0
    grouped = verify.grouped(frame, "group", "value", operation="sum")
    assert grouped["groups"] == [{"group": "a", "value": 1.0}, {"group": "b", "value": 5.0}]
    assert verify.ratio(1, 4)["value"] == 0.25
    assert len(checks) == 4


def test_churn_class_balance_reports_counts_percentages_and_missingness():
    frame = pd.DataFrame({"churned": [True, False, True, None]})
    result = churn_class_balance(frame)
    assert result["classes"] == [
        {"class": "Churned", "count": 2, "percent": 66.67},
        {"class": "Active", "count": 1, "percent": 33.33},
    ]
    assert result["excluded_missing"] == 1


def test_grouped_mean_excludes_missing_values_without_mutating_input():
    frame = pd.DataFrame({"group": ["a", "a", "b"], "fee": [10.0, None, 30.0]})
    result = grouped_mean(frame, "fee", "group")
    assert result["groups"] == [
        {"group": "a", "rows": 2, "n": 1, "excluded_missing": 1, "mean": 10.0},
        {"group": "b", "rows": 1, "n": 1, "excluded_missing": 0, "mean": 30.0},
    ]
    assert frame["fee"].isna().sum() == 1


def test_rate_by_bins_uses_explicit_left_closed_boundaries():
    frame = pd.DataFrame({"tenure": [6, 1, 3, 2], "churned": [False, True, True, False]})
    result = rate_by_bins(frame, "tenure", [-np.inf, 3, 6, np.inf],
                          ["<3", "3-<6", "6+"])
    assert result["bands"] == [
        {"band": "<3", "n": 2, "churned": 1, "churn_rate": 0.5},
        {"band": "3-<6", "n": 1, "churned": 1, "churn_rate": 1.0},
        {"band": "6+", "n": 1, "churned": 0, "churn_rate": 0.0},
    ]


def test_exact_duplicate_summary_distinguishes_extra_rows_from_involved_rows():
    frame = pd.DataFrame({"id": [1, 1, 2], "value": ["a", "a", "b"]})
    result = exact_duplicate_summary(frame)
    assert result["duplicate_rows"] == 1
    assert result["rows_in_duplicate_groups"] == 2
