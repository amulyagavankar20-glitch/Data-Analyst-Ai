import numpy as np
import pandas as pd

from src.agents.toolkit import (class_balance, compare_groups, correlate,
                                detect_outliers, decompose, distribution,
                                forecast_baseline, regress)


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
