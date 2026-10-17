"""Worker protocol checks using benign snippets (isolation itself is tested with Docker)."""
import base64
import io
import json
import subprocess
import sys

import pandas as pd


def _run_worker(frame, code):
    source = io.BytesIO()
    frame.to_parquet(source, index=None, compression="zstd")
    payload = json.dumps({"code": code, "data": base64.b64encode(source.getvalue()).decode()})
    result = subprocess.run([sys.executable, "-m", "src.agents.executor_worker"],
                            input=payload, capture_output=True, text=True,
                            timeout=15, check=True)
    return json.loads(result.stdout)


def test_worker_preserves_parquet_types_and_supports_lambda_and_charts():
    frame = pd.DataFrame({
        "postal": pd.Series(["00501", "00710"], dtype="string"),
        "when": pd.date_range("2025-01-01", periods=2, tz="UTC"),
        "count": pd.Series([1, None], dtype="Int64"),
        "category": pd.Series(["a", "b"], dtype="category"),
        "big": [2**60 + 1, 2**60 + 2],
    })
    response = _run_worker(
        frame,
        "result = df.apply(lambda row: pd.isna(row).sum(), axis=1).tolist()\n"
        "print(result)\nfig = px.bar(df, x='postal', y='big', title='Postal values')",
    )
    assert response["ok"] is True
    assert "[0, 1]" in response["output"]
    assert len(response["charts"]) == 1
    restored = pd.read_parquet(io.BytesIO(base64.b64decode(response["dataframe"])))
    pd.testing.assert_frame_equal(restored, frame)


def test_worker_catches_base_exception_as_a_safe_code_error():
    frame = pd.DataFrame({"x": [1]})
    response = _run_worker(frame, "raise SystemExit(2)")
    assert response["ok"] is False
    assert response["category"] == "code"
    assert "failed during execution" in response["error"]


def test_worker_stops_output_flood_at_limit():
    response = _run_worker(pd.DataFrame({"x": [1]}), "print('x' * 2_000_000)")
    assert response["ok"] is False
    assert response["category"] == "limit"


def test_worker_exposes_vetted_statistics_and_visualization_helpers():
    frame = pd.DataFrame({"score": [1, 2, 5, 6], "group": ["a", "a", "b", "b"]})
    response = _run_worker(
        frame,
        "result = stats.compare_groups(df, 'score', 'group', 'a', 'b')\n"
        "print(result['p_value'])\nfig = viz.class_balance(df, 'group')",
    )
    assert response["ok"] is True
    assert "p_value" in response["output"]
    assert len(response["charts"]) == 1
