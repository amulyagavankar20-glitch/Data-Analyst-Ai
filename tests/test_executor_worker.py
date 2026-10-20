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


def test_worker_records_execution_evidence_and_deterministic_calculations():
    frame = pd.DataFrame({"group": ["a", "a", "b"], "amount": [10.0, None, 30.0]})
    response = _run_worker(
        frame,
        "result = verify.grouped(df, 'group', 'amount', operation='mean')\n"
        "print(result)\nverify.row_count(df)",
    )
    assert response["ok"] is True
    assert response["evidence"]["input_shape"] == [3, 2]
    assert response["evidence"]["output_shape"] == [3, 2]
    assert response["evidence"]["code_referenced_columns"] == ["group", "amount"]
    assert len(response["evidence"]["code_sha256"]) == 64
    assert response["verification"][0]["operation"] == "grouped_mean"
    assert response["verification"][0]["groups"] == [
        {"group": "a", "value": 10.0}, {"group": "b", "value": 30.0}
    ]
    assert response["verification"][1] == {"operation": "row_count", "value": 3, "rows": 3}


def test_worker_labels_blank_plotly_axes_from_hover_fields_and_records_duplicates():
    frame = pd.DataFrame({"status": ["Active", "Churned", "Active"],
                          "count": [2, 1, 2]})
    response = _run_worker(
        frame,
        "result = verify.duplicates(df)\n"
        "print(result)\n"
        "fig = px.bar(df, x='status', y='count', title='Class balance')\n"
        "fig.update_layout(xaxis_title='')",
    )
    assert response["ok"] is True
    assert response["verification"][0]["duplicate_rows"] == 1
    figure = json.loads(response["charts"][0])
    assert figure["layout"]["xaxis"]["title"]["text"] == "status"
    assert figure["layout"]["yaxis"]["title"]["text"] == "count"
