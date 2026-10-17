"""Docker boundary checks; skipped unless the executor image is available locally."""
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from src.agents.executor import CodeError, LimitError, run_isolated_code


def _executor_available():
    if not shutil.which("docker") or not os.getenv("ANALYSIS_EXECUTOR_URL"):
        return False
    try:
        return subprocess.run(
            ["docker", "image", "inspect", "ai-data-analyst-executor:latest"],
            capture_output=True, timeout=5, check=False,
        ).returncode == 0 and subprocess.run(
            ["docker", "info"], capture_output=True, timeout=5, check=False,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(not _executor_available(), reason="Docker executor image/daemon unavailable")
if os.getenv("CI") and not _executor_available():
    raise RuntimeError("CI requires Docker and ai-data-analyst-executor:latest")
if os.getenv("CI") and not os.getenv("ANALYSIS_EXECUTOR_URL"):
    raise RuntimeError("CI requires the authenticated executor service")


def test_dataframe_reassignment_and_multiple_plotly_charts_are_returned():
    frame = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    result = run_isolated_code(
        "df = df.drop(index=0)\n"
        "fig = px.line(df, x='x', y='y', title='Line chart')\n"
        "fig2 = px.bar(df, x='x', y='y', title='Bar chart')\n"
        "result = len(df)\nprint(result)", frame,
    )
    assert len(result["dataframe"]) == 2
    assert len(result["charts"]) == 2
    assert result["charts"][0].layout.title.text == "Line chart"
    assert result["charts"][1].layout.title.text == "Bar chart"
    assert "2" in result["output"]


def test_worker_cannot_read_application_files():
    frame = pd.DataFrame({"x": [1]})
    assert Path("app.py").is_file()  # Positive control: this file exists on the app host.
    with pytest.raises(CodeError):
        run_isolated_code("result = pd.read_csv('/app/app.py')", frame)


def test_worker_has_no_network_access():
    frame = pd.DataFrame({"x": [1]})
    with pytest.raises(CodeError):
        run_isolated_code("result = pd.read_csv('https://example.com')", frame)


@pytest.mark.slow
def test_worker_terminates_nonterminating_code():
    frame = pd.DataFrame({"x": [1]})
    with pytest.raises(LimitError):
        run_isolated_code("while True:\n    pass", frame)
    remaining = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "name=analysis-"],
        capture_output=True, text=True, timeout=5, check=True,
    ).stdout.strip()
    assert not remaining
