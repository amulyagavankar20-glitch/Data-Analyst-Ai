"""Container entry point. Input and output are JSON over standard streams."""
import contextlib
import io
import base64
import json
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.basedatatypes import BaseFigure
from plotly.subplots import make_subplots
from src.agents.toolkit import stats, ts, viz

MAX_CODE_OUTPUT = 1024 * 1024
MAX_RESPONSE = 48 * 1024 * 1024
MAX_CHARTS = 4
MAX_CHART_BYTES = 4 * 1024 * 1024


def safe_error(exc, dataframe=None):
    """Summarize an execution problem without returning data values or paths."""
    summaries = {
        "SyntaxError": "Generated code has invalid syntax.",
        "NameError": "Generated code referenced an undefined name.",
        "KeyError": "Generated code referenced a missing column or key.",
        "TypeError": "Generated code used incompatible value types.",
        "ValueError": "Generated code could not convert or validate a value.",
        "ZeroDivisionError": "Generated code attempted division by zero.",
        "IndexError": "Generated code referenced a missing row or position.",
        "TimeoutError": "Generated code exceeded an operation time limit.",
        "MemoryError": "Generated code exceeded available memory.",
    }
    if isinstance(exc, KeyError) and dataframe is not None:
        missing = str(exc.args[0]) if exc.args else ""
        if len(missing) <= 120 and missing in {str(column) for column in dataframe.columns}:
            return f"Generated code referenced the missing column {missing!r}."
    if isinstance(exc, SyntaxError) and getattr(exc, "lineno", None):
        return f"Generated code has invalid syntax near line {exc.lineno}."
    return summaries.get(type(exc).__name__, "Generated code failed during execution.")


class LimitedOutput(io.StringIO):
    def write(self, value):
        current = self.tell()
        if current + len(value) > MAX_CODE_OUTPUT:
            raise OverflowError("Output exceeded worker limit")
        super().write(value)
        return len(value)


def main():
    try:
        request = json.loads(sys.stdin.buffer.read(32 * 1024 * 1024 + 1))
        if not isinstance(request, dict) or len(request.get("code", "")) > 100_000:
            raise ValueError("Invalid request")
        dataset = base64.b64decode(request["data"], validate=True)
        if len(dataset) > 16 * 1024 * 1024:
            raise OverflowError("Dataset exceeds worker limit")
        dataframe = pd.read_parquet(io.BytesIO(dataset))
        output = LimitedOutput()
        local_scope = {"df": dataframe, "pd": pd, "np": np, "px": px,
                       "go": go, "make_subplots": make_subplots,
                       "stats": stats, "ts": ts, "viz": viz}
        with contextlib.redirect_stdout(output):
            exec(request["code"], local_scope, local_scope)
        if "result" in local_scope:
            output.write("\nResult: " + str(local_scope["result"])[:10_000])
        charts: list[str] = []
        seen_charts = set()
        for value in local_scope.values():
            if not isinstance(value, BaseFigure) or id(value) in seen_charts:
                continue
            if len(charts) >= MAX_CHARTS:
                break
            chart_json = value.to_json(validate=True)
            if len(chart_json.encode("utf-8")) > MAX_CHART_BYTES:
                raise OverflowError("Chart output exceeded limit")
            charts.append(chart_json)
            seen_charts.add(id(value))
        frame_buffer = io.BytesIO()
        local_scope["df"].to_parquet(frame_buffer, index=None, compression="zstd")
        if frame_buffer.tell() > 16 * 1024 * 1024:
            raise OverflowError("Dataset output exceeds worker limit")
        response = {"ok": True, "output": output.getvalue(), "charts": charts,
                    "dataframe": base64.b64encode(frame_buffer.getvalue()).decode("ascii")}
    except BaseException as exc:
        limited = isinstance(exc, (MemoryError, OverflowError, TimeoutError))
        response = {"ok": False, "category": "limit" if limited else "code",
                    "error": "Analysis exceeded a resource limit." if limited else safe_error(exc, locals().get("dataframe"))}
    encoded = json.dumps(response).encode("utf-8")
    if len(encoded) > MAX_RESPONSE:
        encoded = json.dumps({"ok": False, "category": "limit",
                              "error": "Analysis output exceeded its response limit."}).encode("utf-8")
    sys.stdout.buffer.write(encoded)


if __name__ == "__main__":
    main()
