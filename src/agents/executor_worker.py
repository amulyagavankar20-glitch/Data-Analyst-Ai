"""Container entry point. Input and output are JSON over standard streams."""
import contextlib
import io
import json
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.basedatatypes import BaseFigure
from plotly.subplots import make_subplots

MAX_CODE_OUTPUT = 1024 * 1024
MAX_RESPONSE = 46 * 1024 * 1024
MAX_CHARTS = 4
MAX_CHART_BYTES = 4 * 1024 * 1024


def safe_error(exc):
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
    return summaries.get(type(exc).__name__, "Generated code failed during execution.")


class LimitedOutput(io.StringIO):
    def write(self, value):
        current = self.tell()
        remaining = MAX_CODE_OUTPUT - current
        if remaining <= 0:
            return len(value)
        super().write(value[:remaining])
        if len(value) > remaining:
            super().write("\n[Output truncated]\n")
        return len(value)


def main():
    try:
        request = json.loads(sys.stdin.buffer.read(20 * 1024 * 1024 + 1))
        if not isinstance(request, dict) or len(request.get("code", "")) > 100_000:
            raise ValueError("Invalid request")
        dataframe = pd.read_json(request["dataframe"], orient="split")
        output = LimitedOutput()
        local_scope = {"df": dataframe, "pd": pd, "np": np, "px": px,
                       "go": go, "make_subplots": make_subplots}
        with contextlib.redirect_stdout(output):
            exec(request["code"], {"__builtins__": __builtins__}, local_scope)
        if "result" in local_scope:
            output.write("\nResult: " + str(local_scope["result"])[:10_000])
        charts = []
        seen_charts = set()
        for value in local_scope.values():
            if not isinstance(value, BaseFigure) or id(value) in seen_charts:
                continue
            if len(charts) >= MAX_CHARTS:
                break
            chart_json = value.to_json(validate=True)
            if len(chart_json.encode("utf-8")) > MAX_CHART_BYTES:
                raise ValueError("Chart output exceeded limit")
            charts.append(chart_json)
            seen_charts.add(id(value))
        response = {"ok": True, "output": output.getvalue(), "charts": charts,
                    "dataframe": local_scope["df"].to_json(orient="split", date_format="iso")}
    except Exception as exc:
        response = {"ok": False, "error": safe_error(exc)}
    encoded = json.dumps(response).encode("utf-8")
    if len(encoded) > MAX_RESPONSE:
        encoded = json.dumps({"ok": False, "error": "OutputLimit"}).encode("utf-8")
    sys.stdout.buffer.write(encoded)


if __name__ == "__main__":
    main()
