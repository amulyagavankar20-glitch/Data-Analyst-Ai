"""Client for a separately deployed, authenticated analysis executor service."""
import base64
from io import BytesIO
import json
import logging
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlparse

import pandas as pd
import plotly.io as pio

logger = logging.getLogger(__name__)


class ExecutorError(RuntimeError):
    category = "executor"


class CodeError(ExecutorError):
    category = "code"


class LimitError(ExecutorError):
    category = "limit"


class InfraError(ExecutorError):
    category = "infrastructure"


ExecutionError = CodeError  # Backwards-compatible name for user-code failures.


def run_isolated_code(code, dataframe, dataset_blob=None):
    """Send one bounded analysis request to an isolated executor host.

    The app process never starts Docker and never receives Docker socket access.
    """
    endpoint = os.getenv("ANALYSIS_EXECUTOR_URL", "").rstrip("/")
    token = os.getenv("ANALYSIS_EXECUTOR_TOKEN", "")
    if not endpoint or not token:
        raise InfraError("The isolated analysis service is not configured.")
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise InfraError("The isolated analysis service must use HTTPS.")
    try:
        if dataset_blob is None:
            buffer = BytesIO()
            dataframe.to_parquet(buffer, index=None, compression="zstd")
            dataset_blob = buffer.getvalue()
        payload = json.dumps({"code": code, "data": base64.b64encode(dataset_blob).decode("ascii")}).encode()
        request = Request(endpoint + "/v1/execute", data=payload, method="POST", headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        })
        timeout = min(max(float(os.getenv("ANALYSIS_EXECUTOR_TIMEOUT", "35")), 1), 60)
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise InfraError("The analysis service returned an unexpected response.")
            result = json.loads(response.read(48 * 1024 * 1024 + 1))
        if not result.get("ok"):
            category = result.get("category")
            error = result.get("error", "Analysis failed.")
            if category == "limit":
                raise LimitError(error)
            if category == "code":
                raise CodeError(error)
            raise InfraError(error)
        output = base64.b64decode(result["dataframe"], validate=True)
        updated_df = pd.read_parquet(BytesIO(output))
        charts = [pio.from_json(chart) for chart in result.get("charts", [])]
        return {"output": result.get("output", "")[:1_000_000], "charts": charts,
                "dataframe": updated_df, "dataframe_blob": output,
                "evidence": result.get("evidence", {}),
                "verification": result.get("verification", [])}
    except ExecutorError:
        raise
    except HTTPError as exc:
        logger.warning("Executor returned HTTP status %s", exc.code)
        raise InfraError("The analysis service rejected the request. Check its token and service logs.") from exc
    except (URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        logger.warning("Executor connection failed (%s)", type(reason).__name__)
        raise InfraError("Cannot reach the analysis service. Check ANALYSIS_EXECUTOR_URL and its /health endpoint.") from exc
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("Executor response was invalid (%s)", type(exc).__name__)
        raise InfraError("The analysis service returned an invalid response. Check the worker image and service logs.") from exc
