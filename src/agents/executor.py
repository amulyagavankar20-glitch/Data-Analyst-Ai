"""Docker-backed, resource-limited execution boundary for generated code."""
from io import StringIO
import json
import os
import subprocess
import threading
import uuid

import pandas as pd
import plotly.io as pio

MAX_INPUT_BYTES = 20 * 1024 * 1024
MAX_OUTPUT_BYTES = 48 * 1024 * 1024
EXECUTION_TIMEOUT_SECONDS = 25


class ExecutionError(RuntimeError):
    """An isolated analysis could not be completed."""


def _stop_container(name):
    """Remove an orphaned container if the Docker client was stopped or timed out."""
    try:
        subprocess.run(["docker", "rm", "--force", name], capture_output=True,
                       timeout=5, check=False, shell=False)
    except (OSError, subprocess.TimeoutExpired):
        pass


def docker_command(name, image):
    """Return the fully constrained worker launch command."""
    return [
        "docker", "run", "--rm", "-i", "--name", name, "--network=none", "--read-only",
        "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=64",
        "--memory=512m", "--memory-swap=512m", "--cpus=1", "--ulimit=cpu=20:20",
        "--ulimit=nofile=64:64", "--ulimit=fsize=10485760:10485760", "--user=65534:65534",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m", image,
    ]


def run_isolated_code(code, dataframe):
    """Run generated code in a disposable container with no network or host mounts."""
    image = os.getenv("ANALYSIS_EXECUTOR_IMAGE", "ai-data-analyst-executor:latest")
    try:
        frame = dataframe.to_json(orient="split", date_format="iso")
        request = json.dumps({"code": code, "dataframe": frame}).encode("utf-8")
    except Exception as exc:
        raise ExecutionError("The dataset could not be prepared for isolated analysis.") from exc
    if len(request) > MAX_INPUT_BYTES:
        raise ExecutionError("The dataset is too large for the isolated analysis worker.")

    name = "analysis-" + uuid.uuid4().hex
    command = docker_command(name, image)
    process = None
    stdout = bytearray()
    stderr = bytearray()
    overflow = threading.Event()

    def drain(stream, target, limit):
        while True:
            chunk = stream.read(8192)
            if not chunk:
                return
            if len(target) + len(chunk) > limit:
                overflow.set()
                return
            target.extend(chunk)

    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, shell=False)
        out_thread = threading.Thread(target=drain, args=(process.stdout, stdout, MAX_OUTPUT_BYTES), daemon=True)
        err_thread = threading.Thread(target=drain, args=(process.stderr, stderr, 64 * 1024), daemon=True)
        out_thread.start()
        err_thread.start()
        try:
            process.stdin.write(request)
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            return_code = process.wait(timeout=EXECUTION_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            _stop_container(name)
            raise ExecutionError("Analysis exceeded its time limit and was stopped.") from exc
        if overflow.is_set():
            _stop_container(name)
            raise ExecutionError("Analysis output exceeded the allowed limit.")
        out_thread.join(timeout=2)
        err_thread.join(timeout=2)
    except ExecutionError:
        raise
    except FileNotFoundError as exc:
        raise ExecutionError("The isolated analysis service is unavailable.") from exc
    except OSError as exc:
        raise ExecutionError("The isolated analysis service could not be started.") from exc
    finally:
        if process and process.poll() is None:
            process.kill()
        _stop_container(name)

    if return_code != 0:
        raise ExecutionError("The isolated analysis worker failed. Check its deployment.")
    try:
        response = json.loads(stdout.decode("utf-8"))
        if not response.get("ok"):
            raise ExecutionError(response.get("error", "The generated analysis could not be executed."))
        result_df = pd.read_json(StringIO(response["dataframe"]), orient="split")
        charts = [pio.from_json(item, output_type="Figure") for item in response.get("charts", [])]
        return {"output": response.get("output", ""), "dataframe": result_df,
                "charts": charts}
    except ExecutionError:
        raise
    except Exception as exc:
        raise ExecutionError("The isolated analysis worker returned an invalid response.") from exc
