"""Small token-authenticated HTTP bridge for a dedicated executor host.

Deploy only on a dedicated rootless Docker/gVisor host behind TLS. Never expose
this service directly to the public internet.
"""
import hmac
import json
import os
import subprocess
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_REQUEST = 22 * 1024 * 1024
MAX_RESPONSE = 48 * 1024 * 1024
LOCK = threading.BoundedSemaphore(int(os.getenv("EXECUTOR_CONCURRENCY", "2")))
TOKEN = os.environ.get("ANALYSIS_EXECUTOR_TOKEN", "")
IMAGE = os.getenv("ANALYSIS_EXECUTOR_IMAGE", "ai-data-analyst-executor:latest")
RUNTIME = os.getenv("EXECUTOR_RUNTIME", "runsc")


def docker_command(image=IMAGE, container_name=None):
    """Build the complete worker invocation from server-owned constants only."""
    return ["docker", "run", "--rm", "-i", "--name", container_name or "analysis-test",
            "--network=none", "--read-only",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--memory=512m",
            "--memory-swap=512m", "--cpus=1", "--pids-limit=64", "--user=65534:65534",
            "--ulimit=cpu=20:20", "--tmpfs=/tmp:rw,noexec,nosuid,size=64m",
            "--runtime=" + RUNTIME, image]


class Handler(BaseHTTPRequestHandler):
    server_version = "AnalysisExecutor"

    def respond(self, status, payload):
        data = json.dumps(payload, separators=(",", ":")).encode()
        self.respond_raw(status, data)

    def respond_raw(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            return self.respond(200, {"status": "ok"})
        return self.respond(404, {"error": "Not found"})

    def do_POST(self):
        self.connection.settimeout(5)
        if self.path != "/v1/execute":
            return self.respond(404, {"error": "Not found"})
        supplied = self.headers.get("Authorization", "")
        if not TOKEN or not hmac.compare_digest(supplied, "Bearer " + TOKEN):
            return self.respond(401, {"error": "Unauthorized"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            size = MAX_REQUEST + 1
        if size <= 0 or size > MAX_REQUEST:
            return self.respond(413, {"error": "Request size limit exceeded"})
        if not LOCK.acquire(timeout=2):
            return self.respond(429, {"error": "Executor is busy"})
        try:
            raw = self.rfile.read(size)
            request = json.loads(raw)
            if not isinstance(request, dict) or not isinstance(request.get("code"), str):
                return self.respond(400, {"error": "Invalid request"})
            if len(request["code"]) > 100_000 or len(raw) > MAX_REQUEST:
                return self.respond(413, {"error": "Request size limit exceeded"})
            container_name = "analysis-" + uuid.uuid4().hex
            command = docker_command(container_name=container_name)
            proc = subprocess.run(command, input=raw, capture_output=True, timeout=25, check=False)
            if proc.returncode != 0 or len(proc.stdout) > MAX_RESPONSE:
                _remove_failed_container(container_name)
                if proc.returncode in {124, 137}:
                    return self.respond(200, {"ok": False, "category": "limit",
                                              "error": "Analysis exceeded its CPU or memory limit."})
                return self.respond(503, {"category": "infrastructure", "error": "Worker failed."})
            response = json.loads(proc.stdout)
            if not isinstance(response, dict):
                return self.respond(503, {"category": "infrastructure", "error": "Worker returned invalid output."})
            return self.respond_raw(200, proc.stdout)
        except subprocess.TimeoutExpired:
            if "container_name" in locals():
                _remove_failed_container(container_name)
            return self.respond(200, {"ok": False, "category": "limit", "error": "Analysis exceeded its execution time limit."})
        except BaseException:
            return self.respond(503, {"category": "infrastructure", "error": "Executor is temporarily unavailable."})
        finally:
            LOCK.release()

    def log_message(self, _format, *_args):
        return


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    request_queue_size = 16
    daemon_threads = True

    def __init__(self, *args, **kwargs):
        self.request_slots = threading.BoundedSemaphore(8)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        if not self.request_slots.acquire(timeout=2):
            try:
                request.sendall(b"HTTP/1.1 429 Too Many Requests\r\nConnection: close\r\nContent-Length: 0\r\n\r\n")
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.request_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.request_slots.release()


def _remove_failed_container(container_name):
    try:
        subprocess.run(["docker", "rm", "--force", container_name],
                       capture_output=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return


if __name__ == "__main__":
    if len(TOKEN) < 32:
        raise SystemExit("ANALYSIS_EXECUTOR_TOKEN must contain at least 32 characters")
    try:
        info = subprocess.run(["docker", "info", "--format", "{{json .Runtimes}}|{{json .SecurityOptions}}"],
                              capture_output=True, text=True, timeout=5, check=True).stdout
        runtime_json, security_json = info.split("|", 1)
        runtimes = json.loads(runtime_json)
        security = json.loads(security_json)
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit("Cannot verify executor Docker isolation configuration") from exc
    if RUNTIME not in runtimes:
        raise SystemExit(f"Required Docker runtime {RUNTIME!r} is not installed")
    rootless = any("rootless" in str(option).lower() for option in security)
    ci_runtime_override = os.getenv("EXECUTOR_CI_ALLOW_RUNC") == "1"
    if RUNTIME != "runsc" and not rootless and not ci_runtime_override:
        raise SystemExit("Production executor requires gVisor runsc or rootless Docker")
    BoundedThreadingHTTPServer((os.getenv("EXECUTOR_BIND", "127.0.0.1"),
                                int(os.getenv("EXECUTOR_PORT", "8765"))), Handler).serve_forever()
