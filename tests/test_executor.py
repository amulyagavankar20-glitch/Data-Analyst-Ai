from executor_service.server import docker_command
from src.agents.executor import InfraError, run_isolated_code
from urllib.error import URLError
import pandas as pd
import pytest


def test_worker_is_started_with_security_and_resource_limits():
    command = docker_command("executor:test")
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--memory=512m" in command
    assert "--cpus=1" in command
    assert "--ulimit=cpu=20:20" in command
    assert "--pids-limit=64" in command
    assert "--user=65534:65534" in command
    assert "--runtime=runsc" in command
    assert not any("volume" in part or part == "-v" for part in command)


def test_worker_has_no_host_mounts_or_network():
    command = docker_command("executor:test")
    assert command.count("--network=none") == 1
    assert any(part.startswith("--tmpfs=") for part in command)
    assert not any(part.startswith("--mount") or part.startswith("-v") for part in command)


def test_executor_client_fails_closed_without_service_configuration(monkeypatch):
    monkeypatch.delenv("ANALYSIS_EXECUTOR_URL", raising=False)
    monkeypatch.delenv("ANALYSIS_EXECUTOR_TOKEN", raising=False)
    with pytest.raises(InfraError):
        run_isolated_code("result = 1", pd.DataFrame({"x": [1]}))


def test_executor_client_rejects_remote_plain_http(monkeypatch):
    monkeypatch.setenv("ANALYSIS_EXECUTOR_URL", "http://executor.internal")
    monkeypatch.setenv("ANALYSIS_EXECUTOR_TOKEN", "a" * 32)
    with pytest.raises(InfraError, match="HTTPS"):
        run_isolated_code("result = 1", pd.DataFrame({"x": [1]}))


def test_executor_client_gives_actionable_connection_error(monkeypatch):
    monkeypatch.setenv("ANALYSIS_EXECUTOR_URL", "https://executor.example.invalid")
    monkeypatch.setenv("ANALYSIS_EXECUTOR_TOKEN", "a" * 32)
    monkeypatch.setattr("src.agents.executor.urlopen",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("unavailable")))
    with pytest.raises(InfraError, match="ANALYSIS_EXECUTOR_URL.*health"):
        run_isolated_code("result = 1", pd.DataFrame({"x": [1]}))
