from src.agents.executor import docker_command


def test_worker_is_started_with_security_and_resource_limits():
    command = docker_command("analysis-test", "executor:test")
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--memory=512m" in command
    assert "--cpus=1" in command
    assert "--ulimit=cpu=20:20" in command
    assert "--pids-limit=64" in command
    assert "--user=65534:65534" in command
    assert not any("volume" in part or part == "-v" for part in command)


def test_worker_has_no_host_mounts_or_network():
    command = docker_command("analysis-test", "executor:test")
    assert command.count("--network=none") == 1
    assert "--tmpfs" in command
    assert not any(part.startswith("--mount") or part.startswith("-v") for part in command)
