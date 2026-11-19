from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from examples.swe_smith.hierarchical.docker_runner import (
    RUN_LABEL,
    DockerRunner,
    _attempt_name,
    remove_run_containers,
)


def test_attempt_name_parses_rollout_and_attempt():
    url = "http://127.0.0.1:8080/api/rollouts/rollout-123/attempt/att-456/events"
    assert _attempt_name(url) == "agl-rollout-rollout-123-att-456"

    with pytest.raises(ValueError, match="AGL_EVENT_URL must address a rollout attempt"):
        _attempt_name("http://127.0.0.1:8080/api/invalid")


def test_docker_runner_missing_required_env():
    runner = DockerRunner()
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(KeyError, match="IMAGE_NAME"):
            runner.run()


def test_docker_runner_lifecycle_create_and_attach(tmp_path):
    run_dir = tmp_path / "exp-1"
    run_dir.mkdir()
    env = {
        "IMAGE_NAME": "jyangballin/swesmith.sympy__1.1",
        "AGL_RUN_DIR": str(run_dir),
        "AGL_EVENT_URL": "http://127.0.0.1:8080/api/rollouts/r1/attempt/a1/events",
        "AGL_TASK_INPUT": "Issue statement",
        "ARBAS_AGENT_CONFIG": '{"orchestrator":{"max_turns":32}}',
        "AGL_OPENAI_BASE_URL": "http://127.0.0.1:8080/proxy/rollout/r1/attempt/a1/mode/train/openai/v1",
        "AGL_KEY": "shared-key",
        "OPENROUTER_API_KEY": "sk-or-test",
    }

    runner = DockerRunner()
    mock_cleanup = MagicMock()
    mock_cleanup.__enter__.return_value = mock_cleanup
    mock_cleanup.__exit__.return_value = None
    mock_cleanup.stdin = MagicMock()
    mock_cleanup.wait.return_value = 0

    with (
        patch.dict(os.environ, env, clear=True),
        patch("subprocess.run") as mock_run,
        patch("subprocess.Popen", return_value=mock_cleanup) as mock_popen,
    ):
        mock_run.return_value = MagicMock(returncode=0)

        runner.run()

        assert mock_run.call_count == 2
        # Call 1: docker create
        create_cmd = mock_run.call_args_list[0][0][0]
        assert create_cmd[0] == "docker"
        assert create_cmd[1] == "create"
        assert "--name" in create_cmd
        name_idx = create_cmd.index("--name")
        assert create_cmd[name_idx + 1] == "agl-rollout-r1-a1"

        assert "--network" in create_cmd
        net_idx = create_cmd.index("--network")
        assert create_cmd[net_idx + 1] == "host"

        assert "--label" in create_cmd
        lbl_idx = create_cmd.index("--label")
        assert create_cmd[lbl_idx + 1] == f"{RUN_LABEL}={run_dir.resolve()}"

        # Check mount points
        mount_args = [arg for arg in create_cmd if arg.startswith("type=bind")]
        assert any("dst=/agl/hierarchical" in m and "readonly" in m for m in mount_args)
        assert any("dst=/agl-logs/exp-1/agl-rollout-r1-a1" in m for m in mount_args)
        assert any("dst=/agl-logs/exp-1/oracle" in m for m in mount_args)

        # Check secrets/vars are passed via --env NAME without values in argv
        assert "--env" in create_cmd
        env_indices = [i for i, arg in enumerate(create_cmd) if arg == "--env"]
        passed_env_names = [create_cmd[i + 1] for i in env_indices]
        assert "OPENROUTER_API_KEY" in passed_env_names
        assert "AGL_TASK_INPUT" in passed_env_names
        assert "AGL_OPENAI_BASE_URL" in passed_env_names
        assert not any("sk-or-test" in arg for arg in create_cmd)

        assert "jyangballin/swesmith.sympy__1.1:arbas" in create_cmd

        # Check cleanup process
        mock_popen.assert_called_once()
        popen_args = mock_popen.call_args[0][0]
        assert popen_args[0] == "sh"
        assert "exec docker rm --force" in popen_args[2]
        assert popen_args[4] == "agl-rollout-r1-a1"
        assert mock_popen.call_args[1].get("start_new_session") is True

        # Call 2: docker start --attach
        start_cmd = mock_run.call_args_list[1][0][0]
        assert start_cmd == ["docker", "start", "--attach", "agl-rollout-r1-a1"]

        mock_cleanup.stdin.close.assert_called_once()
        mock_cleanup.wait.assert_called_once()


def test_docker_runner_start_error_cleans_up_and_propagates(tmp_path):
    run_dir = tmp_path / "exp-1"
    run_dir.mkdir()
    env = {
        "IMAGE_NAME": "jyangballin/swesmith.sympy__1.1",
        "AGL_RUN_DIR": str(run_dir),
        "AGL_EVENT_URL": "http://127.0.0.1:8080/api/rollouts/r1/attempt/a1/events",
    }

    runner = DockerRunner()
    mock_cleanup = MagicMock()
    mock_cleanup.__enter__.return_value = mock_cleanup
    mock_cleanup.__exit__.return_value = None
    mock_cleanup.stdin = MagicMock()
    mock_cleanup.wait.return_value = 0

    with (
        patch.dict(os.environ, env, clear=True),
        patch("subprocess.run") as mock_run,
        patch("subprocess.Popen", return_value=mock_cleanup),
    ):
        # First call (create) succeeds, second call (start) raises CalledProcessError
        mock_run.side_effect = [
            MagicMock(returncode=0),
            subprocess.CalledProcessError(1, ["docker", "start"]),
        ]

        with pytest.raises(subprocess.CalledProcessError):
            runner.run()

        mock_cleanup.stdin.close.assert_called_once()
        mock_cleanup.wait.assert_called_once()


def test_docker_runner_cleanup_failure_raises_runtime_error(tmp_path):
    run_dir = tmp_path / "exp-1"
    run_dir.mkdir()
    env = {
        "IMAGE_NAME": "jyangballin/swesmith.sympy__1.1",
        "AGL_RUN_DIR": str(run_dir),
        "AGL_EVENT_URL": "http://127.0.0.1:8080/api/rollouts/r1/attempt/a1/events",
    }

    runner = DockerRunner()
    mock_cleanup = MagicMock()
    mock_cleanup.__enter__.return_value = mock_cleanup
    mock_cleanup.__exit__.return_value = None
    mock_cleanup.stdin = MagicMock()
    mock_cleanup.wait.return_value = 1  # Exit code 1 on cleanup

    with (
        patch.dict(os.environ, env, clear=True),
        patch("subprocess.run", return_value=MagicMock(returncode=0)),
        patch("subprocess.Popen", return_value=mock_cleanup),
    ):
        with pytest.raises(RuntimeError, match="Failed to remove Docker container agl-rollout-r1-a1: exit 1"):
            runner.run()


def test_remove_run_containers(tmp_path):
    run_dir = tmp_path / "test-run"
    run_dir.mkdir()

    with patch("subprocess.run") as mock_run:
        # ps returns two containers
        mock_run.side_effect = [
            MagicMock(stdout="container1 container2\n"),
            MagicMock(returncode=0),
        ]

        remove_run_containers(run_dir)

        assert mock_run.call_count == 2
        ps_cmd = mock_run.call_args_list[0][0][0]
        assert ps_cmd == ["docker", "ps", "-aq", "--filter", f"label={RUN_LABEL}={run_dir.resolve()}"]

        rm_cmd = mock_run.call_args_list[1][0][0]
        assert rm_cmd == ["docker", "rm", "--force", "container1", "container2"]
