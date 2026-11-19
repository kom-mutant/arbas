"""Run each SWE-smith rollout in its own Docker container."""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

from ..pull_images import openai_image_name

log = logging.getLogger(__name__)
RUN_LABEL = "arbas.run"
_AGENT_SCRIPT = """
set -euo pipefail
cd /agl
python -m hierarchical.run 2>&1 | tee "$HIER_OUTPUT_DIR/$AGL_RUN_NAME-$AGL_ROLLOUT_JOB_NAME.log"
"""


def _attempt_name(event_url: str) -> str:
    match = re.search(r"/rollouts/([a-zA-Z0-9_-]+)/attempt/([a-zA-Z0-9_-]+)/events$", urlsplit(event_url).path)
    if match is None:
        raise ValueError("AGL_EVENT_URL must address a rollout attempt")
    return f"agl-rollout-{match[1]}-{match[2]}"


def remove_run_containers(run_dir: Path) -> None:
    result = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"label={RUN_LABEL}={run_dir.resolve()}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if containers := result.stdout.split():
        subprocess.run(["docker", "rm", "--force", *containers], check=True, timeout=30)


class DockerRunner:
    def run(self) -> None:
        image = openai_image_name(os.environ["IMAGE_NAME"])
        run_dir = Path(os.environ["AGL_RUN_DIR"]).resolve()
        run_name = run_dir.name
        attempt_name = _attempt_name(os.environ["AGL_EVENT_URL"])
        host_logs = run_dir / "agl-logs"
        output_dir = host_logs / run_name / attempt_name
        oracle_dir = output_dir.parent / "oracle"
        output_dir.mkdir(parents=True, exist_ok=True)
        oracle_dir.mkdir(exist_ok=True)
        container_logs = f"/agl-logs/{run_name}"
        pkg_dir = Path(__file__).resolve().parent

        env = {
            **os.environ,
            "AGL_ROLLOUT_JOB_NAME": attempt_name,
            "AGL_RUN_NAME": run_name,
            "HIER_OUTPUT_DIR": f"{container_logs}/{attempt_name}",
        }
        network = env.get("AGL_DOCKER_NETWORK", "host")
        command = [
            "docker", "create", "--name", attempt_name, "--label", f"{RUN_LABEL}={run_dir}",
            "--network", network, "--pull=never", "--init",
            "--mount", f"type=bind,src={pkg_dir},dst=/agl/hierarchical,readonly",
            "--mount", f"type=bind,src={output_dir},dst={env['HIER_OUTPUT_DIR']}",
            "--mount", f"type=bind,src={oracle_dir},dst={container_logs}/oracle",
        ]
        for name in (
            "AGL_TASK_INPUT", "ARBAS_AGENT_CONFIG", "AGL_OPENAI_BASE_URL",
            "AGL_EVENT_URL", "AGL_KEY", "AGL_ROLLOUT_JOB_NAME", "AGL_RUN_NAME",
            "HIER_OUTPUT_DIR", "OPENROUTER_API_KEY", "POLZA_API_KEY", "SUBAGENT_API_KEY", "JUDGE_API_KEY",
        ):
            if name in env:
                command.extend(["--env", name])
        command.extend([image, "bash", "-lc", _AGENT_SCRIPT])

        subprocess.run(command, check=True, env=env, stdout=subprocess.DEVNULL)
        try:
            # LocalReconciler uses SIGKILL on the worker group. This process
            # survives it and receives EOF when the worker's pipe closes.
            with subprocess.Popen(
                ["sh", "-c", 'cat >/dev/null; exec docker rm --force "$1" >/dev/null', "cleanup", attempt_name],
                stdin=subprocess.PIPE,
                start_new_session=True,
            ) as cleanup:
                assert cleanup.stdin is not None
                try:
                    subprocess.run(["docker", "start", "--attach", attempt_name], check=True)
                finally:
                    cleanup.stdin.close()
                    status = cleanup.wait()
            if status:
                raise RuntimeError(f"Failed to remove Docker container {attempt_name}: exit {status}")
        except OSError:
            subprocess.run(["docker", "rm", "--force", attempt_name], check=False, timeout=15)
            raise
