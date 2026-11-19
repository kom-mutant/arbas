import json
import os
import re
import shlex
import socket
import subprocess
import sys
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from omegaconf import OmegaConf

from agentlightning.controller.k8s_reconciler import build_job_spec
from agentlightning.schemas import Rollout, RolloutConfig, RolloutK8sConfig, RolloutLifecycleStatus
from examples.swe_smith import launch
from examples.swe_smith.hierarchical.config import AgentSettings, load_config
from examples.swe_smith.train_smith_agent import build_agent_config, load_experiment_config, load_split_file

CONFIG = Path(__file__).resolve().parents[2] / "examples/swe_smith/training.yaml"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _executable(path: Path, contents: str) -> None:
    path.write_text("#!/bin/sh\nset -eu\n" + contents + "\n")
    path.chmod(0o755)


def test_settings_reach_agent_and_real_controller_template(tmp_path):
    source = OmegaConf.to_container(load_experiment_config(CONFIG))
    source["agents"]["orchestrator"]["max_turns"] = 12
    source["agents"]["subagent"]["max_steps"] = 24
    source["agents"]["orchestrator"]["max_tokens"] = 3072
    source["agents"]["oracle"]["models"] = ["test/model-a", "test/model-b"]
    source["agents"]["oracle"]["checkpoint"] = "/custom/oracle.pkl"
    source["agents"]["subagent"]["model"] = "literal {{ example }}"
    source["actor_rollout_ref"]["rollout"]["temperature"] = 0.55
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(source))

    agent_config_json = build_agent_config(config_path, "test-run")
    with patch.dict(
        os.environ,
        {
            "ARBAS_AGENT_CONFIG": agent_config_json,
            "AGL_OPENAI_BASE_URL": "https://gateway.invalid/v1",
            "POLZA_API_KEY": "polza-secret",
        },
        clear=True,
    ):
        config = load_config()
    assert config.subagent.max_steps == 24
    assert config.orchestrator.max_turns == 12
    assert config.subagent.model == "literal {{ example }}"
    assert config.orchestrator.max_tokens == 3072
    assert config.orchestrator.temperature == 0.55
    assert config.oracle.models == ["test/model-a", "test/model-b"]
    assert config.sandbox.f2p_only is True
    assert config.judge.reasoning is None

    template = (CONFIG.parent / "job-template-openai.yaml").read_text()
    dataset = load_split_file(
        str(CONFIG.parent / "train.jsonl"),
        agent_config_json=agent_config_json,
        run_name="test-run",
    )
    rollout = Rollout(
        rollout_id="test-id",
        input=dataset[0],
        config=RolloutConfig(k8s=RolloutK8sConfig(job_template=template)),
        status=RolloutLifecycleStatus(created_at=1.0, updated_at=1.0),
    )
    controller = OmegaConf.create(
        {
            "agl_server": {"url": "http://server:8080", "key": "agl-secret"},
            "k8s_runner": {"namespace": "default", "ttl_after_finished": 60},
        }
    )
    job = build_job_spec(rollout, controller)
    container = job["spec"]["template"]["spec"]["containers"][0]
    assert f"$AGL_RUN_NAME-{job['metadata']['name']}.log" in container["args"][0]
    volumes = {item["name"]: item for item in job["spec"]["template"]["spec"]["volumes"]}
    assert volumes["agent-scripts"]["configMap"]["name"] == "swe-smith-agent-scripts"
    assert "oracle-scripts" not in volumes
    assert volumes["rollout-logs"]["hostPath"]["path"] == "/agl-logs"
    assert {mount["name"]: mount["mountPath"] for mount in container["volumeMounts"]} == {
        "agent-scripts": "/agl/hierarchical",
        "rollout-logs": "/agl-logs",
    }
    env = {item["name"]: item for item in container["env"]}
    parsed_env_cfg = AgentSettings.model_validate_json(env["ARBAS_AGENT_CONFIG"]["value"])
    assert parsed_env_cfg.subagent.model == "literal {{ example }}"
    assert parsed_env_cfg.subagent.max_steps == 24
    assert parsed_env_cfg.oracle.checkpoint == "/custom/oracle.pkl"
    assert env["AGL_RUN_NAME"]["value"] == "test-run"
    assert env["AGL_KEY"]["value"] == "agl-secret"
    for provider in ("polza", "openrouter"):
        assert env[f"{provider.upper()}_API_KEY"]["valueFrom"]["secretKeyRef"] == {
            "name": f"swe-smith-{provider}",
            "key": "api-key",
            "optional": True,
        }


def test_dataset_settings_survive_arrow_serialization(tmp_path):
    pa = pytest.importorskip("pyarrow")
    source = OmegaConf.to_container(load_experiment_config(CONFIG))
    source["agents"]["oracle"]["models"] = ["test/model-a", "test/model-b"]
    source["agents"]["oracle"]["checkpoint"] = "auto"
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(source))

    agent_config_json = build_agent_config(config_path, "test-run")
    parsed = AgentSettings.model_validate_json(agent_config_json)
    assert parsed.oracle.checkpoint == "/agl-logs/test-run/oracle/linucb.pkl"

    rows = load_split_file(
        str(CONFIG.parent / "train.jsonl"),
        agent_config_json=agent_config_json,
        run_name="test-run",
    )
    restored = pa.Table.from_pylist(rows).to_pylist()
    assert restored[0]["arbas_agent_config"] == agent_config_json
    assert restored[0]["agl_run_name"] == "test-run"


def test_allocate_run_dir_uses_mmdd_hh_and_collision_suffix(tmp_path):
    first = launch.allocate_run_dir(tmp_path, "a100-2b")
    assert re.fullmatch(r"a100-2b-\d{4}-\d{2}", first.name)
    second = launch.allocate_run_dir(tmp_path, "a100-2b")
    assert second.name == f"{first.name}-2"
    third = launch.allocate_run_dir(tmp_path, "a100-2b")
    assert third.name == f"{first.name}-3"


def test_warn_if_low_scratch_warns_under_16gib_without_aborting(tmp_path, capsys):
    Usage = namedtuple("usage", ["total", "used", "free"])
    gib = 1024**3

    with patch.object(launch.shutil, "disk_usage", return_value=Usage(total=420 * gib, used=400 * gib, free=20 * gib)):
        launch.warn_if_low_scratch(tmp_path)
    assert capsys.readouterr().err == ""

    with patch.object(launch.shutil, "disk_usage", return_value=Usage(total=200 * gib, used=185 * gib, free=15 * gib)):
        launch.warn_if_low_scratch(tmp_path)
    err = capsys.readouterr().err
    assert "WARNING:" in err
    assert "15.0 GiB free" in err
    assert "AGL_TMPDIR" in err

    with pytest.raises(SystemExit, match="temporary directory does not exist"):
        launch.warn_if_low_scratch(tmp_path / "missing-dir")


def test_scratch_dir_respects_slurm_hierarchy(monkeypatch, tmp_path):
    d_agl = tmp_path / "agl"
    d_slurm = tmp_path / "slurm"
    d_tmpdir = tmp_path / "tmpdir"

    monkeypatch.delenv("AGL_TMPDIR", raising=False)
    monkeypatch.delenv("SLURM_JOB_SCRATCH", raising=False)
    monkeypatch.delenv("TMPDIR", raising=False)
    assert launch._scratch_dir() == Path("/tmp")

    monkeypatch.setenv("TMPDIR", str(d_tmpdir))
    assert launch._scratch_dir() == d_tmpdir

    monkeypatch.setenv("SLURM_JOB_SCRATCH", str(d_slurm))
    assert launch._scratch_dir() == d_slurm

    monkeypatch.setenv("AGL_TMPDIR", str(d_agl))
    assert launch._scratch_dir() == d_agl


def test_run_sh_binds_hf_home_on_slurm(tmp_path):
    root = CONFIG.parents[2]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _executable(
        bin_dir / "python",
        'printf "%s\\n" "$HF_HOME" > "$MOCK_PY_ENV"\n',
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "VIRTUAL_ENV": sys.prefix,
        "MOCK_PY_ENV": str(tmp_path / "env"),
        "SLURM_CACHE_DIR": "/scratch/cache/alice",
        "HF_HOME": "/old/stale/path",
    }
    env.pop("ARBAS_MACHINE_FILE", None)

    res = subprocess.run(
        ["bash", "examples/swe_smith/run.sh", "train", "exp", "--config", "custom.yaml"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert (tmp_path / "env").read_text().strip() == "/scratch/cache/alice/huggingface"


def test_run_sh_loads_profile_preserves_cuda_and_execs_launcher(tmp_path):
    root = CONFIG.parents[2]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _executable(
        bin_dir / "python",
        'printf "%s\\n" "$@" > "$MOCK_PY_ARGS"\n'
        'printf "%s|%s|%s\\n" "$CUDA_VISIBLE_DEVICES" "${AGL_KEY:-}" "${SSH_TARGET:-}" > "$MOCK_PY_ENV"',
    )
    config_home = tmp_path / "home"
    arbas_dir = config_home / ".config/arbas"
    arbas_dir.mkdir(parents=True)
    (arbas_dir / "machine.env").write_text("AGL_KEY=train-key\nCUDA_VISIBLE_DEVICES=0,1\n")
    (arbas_dir / "ipavlov.env").write_text("SSH_TARGET=ipavlov-a100\nAGL_KEY=ctrl-key\n")
    env = {
        **os.environ,
        "HOME": str(config_home),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "VIRTUAL_ENV": sys.prefix,
        "CUDA_VISIBLE_DEVICES": "5",
        "MOCK_PY_ARGS": str(tmp_path / "args"),
        "MOCK_PY_ENV": str(tmp_path / "env"),
    }
    env.pop("ARBAS_MACHINE_FILE", None)

    res = subprocess.run(
        ["bash", "examples/swe_smith/run.sh", "train", "exp", "--config", "custom.yaml"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert (tmp_path / "args").read_text().splitlines() == [
        "-u",
        "-m",
        "examples.swe_smith.launch",
        "train",
        "exp",
        "--config",
        "custom.yaml",
    ]
    assert (tmp_path / "env").read_text().strip() == "5|train-key|"

    res2 = subprocess.run(
        ["bash", "examples/swe_smith/run.sh", "controller", "ipavlov"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert res2.returncode == 0, res2.stdout + res2.stderr
    assert (tmp_path / "env").read_text().strip() == "5|ctrl-key|ipavlov-a100"


@pytest.mark.parametrize(("include_routed_experts", "trainer_status"), [(False, 0), (True, 7)])
def test_train_launches_both_processes_with_named_logs(
    tmp_path, monkeypatch, capsys, include_routed_experts, trainer_status
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    port = _free_port()
    interpreter = shlex.quote(sys.executable)

    _executable(
        bin_dir / "python",
        "shift\n"  # drop "-u"
        "shift\n"  # drop train_smith_agent.py
        'printf "%s\\n" "$@" > "$MOCK_TRAINER_ARGS"\n'
        'printf "%s|%s|%s\\n" "$TMPDIR" "$RAY_TMPDIR" "$TENSORBOARD_DIR" > "$MOCK_TRAINER_ENV"\n'
        'printf "%s\\n" "$CUDA_VISIBLE_DEVICES"\n'
        'exit "$MOCK_TRAINER_STATUS"',
    )
    _executable(
        bin_dir / "agl-server",
        f'printf "%s\\n" "$@" > "$MOCK_SERVER_ARGS"\n'
        f'echo "server started with $AGL_KEY"\n'
        f'exec {interpreter} -c "\n'
        f"import http.server, os, signal, sys\n"
        f"from pathlib import Path\n"
        f'ready = Path(os.environ[\\"MOCK_SERVER_READY\\"])\n'
        f"class H(http.server.BaseHTTPRequestHandler):\n"
        f"  def do_GET(self):\n"
        f'    self.send_response(200); self.end_headers(); self.wfile.write(b\\"ok\\")\n'
        f"  def log_message(self, *a): pass\n"
        f'srv = http.server.ThreadingHTTPServer((\\"127.0.0.1\\", {port}), H)\n'
        f"ready.touch()\n"
        f"signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        f"srv.serve_forever()\n"
        f'"',
    )

    config = yaml.safe_load(CONFIG.read_text())
    config["actor_rollout_ref"]["rollout"]["include_routed_experts"] = include_routed_experts
    config["actor_rollout_ref"]["rollout"]["temperature"] = 0.85
    config["actor_rollout_ref"]["rollout"]["val_kwargs"] = {"temperature": 0.25}
    for split in ("train", "val"):
        dataset = CONFIG.parent / f"{split}.jsonl"
        (tmp_path / dataset.name).write_bytes(dataset.read_bytes())
        config["data"][f"{split}_files"] = dataset.name
    config["data"]["max_train_instances"] = 1
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config))

    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("AGL_KEY", "shared-test-key")
    monkeypatch.setenv("AGL_SERVER_PORT", str(port))
    monkeypatch.setenv("AGL_OUTPUT_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("AGL_TMPDIR", str(tmp_path))
    monkeypatch.setenv("MOCK_SERVER_READY", str(tmp_path / "ready"))
    monkeypatch.setenv("MOCK_SERVER_ARGS", str(tmp_path / "server-args"))
    monkeypatch.setenv("MOCK_TRAINER_ARGS", str(tmp_path / "trainer-args"))
    monkeypatch.setenv("MOCK_TRAINER_ENV", str(tmp_path / "trainer-env"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setenv("MOCK_TRAINER_STATUS", str(trainer_status))
    monkeypatch.setattr(launch.sys, "executable", str(bin_dir / "python"))

    with (
        patch.object(launch, "healthy", return_value=True),
        pytest.raises(SystemExit, match="already running"),
    ):
        launch.main(["train", "--config", str(config_path)])

    status = launch.main(["train", "h200-2b", "--config", str(config_path)])
    assert status == trainer_status
    captured = capsys.readouterr()

    server_args = (tmp_path / "server-args").read_text()
    assert "default_proxy.train.temperature=0.85" in server_args
    assert "default_proxy.val.temperature=0.25" in server_args
    assert f"default_proxy.include_routed_experts={str(include_routed_experts).lower()}" in server_args

    active = json.loads((tmp_path / "runs/active.json").read_text())["run_id"]
    assert re.fullmatch(r"h200-2b-\d{4}-\d{2}", active)
    run_dir = tmp_path / "runs" / active
    assert (tmp_path / "runs/latest").resolve() == run_dir
    snapshot = run_dir / "training.yaml"
    saved_config = yaml.safe_load(snapshot.read_text())
    assert saved_config["experiment"]["name"] == "h200-2b"
    assert saved_config["data"]["train_files"] == str(tmp_path / "train.jsonl")
    assert saved_config["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"]["MAX_JOBS"] == "2"
    assert "server started with [redacted]" in (run_dir / f"{active}-server.log").read_text()
    assert (run_dir / f"{active}-trainer.log").read_text().strip() == "5"
    trainer_args = (tmp_path / "trainer-args").read_text()
    assert str(snapshot) in trainer_args
    assert "--smoke" not in trainer_args
    seen_tmpdir, seen_ray_tmpdir, seen_tb_dir = (tmp_path / "trainer-env").read_text().strip().split("|")
    assert seen_tmpdir == seen_ray_tmpdir
    assert Path(seen_tmpdir).parent == tmp_path
    assert Path(seen_tb_dir) == run_dir / "tensorboard"
    assert "run.sh controller" in captured.out
    assert not list(tmp_path.glob("agl-tmp.*"))


@pytest.mark.parametrize("provider", ["openrouter", "polza"])
def test_controller_with_minimal_env_configures_single_configmap_and_tunnel(tmp_path, provider):
    root = CONFIG.parents[2]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    port = _free_port()
    interpreter = shlex.quote(sys.executable)
    _executable(bin_dir / "python", f'exec {interpreter} "$@"')

    _executable(
        bin_dir / "ssh",
        f'printf "%s\\n" "$*" >> "$MOCK_SSH_LOG"\n'
        f'case "$*" in\n'
        f'  *"-O exit"*)\n'
        f'    if [ -e "$MOCK_TUNNEL_PID" ]; then kill "$(cat "$MOCK_TUNNEL_PID")" 2>/dev/null || true; fi\n'
        f'    rm -f "$MOCK_TUNNEL_READY" "$MOCK_TUNNEL_PID";;\n'
        f'  *"cat "*)\n'
        f'    printf \'{{"run_id":"exp-1117-18","images":["repo/first","repo/second"]}}\';;\n'
        f"  *)\n"
        f'    {interpreter} -c "\n'
        f"import http.server\n"
        f"class H(http.server.BaseHTTPRequestHandler):\n"
        f'  def do_GET(self): self.send_response(200); self.end_headers(); self.wfile.write(b\\"ok\\")\n'
        f"  def log_message(self, *a): pass\n"
        f'http.server.ThreadingHTTPServer((\\"127.0.0.1\\", {port}), H).serve_forever()\n'
        f'" >/dev/null 2>&1 &\n'
        f'    echo $! > "$MOCK_TUNNEL_PID"\n'
        f'    touch "$MOCK_TUNNEL_READY"\n'
        f"    sleep 0.1;;\n"
        f"esac",
    )
    _executable(
        bin_dir / "kubectl",
        'printf "%s\\n" "$*" >> "$MOCK_KUBECTL_LOG"\n'
        'case "$*" in *"apply -f -"*) cat >/dev/null;; *"--dry-run=client"*) echo "kind: Secret";; esac',
    )
    _executable(bin_dir / "agl-controller", 'printf "%s\\n" "$@" > "$MOCK_CONTROLLER_ARGS"\necho ready')

    (tmp_path / "docker.py").write_text(
        "import os\n"
        "class Images:\n"
        "    def get(self, name):\n"
        "        assert not {'OPENROUTER_API_KEY', 'POLZA_API_KEY', 'AGL_KEY'} & os.environ.keys()\n"
        "        with open(os.environ['MOCK_IMAGES_LOG'], 'a') as f: f.write(name + '\\n')\n"
        "        return object()\n"
        "class Client:\n"
        "    images = Images()\n"
        "def from_env(): return Client()\n"
    )

    config_home = tmp_path / "home"
    arbas_dir = config_home / ".config/arbas"
    arbas_dir.mkdir(parents=True)
    # 4-field minimal profile; AGL_SERVER_PORT overridden to dynamic free port for local health server.
    (arbas_dir / "ipavlov.env").write_text(
        f"SSH_TARGET=gpu-host\n"
        f"REMOTE_REPO_DIR=/data/home/user/arbas\n"
        f"AGL_KEY=shared-test-key\n"
        f"{provider.upper()}_API_KEY=router-test-secret\n"
        f"AGL_SERVER_PORT={port}\n"
    )
    env = {
        **os.environ,
        "HOME": str(config_home),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "VIRTUAL_ENV": sys.prefix,
        "PYTHONPATH": f"{tmp_path}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        "AGL_OUTPUT_DIR": str(tmp_path / "runs"),
        "MOCK_TUNNEL_READY": str(tmp_path / "tunnel"),
        "MOCK_TUNNEL_PID": str(tmp_path / "tunnel.pid"),
        "MOCK_SSH_LOG": str(tmp_path / "ssh-log"),
        "MOCK_KUBECTL_LOG": str(tmp_path / "kubectl-log"),
        "MOCK_CONTROLLER_ARGS": str(tmp_path / "controller-args"),
        "MOCK_IMAGES_LOG": str(tmp_path / "images"),
    }
    env.pop("ARBAS_MACHINE_FILE", None)
    env.pop("OPENROUTER_API_KEY", None)
    env.pop("POLZA_API_KEY", None)

    result = subprocess.run(
        ["bash", "examples/swe_smith/run.sh", "controller", "ipavlov"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "router-test-secret" not in result.stdout + result.stderr
    assert "Prepared 2 SWE-smith :arbas images" in result.stdout
    assert set((tmp_path / "images").read_text().splitlines()) == {"repo/first:arbas", "repo/second:arbas"}
    assert (tmp_path / "runs/exp-1117-18/exp-1117-18-controller.log").read_text().strip() == "ready"

    kubectl_log = (tmp_path / "kubectl-log").read_text()
    assert "swe-smith-agent-scripts" in kubectl_log
    assert "swe-smith-oracle-scripts" not in kubectl_log
    assert f"create secret generic swe-smith-{provider}" in kubectl_log
    assert kubectl_log.count("create secret generic") == 1
    assert "router-test-secret" not in kubectl_log
    secret_path = re.search(r"--from-file=api-key=(\S+)", kubectl_log)
    assert secret_path and not Path(secret_path.group(1)).exists()

    ssh_log = (tmp_path / "ssh-log").read_text()
    assert f"0.0.0.0:{port}:127.0.0.1:8080" in ssh_log
    assert "/data/home/user/arbas/outputs/swe-smith/active.json" in ssh_log
    assert "-O exit" in ssh_log
    assert not (tmp_path / "tunnel").exists()

    controller_args = (tmp_path / "controller-args").read_text()
    assert "runner_type=k8s" in controller_args
    assert f"agl_server.agent_url=http://host.docker.internal:{port}" in controller_args
    assert "k8s_runner.namespace=default" in controller_args


@pytest.mark.parametrize("existing_tunnel", [True, False])
def test_controller_handles_optional_secret_and_existing_tunnel(tmp_path, existing_tunnel):
    root = CONFIG.parents[2]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    port = _free_port()
    interpreter = shlex.quote(sys.executable)
    _executable(bin_dir / "python", f'exec {interpreter} "$@"')
    _executable(
        bin_dir / "ssh",
        'printf "%s\\n" "$*" >> "$MOCK_SSH_LOG"\n'
        'case "$*" in *"-O exit"*) rm -f "$MOCK_TUNNEL_READY";; '
        '*"cat "*) printf \'{"run_id":"x","images":["repo/first","repo/second"]}\';; esac',
    )
    _executable(
        bin_dir / "kubectl",
        'printf "%s\\n" "$*" >> "$MOCK_KUBECTL_LOG"\n'
        'case "$*" in *"apply -f -"*) cat >/dev/null;; *"--dry-run=client"*) echo "kind: ConfigMap";; esac',
    )
    _executable(bin_dir / "agl-controller", "echo ready")

    (tmp_path / "docker.py").write_text(
        "import os\n"
        "class Images:\n"
        "    def get(self, name):\n"
        "        assert 'OPENROUTER_API_KEY' not in os.environ and 'AGL_KEY' not in os.environ\n"
        "        with open(os.environ['MOCK_IMAGES_LOG'], 'a') as f: f.write(name + '\\n')\n"
        "        return object()\n"
        "class Client:\n"
        "    images = Images()\n"
        "def from_env(): return Client()\n"
    )

    server = None
    if existing_tunnel:
        import http.server
        import threading

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        (tmp_path / "tunnel").touch()

    machine = tmp_path / "machine.env"
    machine.write_text(
        f"AGL_KEY=shared-test-key\nAGL_SERVER_PORT={port}\n"
        + ("SSH_TARGET=gpu-host\nAGL_REMOTE_OUTPUT_DIR='/path with spaces/outputs'\n" if existing_tunnel else "")
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "VIRTUAL_ENV": sys.prefix,
        "PYTHONPATH": f"{tmp_path}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        "ARBAS_MACHINE_FILE": str(machine),
        "AGL_OUTPUT_DIR": str(tmp_path / "runs"),
        "MOCK_TUNNEL_READY": str(tmp_path / "tunnel"),
        "MOCK_SSH_LOG": str(tmp_path / "ssh-log"),
        "MOCK_KUBECTL_LOG": str(tmp_path / "kubectl-log"),
        "MOCK_IMAGES_LOG": str(tmp_path / "images"),
    }
    env.pop("OPENROUTER_API_KEY", None)

    try:
        if not existing_tunnel:
            import http.server
            import threading

            class Handler2(http.server.BaseHTTPRequestHandler):
                def do_GET(self):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"ok")

                def log_message(self, *args):
                    pass

            server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler2)
            threading.Thread(target=server.serve_forever, daemon=True).start()

        result = subprocess.run(
            ["bash", "examples/swe_smith/run.sh", "controller"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()

    assert result.returncode == 0, result.stdout + result.stderr
    assert "create secret generic" not in (tmp_path / "kubectl-log").read_text()
    if existing_tunnel:
        assert (tmp_path / "tunnel").exists()
        assert "-O exit" not in (tmp_path / "ssh-log").read_text()
        assert "'/path with spaces/outputs/active.json'" in (tmp_path / "ssh-log").read_text()
        assert set((tmp_path / "images").read_text().splitlines()) == {"repo/first:arbas", "repo/second:arbas"}
    else:
        ctrl_dirs = list((tmp_path / "runs").glob("controller-*"))
        assert len(ctrl_dirs) == 1
        assert (ctrl_dirs[0] / f"{ctrl_dirs[0].name}-controller.log").read_text().strip() == "ready"


def test_controller_local_mode_runs_local_runner_without_k8s(tmp_path):
    root = CONFIG.parents[2]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    port = _free_port()
    interpreter = shlex.quote(sys.executable)
    _executable(bin_dir / "python", f'exec {interpreter} "$@"')
    _executable(bin_dir / "agl-controller", 'printf "%s\\n" "$@" > "$MOCK_CONTROLLER_ARGS"\necho ready')
    _executable(bin_dir / "kubectl", 'echo "kubectl should not be called" >&2; exit 1')
    _executable(bin_dir / "ssh", 'echo "ssh should not be called" >&2; exit 1')
    _executable(
        bin_dir / "docker",
        'printf "%s\\n" "$*" >> "$MOCK_DOCKER_LOG"\n'
        'exit 0',
    )

    (tmp_path / "docker.py").write_text(
        "import os\n"
        "class Images:\n"
        "    def get(self, name):\n"
        "        with open(os.environ['MOCK_IMAGES_LOG'], 'a') as f: f.write(name + '\\n')\n"
        "        return object()\n"
        "class Client:\n"
        "    images = Images()\n"
        "def from_env(): return Client()\n"
    )

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    active_run_dir = runs_dir / "exp-local-1"
    active_run_dir.mkdir()
    (runs_dir / "active.json").write_text(
        json.dumps({"run_id": "exp-local-1", "images": ["repo/local-task"]}),
        encoding="utf-8",
    )

    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "VIRTUAL_ENV": sys.prefix,
        "PYTHONPATH": f"{tmp_path}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        "AGL_OUTPUT_DIR": str(runs_dir),
        "AGL_SERVER_PORT": str(port),
        "MOCK_CONTROLLER_ARGS": str(tmp_path / "controller-args"),
        "MOCK_IMAGES_LOG": str(tmp_path / "images"),
        "MOCK_DOCKER_LOG": str(tmp_path / "docker-log"),
    }
    env.pop("ARBAS_MACHINE_FILE", None)
    env.pop("SSH_TARGET", None)
    env.pop("AGL_SSH_TARGET", None)

    try:
        result = subprocess.run(
            ["bash", "examples/swe_smith/run.sh", "controller", "local"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert result.returncode == 0, result.stdout + result.stderr
    docker_log = (tmp_path / "docker-log").read_text()
    assert "run --rm --network host repo/local-task:arbas" in docker_log
    assert f"http://127.0.0.1:{port}/healthz" in docker_log

    controller_args = (tmp_path / "controller-args").read_text()
    assert "runner_type=local" in controller_args
    assert f"agl_server.url=http://127.0.0.1:{port}" in controller_args
    assert "local_runner.maximum_size=32" in controller_args
    assert "k8s_runner" not in controller_args
    assert (active_run_dir / "exp-local-1-controller.log").read_text().strip() == "ready"
    assert (tmp_path / "images").read_text().strip() == "repo/local-task:arbas"


def test_controller_local_mode_fails_if_container_network_fails(tmp_path):
    root = CONFIG.parents[2]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    port = _free_port()
    interpreter = shlex.quote(sys.executable)
    _executable(bin_dir / "python", f'exec {interpreter} "$@"')
    _executable(bin_dir / "agl-controller", 'echo ready')
    # Docker fails the connectivity check
    _executable(
        bin_dir / "docker",
        'echo "Connection refused from inside container" >&2; exit 7',
    )

    (tmp_path / "docker.py").write_text(
        "import os\n"
        "class Images:\n"
        "    def get(self, name):\n"
        "        return object()\n"
        "class Client:\n"
        "    images = Images()\n"
        "def from_env(): return Client()\n"
    )

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    (runs_dir / "active.json").write_text(
        json.dumps({"run_id": "exp-local-fail", "images": ["repo/fail-task"]}),
        encoding="utf-8",
    )

    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "VIRTUAL_ENV": sys.prefix,
        "PYTHONPATH": f"{tmp_path}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        "AGL_OUTPUT_DIR": str(runs_dir),
        "AGL_SERVER_PORT": str(port),
    }
    env.pop("ARBAS_MACHINE_FILE", None)
    env.pop("SSH_TARGET", None)
    env.pop("AGL_SSH_TARGET", None)

    try:
        result = subprocess.run(
            ["bash", "examples/swe_smith/run.sh", "controller", "local"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert result.returncode != 0
    assert "ERROR: container cannot reach server" in result.stderr
    assert "In rootless Docker, --network=host attaches to RootlessKit's network namespace" in result.stderr
    assert "AGL_SERVER_AGENT_URL" in result.stderr


def test_trainer_and_server_roles_and_missing_active_json(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    port = _free_port()
    interpreter = shlex.quote(sys.executable)

    _executable(
        bin_dir / "python",
        'printf "%s\\n" "$@" > "$MOCK_TRAINER_ARGS"\nexit 0',
    )
    _executable(
        bin_dir / "agl-server",
        'echo "server secret=$AGL_KEY"\nexit 0',
    )

    cfg = yaml.safe_load(CONFIG.read_text())
    for split in ("train", "val"):
        dataset = CONFIG.parent / f"{split}.jsonl"
        (tmp_path / dataset.name).write_bytes(dataset.read_bytes())
        cfg["data"][f"{split}_files"] = dataset.name
    cfg["data"]["max_train_instances"] = 1
    cfg_path = tmp_path / "exp.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("AGL_KEY", "srv-secret")
    monkeypatch.setenv("AGL_SERVER_PORT", str(port))
    monkeypatch.setenv("AGL_OUTPUT_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("MOCK_TRAINER_ARGS", str(tmp_path / "trainer-args"))
    monkeypatch.setattr(launch.sys, "executable", str(bin_dir / "python"))

    assert launch.main(["server", "srv-exp", "--config", str(cfg_path)]) == 0
    run_id = json.loads((tmp_path / "runs/active.json").read_text())["run_id"]
    assert "server secret=[redacted]" in (tmp_path / "runs" / run_id / f"{run_id}-server.log").read_text()

    with (
        patch.object(launch, "healthy", return_value=True),
        pytest.raises(SystemExit, match=r"active\.json not found"),
    ):
        (tmp_path / "runs/active.json").unlink()
        launch.main(["trainer"])

    (tmp_path / "runs/active.json").write_text(json.dumps({"run_id": run_id, "images": []}))
    with patch.object(launch, "healthy", return_value=True):
        assert launch.main(["trainer"]) == 0
    assert str(tmp_path / "runs" / run_id / "training.yaml") in (tmp_path / "trainer-args").read_text()
    _executable(bin_dir / "python", f'exec {interpreter} "$@"')


def test_supervise_training_interrupt_startup_and_cleanup_order(tmp_path, monkeypatch):
    import signal
    import threading
    import time

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _executable(bin_dir / "slow-server", "trap 'exit 0' TERM\nwhile :; do sleep 0.1; done")

    old_int = signal.getsignal(signal.SIGINT)
    events: list[str] = []

    def trigger_sigint():
        time.sleep(0.2)
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=trigger_sigint, daemon=True).start()
    t0 = time.monotonic()
    rc = launch._supervise_training(
        trainer_args=["-c", "print('should-not-run')"],
        trainer_log=tmp_path / "trainer.log",
        server_cmd=[str(bin_dir / "slow-server")],
        server_log=tmp_path / "server.log",
        local_server_url=f"http://127.0.0.1:{_free_port()}",
    )
    elapsed = time.monotonic() - t0
    assert rc == 128 + signal.SIGINT
    assert elapsed < 3.0
    assert not (tmp_path / "trainer.log").exists()
    assert signal.getsignal(signal.SIGINT) is old_int

    # Verify stop_children runs before trainer_logger.join/log close so trailing output flushes cleanly.
    orphan_pid_file = tmp_path / "orphan.pid"
    trainer_script = (
        "import os, subprocess, sys, time; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)']); "
        f"open({str(orphan_pid_file)!r}, 'w').write(str(p.pid)); "
        "print('trailing-trainer-line', flush=True)"
    )

    def fake_stop_children(_children: Path) -> None:
        from contextlib import suppress

        events.append("stop_children")
        pid = int(orphan_pid_file.read_text())
        with suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
        with suppress(ChildProcessError):
            os.waitpid(pid, 0)

    monkeypatch.setattr(launch, "stop_children", fake_stop_children)
    orig_join = threading.Thread.join

    def tracked_join(self, timeout=None):
        events.append(f"join:{self.name}")
        return orig_join(self, timeout=timeout)

    monkeypatch.setattr(threading.Thread, "join", tracked_join)
    fake_proc_task = tmp_path / "proc_task"
    fake_children_dir = fake_proc_task / str(threading.get_native_id())
    fake_children_dir.mkdir(parents=True)
    (fake_children_dir / "children").write_text("")

    # Exercise the exact Linux cleanup path by mocking sys.platform and prctl/children.
    monkeypatch.setattr(launch.sys, "platform", "linux")
    with (
        patch.object(launch, "Path", wraps=Path) as mock_path,
        patch.object(launch.ctypes, "CDLL") as mock_cdll,
    ):
        mock_cdll.return_value.prctl.return_value = 0

        def path_dispatch(*args, **kwargs):
            if args == ("/proc/self/task",):
                return fake_proc_task
            return Path(*args, **kwargs)

        mock_path.side_effect = path_dispatch
        t1 = time.monotonic()
        rc2 = launch._supervise_training(
            trainer_args=["-c", trainer_script],
            trainer_log=tmp_path / "trainer-cleanup.log",
            server_cmd=None,
            server_log=None,
            local_server_url="http://127.0.0.1:1",
        )
        assert rc2 == 0
        assert time.monotonic() - t1 < 3.0
    assert "trailing-trainer-line" in (tmp_path / "trainer-cleanup.log").read_text()
    assert events[0] == "stop_children"
    assert any(e.startswith("join:") for e in events[1:])


def test_prepare_run_error_paths(tmp_path):
    with pytest.raises(SystemExit, match="config file does not exist"):
        launch.prepare_run(tmp_path / "missing.yaml", tmp_path, "valid-name")
    with pytest.raises(SystemExit, match="lowercase letters"):
        launch.prepare_run(CONFIG, tmp_path, "Invalid_Name")


def test_qwen35_chat_template_is_prefix_preserving():
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=False, lstrip_blocks=False)
    env.filters["tojson"] = lambda x: json.dumps(x, ensure_ascii=False)
    env.filters["items"] = lambda d: list(d.items())
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    template_path = Path(__file__).resolve().parents[2] / "examples/swe_smith/qwen35_chat_template.jinja"
    tmpl = env.from_string(template_path.read_text(encoding="utf-8"))
    tools = [{"type": "function", "function": {"name": "create_subagent", "parameters": {"type": "object"}}}]

    msgs = [
        {"role": "system", "content": "Sys"},
        {"role": "user", "content": "Issue"},
        {
            "role": "assistant",
            "reasoning_content": "First step\n\n",
            "content": None,
            "tool_calls": [{"function": {"name": "create_subagent", "arguments": {"name": "research"}}}],
        },
        {"role": "tool", "content": '{"agent_name": "research_1"}'},
        {
            "role": "assistant",
            "reasoning": None,
            "content": None,
            "tool_calls": [{"function": {"name": "create_subagent", "arguments": {"name": "fix"}}}],
        },
        {"role": "tool", "content": '{"agent_name": "fix_2"}'},
        {"role": "user", "content": "Retry one tool call."},
        {"role": "assistant", "reasoning_content": "Final\n", "content": "Done\n"},
    ]

    p0 = tmpl.render(messages=msgs[:2], tools=tools, add_generation_prompt=True)
    ctx0 = (
        p0
        + "First step\n\n</think>\n\n<tool_call>\n<function=create_subagent>\n"
        + "<parameter=name>\nresearch\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
    )
    p1 = tmpl.render(messages=msgs[:4], tools=tools, add_generation_prompt=True)
    assert p1.startswith(ctx0)
    ctx1 = (
        p1
        + "</think>\n\n<tool_call>\n<function=create_subagent>\n"
        + "<parameter=name>\nfix\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
    )
    p2 = tmpl.render(messages=msgs[:7], tools=tools, add_generation_prompt=True)
    assert p2.startswith(ctx1)
    p3 = tmpl.render(messages=msgs, tools=tools, add_generation_prompt=False)
    assert p3.startswith(p2)
    assert "<think>\nFirst step\n\n</think>" in p3

    no_think_messages = json.loads(json.dumps(msgs[:4]))
    no_think_messages[2].pop("reasoning_content")
    no_think_p0 = tmpl.render(
        messages=no_think_messages[:2], tools=tools, add_generation_prompt=True, enable_thinking=False
    )
    no_think_p1 = tmpl.render(
        messages=no_think_messages, tools=tools, add_generation_prompt=True, enable_thinking=False
    )
    assert no_think_p0.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert no_think_p1.startswith(no_think_p0)
