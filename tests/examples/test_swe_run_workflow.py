from __future__ import annotations

import http.server
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import namedtuple
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from omegaconf import DictConfig, OmegaConf

from examples.swe_smith import launch
from examples.swe_smith.hierarchical.config import AgentSettings, load_config
from examples.swe_smith.train_smith_agent import (
    build_agent_config,
    load_experiment_config,
    load_split_file,
    resolve_experiment,
)

CONFIG = Path(__file__).resolve().parents[2] / "examples/swe_smith/training.yaml"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _executable(path: Path, contents: str) -> None:
    path.write_text("#!/bin/sh\nset -eu\n" + contents + "\n", encoding="utf-8")
    path.chmod(0o755)


def _write_experiment(
    directory: Path, *, max_train_instances: int | None = 1, rollout_overrides: dict | None = None
) -> Path:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for split in ("train", "val"):
        dataset = CONFIG.parent / f"{split}.jsonl"
        (directory / dataset.name).write_bytes(dataset.read_bytes())
        cfg["data"][f"{split}_files"] = dataset.name
    if max_train_instances is not None:
        cfg["data"]["max_train_instances"] = max_train_instances
    if rollout_overrides:
        cfg["actor_rollout_ref"]["rollout"].update(rollout_overrides)
    cfg_path = directory / "experiment.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


@contextmanager
def _health_server(port: int):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _install_fake_docker(bin_dir: Path) -> None:
    _executable(
        bin_dir / "docker",
        'printf "%s\\n" "$*" >> "$MOCK_DOCKER_LOG"\n'
        'case "$1 $2" in\n'
        '  "image inspect")\n'
        '    env | grep -qE "^(OPENROUTER_API_KEY|POLZA_API_KEY|AGL_KEY)=" && exit 99 || true\n'
        '    printf "%s\\n" "$3" >> "$MOCK_IMAGES_LOG"\n'
        "    exit 1;;\n"
        '  "pull "*)\n'
        '    env | grep -qE "^(OPENROUTER_API_KEY|POLZA_API_KEY|AGL_KEY)=" && exit 99 || true\n'
        '    printf "%s\\n" "$2" >> "$MOCK_PULLED_LOG"\n'
        "    exit 0;;\n"
        '  "ps "*)\n'
        "    exit 0;;\n"
        "esac\n"
        "exit 0",
    )


def _stage_run_sh(tmp_path: Path, python_script: str) -> Path:
    repo = tmp_path / "staged-repo"
    script_dir = repo / "examples/swe_smith"
    venv_bin = repo / ".venv/bin"
    script_dir.mkdir(parents=True)
    venv_bin.mkdir(parents=True)
    shutil.copy2(CONFIG.parent / "run.sh", script_dir / "run.sh")
    _executable(venv_bin / "python", python_script)
    return repo


def test_settings_reach_agent_config_and_survive_arrow_serialization(tmp_path):
    pa = pytest.importorskip("pyarrow")
    source = OmegaConf.to_container(load_experiment_config(CONFIG))
    source["agents"]["orchestrator"]["max_turns"] = 12
    source["agents"]["subagent"]["max_steps"] = 24
    source["agents"]["orchestrator"]["max_tokens"] = 3072
    source["agents"]["oracle"]["models"] = ["test/model-a", "test/model-b"]
    source["agents"]["oracle"]["checkpoint"] = "auto"
    source["agents"]["subagent"]["model"] = "literal {{ example }}"
    source["actor_rollout_ref"]["rollout"]["temperature"] = 0.55
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(source), encoding="utf-8")

    agent_config_json = build_agent_config(config_path)
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
    assert config.oracle.checkpoint == "auto"
    assert config.sandbox.f2p_only is True
    assert config.judge.reasoning is None

    rows = load_split_file(
        str(CONFIG.parent / "train.jsonl"),
        agent_config_json=agent_config_json,
        run_name="test-run",
    )
    restored = pa.Table.from_pylist(rows).to_pylist()
    assert restored[0]["arbas_agent_config"] == agent_config_json
    assert restored[0]["agl_run_name"] == "test-run"
    eval_meta = json.loads(restored[0]["eval_meta"])
    assert eval_meta["instance_id"] == restored[0]["instance_id"]
    assert isinstance(eval_meta["FAIL_TO_PASS"], list)


def test_build_config_targets_host_rollout_worker_without_k8s():
    pytest.importorskip("verl")
    from examples.swe_smith.train_smith_agent import build_config

    cfg = build_config(config_path=CONFIG, run_name="test-run")
    assert cfg.agentlightning.local.agent_class == "examples.swe_smith.hierarchical.run:RolloutWorker"
    assert cfg.agentlightning.k8s.job_template_path is None
    assert dict(cfg.agentlightning.local.env_map) == {
        "IMAGE_NAME": "input.image_name",
        "AGL_TASK_INPUT": "input.problem_statement",
        "AGL_EVAL_META": "input.eval_meta",
        "ARBAS_AGENT_CONFIG": "input.arbas_agent_config",
        "AGL_RUN_NAME": "input.agl_run_name",
        "DATA_ID": "input.data_id",
    }


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


def test_run_sh_overrides_foreign_virtualenv_and_binds_hf_home_on_slurm(tmp_path):
    real_root = CONFIG.parents[2]
    interpreter = shlex.quote(sys.executable)
    staged = _stage_run_sh(
        tmp_path,
        f"PYTHONPATH={shlex.quote(str(real_root))} exec {interpreter} -c '\n"
        "import os, sys\n"
        "from pathlib import Path\n"
        "from unittest.mock import patch\n"
        "from examples.swe_smith import launch\n"
        "def _dump(*a, **kw):\n"
        '    Path(os.environ["MOCK_PY_ENV"]).write_text(\n'
        '        os.environ["VIRTUAL_ENV"] + "|" + os.environ["HF_HOME"]\n'
        "    )\n"
        "    return 0\n"
        'with patch.object(launch, "run_train", side_effect=_dump):\n'
        "    raise SystemExit(launch.main(sys.argv[4:]))\n"
        '\' "$@"\n',
    )
    foreign_bin = tmp_path / "foreign-venv/bin"
    foreign_bin.mkdir(parents=True)
    _executable(foreign_bin / "python", "exit 99")

    config_home = tmp_path / "home"
    config_home.mkdir()
    env = {
        **os.environ,
        "HOME": str(config_home),
        "PATH": f"{foreign_bin}{os.pathsep}{os.environ['PATH']}",
        "VIRTUAL_ENV": str(foreign_bin.parent),
        "MOCK_PY_ENV": str(tmp_path / "env"),
        "SLURM_CACHE_DIR": "/scratch/cache/alice",
        "HF_HOME": "/old/stale/path",
    }
    env.pop("ARBAS_MACHINE_FILE", None)

    res = subprocess.run(
        ["bash", "examples/swe_smith/run.sh", "train", "exp", "--config", "custom.yaml"],
        cwd=staged,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert (tmp_path / "env").read_text(encoding="utf-8").strip() == (
        f"{staged.resolve() / '.venv'}|/scratch/cache/alice/huggingface"
    )


def test_run_sh_loads_profile_preserves_cuda_and_execs_launcher(tmp_path):
    real_root = CONFIG.parents[2]
    interpreter = shlex.quote(sys.executable)
    staged = _stage_run_sh(
        tmp_path,
        f"PYTHONPATH={shlex.quote(str(real_root))} exec {interpreter} -c '\n"
        "import os, sys\n"
        "from pathlib import Path\n"
        "from unittest.mock import patch\n"
        "from examples.swe_smith import launch\n"
        'Path(os.environ["MOCK_PY_ARGS"]).write_text("\\n".join(sys.argv[1:]) + "\\n")\n'
        "def _dump(*a, **kw):\n"
        "    vals = [\n"
        '        os.environ.get("CUDA_VISIBLE_DEVICES", ""),\n'
        '        os.environ.get("AGL_KEY", ""),\n'
        '        os.environ.get("SSH_TARGET", ""),\n'
        "    ]\n"
        '    Path(os.environ["MOCK_PY_ENV"]).write_text("|".join(vals) + "\\n")\n'
        "    return 0\n"
        "with (\n"
        '    patch.object(launch, "run_train", side_effect=_dump),\n'
        '    patch.object(launch, "run_worker", side_effect=_dump),\n'
        "):\n"
        "    raise SystemExit(launch.main(sys.argv[4:]))\n"
        '\' "$@"\n',
    )
    config_home = tmp_path / "home"
    arbas_dir = config_home / ".config/arbas"
    arbas_dir.mkdir(parents=True)
    (arbas_dir / "machine.env").write_text("AGL_KEY=train-key\nCUDA_VISIBLE_DEVICES=0,1\n", encoding="utf-8")
    (arbas_dir / "ipavlov.env").write_text("SSH_TARGET=ipavlov-a100\nAGL_KEY=ctrl-key\n", encoding="utf-8")
    env = {
        **os.environ,
        "HOME": str(config_home),
        "CUDA_VISIBLE_DEVICES": "5",
        "MOCK_PY_ARGS": str(tmp_path / "args"),
        "MOCK_PY_ENV": str(tmp_path / "env"),
    }
    env.pop("ARBAS_MACHINE_FILE", None)

    res = subprocess.run(
        ["bash", "examples/swe_smith/run.sh", "train", "exp", "--config", "custom.yaml"],
        cwd=staged,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert (tmp_path / "args").read_text(encoding="utf-8").splitlines() == [
        "-u",
        "-m",
        "examples.swe_smith.launch",
        "train",
        "exp",
        "--config",
        "custom.yaml",
    ]
    assert (tmp_path / "env").read_text(encoding="utf-8").strip() == "5|train-key|"

    res2 = subprocess.run(
        ["bash", "examples/swe_smith/run.sh", "worker", "ipavlov"],
        cwd=staged,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert res2.returncode == 0, res2.stdout + res2.stderr
    assert (tmp_path / "env").read_text(encoding="utf-8").strip() == "5|ctrl-key|ipavlov-a100"

    res3 = subprocess.run(
        ["bash", "examples/swe_smith/run.sh", "train", "exp", "--profile", "ipavlov"],
        cwd=staged,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert res3.returncode == 0, res3.stdout + res3.stderr
    assert (tmp_path / "env").read_text(encoding="utf-8").strip() == "5|ctrl-key|ipavlov-a100"


@pytest.mark.parametrize(("include_routed_experts", "trainer_status"), [(False, 0), (True, 7)])
def test_remote_train_launches_gateway_and_trainer_with_named_logs(
    tmp_path, monkeypatch, capsys, include_routed_experts, trainer_status
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    port = _free_port()
    interpreter = shlex.quote(sys.executable)

    _executable(
        bin_dir / "python",
        'printf "%s\\n" "$@" > "$MOCK_TRAINER_ARGS"\n'
        'printf "%s|%s|%s|%s\\n" "$TMPDIR" "$RAY_TMPDIR" '
        '"$VERL_WEIGHT_TRANSFER_IPC_DIR" "$TENSORBOARD_DIR" > "$MOCK_TRAINER_ENV"\n'
        'printf "%s\\n" "$CUDA_VISIBLE_DEVICES"\n'
        'exit "$MOCK_TRAINER_STATUS"',
    )
    _executable(
        bin_dir / "agl-server",
        f'printf "%s\\n" "$@" > "$MOCK_SERVER_ARGS"\n'
        f'echo "server ready on {port}"\n'
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

    config_path = _write_experiment(
        tmp_path,
        rollout_overrides={
            "include_routed_experts": include_routed_experts,
            "temperature": 0.85,
            "val_kwargs": {"temperature": 0.25},
        },
    )

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("ARBAS_MACHINE_FILE", raising=False)
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
        launch.main(["train", "--remote", "--config", str(config_path)])

    status = launch.main(["train", "h200-2b", "--remote", "--config", str(config_path)])
    assert status == trainer_status
    captured = capsys.readouterr()

    server_args = (tmp_path / "server-args").read_text(encoding="utf-8")
    assert "default_proxy.train.temperature=0.85" in server_args
    assert "default_proxy.val.temperature=0.25" in server_args
    assert f"default_proxy.include_routed_experts={str(include_routed_experts).lower()}" in server_args

    active = json.loads((tmp_path / "runs/active.json").read_text(encoding="utf-8"))
    active_id = active["run_id"]
    assert re.fullmatch(r"h200-2b-\d{4}-\d{2}", active_id)
    assert active["max_workers"] == 8
    assert active["cpus"] == 1.0
    assert active["memory"] == "6g"
    run_dir = tmp_path / "runs" / active_id
    assert (tmp_path / "runs/latest").resolve() == run_dir
    snapshot = run_dir / "training.yaml"
    saved_config = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
    assert saved_config["experiment"]["name"] == "h200-2b"
    assert saved_config["data"]["train_files"] == str(tmp_path / "train.jsonl")
    assert saved_config["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"]["MAX_JOBS"] == "2"
    assert f"server ready on {port}" in (run_dir / f"{active_id}-server.log").read_text(encoding="utf-8")
    assert "shared-test-key" not in (run_dir / f"{active_id}-server.log").read_text(encoding="utf-8")
    assert (run_dir / f"{active_id}-trainer.log").read_text(encoding="utf-8").strip() == "5"
    trainer_args = (tmp_path / "trainer-args").read_text(encoding="utf-8").splitlines()
    assert trainer_args[:3] == ["-u", "-m", "examples.swe_smith.train_smith_agent"]
    assert str(snapshot) in trainer_args
    seen_tmpdir, seen_ray_tmpdir, seen_ipc_dir, seen_tb_dir = (
        (tmp_path / "trainer-env").read_text(encoding="utf-8").strip().split("|")
    )
    assert seen_tmpdir == seen_ray_tmpdir
    assert Path(seen_ipc_dir) == Path(seen_tmpdir) / "ipc"
    assert Path(seen_tmpdir).resolve().parent == tmp_path.resolve()
    assert Path(seen_tb_dir) == run_dir / "tensorboard"
    assert "Sandbox limits: per container cpus=1 memory=6g" in captured.out
    assert "run.sh worker" in captured.out
    assert not list(tmp_path.glob("agl-tmp.*"))


@pytest.mark.parametrize("controller_fails", [False, True])
def test_local_train_owns_controller_and_cleans_up_containers(tmp_path, monkeypatch, capsys, controller_fails):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    (runs_dir / "active.json").write_text('{"run_id":"another-run"}', encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("ARBAS_MACHINE_FILE", raising=False)
    monkeypatch.delenv("AGL_SERVER_PORT", raising=False)
    monkeypatch.delenv("AGL_MAX_LOCAL_JOBS", raising=False)
    monkeypatch.setenv("AGL_OUTPUT_DIR", str(runs_dir))
    monkeypatch.setenv("AGL_TMPDIR", str(tmp_path))

    config = tmp_path / "smoke.yaml"
    config.write_text(f"base: {CONFIG}\ndata:\n  max_train_instances: 1\n", encoding="utf-8")

    events: list[str] = []
    calls: list[dict] = []

    def fake_prepare(_images):
        events.append("prepare_images")

    def fake_alloc(_host):
        events.append("allocate_port")
        return 54321

    def supervise(**kwargs):
        calls.append(kwargs)
        return 0

    with (
        patch.object(launch, "_resolve_triton_cpath", return_value="/inc"),
        patch.object(launch, "healthy", return_value=False),
        patch.object(launch, "prepare_images", side_effect=fake_prepare),
        patch.object(launch, "_warn_if_docker_limits_unsupported"),
        patch.object(launch, "_allocate_ephemeral_port", side_effect=fake_alloc),
        patch.object(launch, "_supervise_training", side_effect=supervise),
    ):
        assert launch.main(["train", "local-smoke", "--config", str(config)]) == 0

    assert events == ["prepare_images", "allocate_port"]
    run_dir = (runs_dir / "latest").resolve()
    assert (runs_dir / "active.json").read_text(encoding="utf-8") == '{"run_id":"another-run"}'
    assert calls[0]["trainer_args"][:2] == ["-m", "examples.swe_smith.train_smith_agent"]
    assert str(run_dir / "training.yaml") in calls[0]["trainer_args"]
    controller_cmd = calls[0]["controller_cmd"]
    assert "agl_server.url=http://127.0.0.1:54321" in controller_cmd
    assert "local_runner.maximum_size=8" in controller_cmd
    assert not any("agent_url" in arg for arg in controller_cmd)
    assert "Container budget: max_workers=8, per container cpus=1 memory=6g" in capsys.readouterr().out

    controller = tmp_path / "controller"
    controller.write_text(
        f"#!{sys.executable}\nimport sys, time\nprint('controller-started')\n"
        + ("sys.exit(7)\n" if controller_fails else "time.sleep(60)\n"),
        encoding="utf-8",
    )
    controller.chmod(0o755)
    with patch.object(launch, "remove_run_containers") as cleanup:
        supervisor_args = {
            **calls[0],
            "trainer_args": ["-c", f"import time; time.sleep({2 if controller_fails else 0.3})"],
            "server_cmd": None,
            "controller_cmd": [str(controller)],
        }
        if controller_fails:
            with pytest.raises(SystemExit, match="controller stopped during training"):
                launch._supervise_training(**supervisor_args)
        else:
            assert launch._supervise_training(**supervisor_args) == 0
    assert (run_dir / f"{run_dir.name}-controller.log").read_text(encoding="utf-8").strip() == "controller-started"
    cleanup.assert_called_once_with(run_dir)


@pytest.mark.parametrize("provider", ["openrouter", "polza"])
def test_worker_opens_loopback_tunnel_pulls_base_images_and_cleans_up(tmp_path, provider):
    root = CONFIG.parents[2]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    port = _free_port()
    interpreter = shlex.quote(sys.executable)
    _install_fake_docker(bin_dir)

    _executable(
        bin_dir / "ssh",
        f'printf "%s\\n" "$*" >> "$MOCK_SSH_LOG"\n'
        f'case "$*" in\n'
        f'  *"-O exit"*)\n'
        f'    if [ -e "$MOCK_TUNNEL_PID" ]; then kill "$(cat "$MOCK_TUNNEL_PID")" 2>/dev/null || true; fi\n'
        f'    rm -f "$MOCK_TUNNEL_READY" "$MOCK_TUNNEL_PID";;\n'
        f'  *"cat "*)\n'
        f'    printf \'{{"run_id":"exp-1117-18","images":["repo/first","repo/second"],'
        f'"max_workers":4,"cpus":1.5,"memory":"6g"}}\';;\n'
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
    _executable(bin_dir / "agl-controller", 'printf "%s\\n" "$@" > "$MOCK_CONTROLLER_ARGS"\necho ready')

    config_home = tmp_path / "home"
    arbas_dir = config_home / ".config/arbas"
    arbas_dir.mkdir(parents=True)
    (arbas_dir / "ipavlov.env").write_text(
        f"SSH_TARGET=gpu-host\n"
        f"REMOTE_REPO_DIR=/data/home/user/arbas\n"
        f"AGL_KEY=shared-test-key\n"
        f"{provider.upper()}_API_KEY=router-test-secret\n"
        f"AGL_SERVER_PORT={port}\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "HOME": str(config_home),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "AGL_OUTPUT_DIR": str(tmp_path / "runs"),
        "MOCK_TUNNEL_READY": str(tmp_path / "tunnel"),
        "MOCK_TUNNEL_PID": str(tmp_path / "tunnel.pid"),
        "MOCK_SSH_LOG": str(tmp_path / "ssh-log"),
        "MOCK_CONTROLLER_ARGS": str(tmp_path / "controller-args"),
        "MOCK_DOCKER_LOG": str(tmp_path / "docker-log"),
        "MOCK_IMAGES_LOG": str(tmp_path / "images"),
        "MOCK_PULLED_LOG": str(tmp_path / "pulled"),
    }
    env.pop("ARBAS_MACHINE_FILE", None)
    env.pop("OPENROUTER_API_KEY", None)
    env.pop("POLZA_API_KEY", None)

    result = subprocess.run(
        [sys.executable, "-u", "-m", "examples.swe_smith.launch", "worker", "ipavlov"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "router-test-secret" not in result.stdout + result.stderr
    assert "Prepared 2 SWE-smith images." in result.stdout
    assert set((tmp_path / "images").read_text(encoding="utf-8").splitlines()) == {"repo/first", "repo/second"}
    assert set((tmp_path / "pulled").read_text(encoding="utf-8").splitlines()) == {"repo/first", "repo/second"}
    assert ":arbas" not in (tmp_path / "docker-log").read_text(encoding="utf-8")
    run_dir = tmp_path / "runs/exp-1117-18"
    assert (run_dir / "exp-1117-18-worker.log").read_text(encoding="utf-8").strip() == "ready"
    assert f"ps -aq --filter label=arbas.run={run_dir.resolve()}" in (tmp_path / "docker-log").read_text(
        encoding="utf-8"
    )

    ssh_log = (tmp_path / "ssh-log").read_text(encoding="utf-8")
    assert f"-L 127.0.0.1:{port}:127.0.0.1:8080" in ssh_log
    assert "0.0.0.0" not in ssh_log
    assert "/data/home/user/arbas/outputs/swe-smith/active.json" in ssh_log
    assert "-O exit" in ssh_log
    assert not (tmp_path / "tunnel").exists()

    controller_args = (tmp_path / "controller-args").read_text(encoding="utf-8")
    assert "runner_type=local" in controller_args
    assert f"agl_server.url=http://127.0.0.1:{port}" in controller_args
    assert "local_runner.maximum_size=4" in controller_args
    assert "agent_url" not in controller_args
    assert "Container budget: max_workers=4, per container cpus=1.5 memory=6g" in result.stdout


def test_worker_budget_respects_agl_max_local_jobs_and_warns_without_cgroups(tmp_path, monkeypatch, capsys):
    with (
        patch.object(launch.importlib.util, "find_spec", side_effect=lambda pkg: None if pkg == "openai" else object()),
        pytest.raises(SystemExit, match=r"\.\[worker\]"),
    ):
        launch._require_worker_deps()

    for bad in ("0", "-1", "abc"):
        monkeypatch.setenv("AGL_MAX_LOCAL_JOBS", bad)
        with pytest.raises(SystemExit, match="AGL_MAX_LOCAL_JOBS"):
            launch._resolve_worker_budget(default_max_workers=8, cpus=1.0, memory="6g")

    monkeypatch.setenv("AGL_MAX_LOCAL_JOBS", "2")
    assert launch._resolve_worker_budget(default_max_workers=8, cpus=1.0, memory="6g") == 2
    assert "max_workers=2, per container cpus=1 memory=6g (peak 2 CPUs, 12.0 GiB RAM)" in capsys.readouterr().out

    with patch.object(
        launch.subprocess,
        "run",
        return_value=subprocess.CompletedProcess([], 0, '{"MemoryLimit": false, "CpuCfsQuota": false}', ""),
    ):
        launch._warn_if_docker_limits_unsupported()
    assert "disabled cgroup limits (memory, cpu)" in capsys.readouterr().err


@pytest.mark.parametrize("existing_tunnel", [True, False])
def test_worker_handles_existing_tunnel_and_local_gateway(tmp_path, existing_tunnel):
    root = CONFIG.parents[2]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    port = _free_port()
    _install_fake_docker(bin_dir)
    _executable(
        bin_dir / "ssh",
        'printf "%s\\n" "$*" >> "$MOCK_SSH_LOG"\n'
        'case "$*" in *"cat "*) printf \'{"run_id":"x","images":["repo/first","repo/second"]}\';; esac',
    )
    _executable(bin_dir / "agl-controller", 'printf "%s\\n" "$@" > "$MOCK_CONTROLLER_ARGS"\necho ready')

    machine = tmp_path / "machine.env"
    machine.write_text(
        f"AGL_KEY=shared-test-key\nAGL_SERVER_PORT={port}\n"
        + ("SSH_TARGET=gpu-host\nAGL_REMOTE_OUTPUT_DIR='/path with spaces/outputs'\n" if existing_tunnel else ""),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "ARBAS_MACHINE_FILE": str(machine),
        "AGL_OUTPUT_DIR": str(tmp_path / "runs"),
        "MOCK_SSH_LOG": str(tmp_path / "ssh-log"),
        "MOCK_CONTROLLER_ARGS": str(tmp_path / "controller-args"),
        "MOCK_DOCKER_LOG": str(tmp_path / "docker-log"),
        "MOCK_IMAGES_LOG": str(tmp_path / "images"),
        "MOCK_PULLED_LOG": str(tmp_path / "pulled"),
    }

    with _health_server(port):
        result = subprocess.run(
            [sys.executable, "-u", "-m", "examples.swe_smith.launch", "worker"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "runner_type=local" in (tmp_path / "controller-args").read_text(encoding="utf-8")
    if existing_tunnel:
        ssh_log = (tmp_path / "ssh-log").read_text(encoding="utf-8")
        assert "-O exit" not in ssh_log
        assert "'/path with spaces/outputs/active.json'" in ssh_log
        assert set((tmp_path / "images").read_text(encoding="utf-8").splitlines()) == {"repo/first", "repo/second"}
    else:
        assert not (tmp_path / "ssh-log").exists()
        worker_dirs = list((tmp_path / "runs").glob("worker-*"))
        assert len(worker_dirs) == 1
        assert (worker_dirs[0] / f"{worker_dirs[0].name}-worker.log").read_text(encoding="utf-8").strip() == "ready"


def test_supervise_training_interrupt_startup_and_cleanup_order(tmp_path, monkeypatch):
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
        pid = int(orphan_pid_file.read_text(encoding="utf-8"))
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
    (fake_children_dir / "children").write_text("", encoding="utf-8")

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
    assert "trailing-trainer-line" in (tmp_path / "trainer-cleanup.log").read_text(encoding="utf-8")
    assert events[0] == "stop_children"
    assert any(e.startswith("join:") for e in events[1:])


def test_prepare_run_error_paths(tmp_path):
    with pytest.raises(SystemExit, match="config file does not exist"):
        launch.prepare_run(tmp_path / "missing.yaml", tmp_path, "valid-name")
    with pytest.raises(SystemExit, match="lowercase letters"):
        launch.prepare_run(CONFIG, tmp_path, "Invalid_Name")


def test_prepare_run_rejects_empty_val_dataset_before_allocating_dir(tmp_path):
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    train_ds = CONFIG.parent / "train.jsonl"
    (tmp_path / "train.jsonl").write_bytes(train_ds.read_bytes())
    (tmp_path / "val.jsonl").write_text("", encoding="utf-8")
    cfg["data"]["train_files"] = "train.jsonl"
    cfg["data"]["val_files"] = "val.jsonl"
    cfg["trainer"]["val_before_train"] = False
    cfg_path = tmp_path / "empty-val.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    with pytest.raises(SystemExit, match="invalid dataset"):
        launch.prepare_run(cfg_path, runs_dir, "empty-val")
    assert list(runs_dir.iterdir()) == []


def test_resolve_triton_cpath_checks_only_active_interpreter_include(tmp_path, monkeypatch):
    inc = tmp_path / "active-include"
    inc.mkdir()
    (inc / "Python.h").write_text("/* stub */\n", encoding="utf-8")
    foreign_inc = tmp_path / "foreign-include"
    foreign_inc.mkdir()
    (foreign_inc / "Python.h").write_text("/* stub */\n", encoding="utf-8")

    monkeypatch.setattr(launch.sysconfig, "get_path", lambda name: str(inc) if name == "include" else None)
    monkeypatch.setenv("CPATH", "/other/dir")
    assert launch._resolve_triton_cpath() == f"{inc}{os.pathsep}/other/dir"

    monkeypatch.setenv("CPATH", f"{inc}{os.pathsep}/other/dir")
    assert launch._resolve_triton_cpath() == f"{inc}{os.pathsep}/other/dir"

    missing_inc = tmp_path / "missing-include"
    missing_inc.mkdir()
    monkeypatch.setattr(launch.sysconfig, "get_path", lambda name: str(missing_inc) if name == "include" else None)
    monkeypatch.setenv("CPATH", str(foreign_inc))
    with pytest.raises(SystemExit, match="setup_training\\.sh"):
        launch._resolve_triton_cpath()


def test_run_train_checks_triton_cpath_on_linux(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    root.mkdir()
    cfg_path = _write_experiment(tmp_path, max_train_instances=None)

    inc = tmp_path / "py-include"
    inc.mkdir()
    (inc / "Python.h").write_text("/* stub */\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("ARBAS_MACHINE_FILE", raising=False)
    monkeypatch.setenv("AGL_OUTPUT_DIR", str(root))
    monkeypatch.setenv("AGL_TMPDIR", str(tmp_path))
    monkeypatch.setenv("AGL_SERVER_PORT", str(_free_port()))
    monkeypatch.delenv("CPATH", raising=False)
    monkeypatch.setattr(launch.sys, "platform", "linux")
    monkeypatch.setattr(launch.sysconfig, "get_path", lambda name: str(inc) if name == "include" else None)

    seen_envs = []

    def supervise(**kwargs):
        seen_envs.append(kwargs["trainer_env"])
        return 0

    with (
        patch.object(launch, "healthy", return_value=False),
        patch.object(launch, "_supervise_training", side_effect=supervise),
    ):
        assert launch.main(["train", "linux-exp", "--remote", "--config", str(cfg_path)]) == 0
    assert seen_envs[0]["CPATH"] == str(inc)


def test_long_scratch_creates_short_unix_socket_symlink_on_linux(tmp_path, monkeypatch):
    long_scratch = tmp_path / ("nested-scratch-dir-" * 4)
    long_scratch.mkdir(parents=True)
    assert not launch._sockets_fit_unix_limit(long_scratch / "agl-tmp.12345678", long_scratch / "agl-tmp.12345678/ipc")

    runs = tmp_path / "runs"
    runs.mkdir()
    cfg_path = _write_experiment(tmp_path, max_train_instances=None)

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("ARBAS_MACHINE_FILE", raising=False)
    monkeypatch.setenv("AGL_OUTPUT_DIR", str(runs))
    monkeypatch.setenv("AGL_TMPDIR", str(long_scratch))
    monkeypatch.setenv("AGL_SERVER_PORT", str(_free_port()))
    monkeypatch.setattr(launch.sys, "platform", "linux")

    captured = {}

    def supervise(**kwargs):
        env = kwargs["trainer_env"]
        captured["trainer_env"] = env
        captured["fits"] = launch._sockets_fit_unix_limit(
            Path(env["RAY_TMPDIR"]), Path(env["VERL_WEIGHT_TRANSFER_IPC_DIR"])
        )
        captured["resolved_parent"] = Path(env["RAY_TMPDIR"]).resolve().parent
        return 0

    with (
        patch.object(launch, "_resolve_triton_cpath", return_value="/inc"),
        patch.object(launch, "healthy", return_value=False),
        patch.object(launch, "_supervise_training", side_effect=supervise),
    ):
        assert launch.main(["train", "long-tmp", "--remote", "--config", str(cfg_path)]) == 0

    assert captured["fits"] is True
    assert captured["trainer_env"]["TMPDIR"] == captured["trainer_env"]["RAY_TMPDIR"]
    assert Path(captured["trainer_env"]["VERL_WEIGHT_TRANSFER_IPC_DIR"]) == (
        Path(captured["trainer_env"]["RAY_TMPDIR"]) / "ipc"
    )
    assert captured["trainer_env"]["RAY_TMPDIR"].startswith("/tmp/agl.")
    assert captured["resolved_parent"] == long_scratch.resolve()


@pytest.mark.parametrize("var", ["AGL_SERVER_PORT", "AGL_SSH_REMOTE_PORT"])
def test_configured_port_validation(monkeypatch, var):
    monkeypatch.delenv(var, raising=False)
    assert launch._configured_port(var) is None

    monkeypatch.setenv(var, "8080")
    assert launch._configured_port(var) == 8080

    for bad_port in ("0", "99999", "abc"):
        monkeypatch.setenv(var, bad_port)
        with pytest.raises(SystemExit, match=var):
            launch._configured_port(var)


def test_load_profile_env_preserves_bash_semantics(tmp_path, monkeypatch):
    config_home = tmp_path / "home"
    arbas_dir = config_home / ".config/arbas"
    arbas_dir.mkdir(parents=True)
    (arbas_dir / "bashy.env").write_text(
        "echo 'loading profile'\n"
        "VAR=outer\n"
        "AGL_KEY='literal$VAR' # inline comment\n"
        'EXPANDED="prefix_${VAR}_suffix"\n'
        'FALLBACK="${ARBAS_TEST_UNSET:-default_val}"\n'
        "CUDA_VISIBLE_DEVICES=0,1\n",
        encoding="utf-8",
    )
    (arbas_dir / "broken.env").write_text("UNCLOSED='oops\n", encoding="utf-8")

    monkeypatch.setenv("HOME", str(config_home))
    monkeypatch.delenv("ARBAS_MACHINE_FILE", raising=False)
    monkeypatch.delenv("ARBAS_TEST_UNSET", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    launch._load_profile_env("bashy")
    assert os.environ["AGL_KEY"] == "literal$VAR"
    assert os.environ["EXPANDED"] == "prefix_outer_suffix"
    assert os.environ["FALLBACK"] == "default_val"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"

    with pytest.raises(SystemExit, match="failed to load"):
        launch._load_profile_env("broken")


def test_cli_validation():
    for bad in (
        ["train", "smoke", "local"],
        ["worker", "--config", "foo.yaml"],
        ["worker", "h200", "--profile", "ipavlov"],
        ["controller"],
        ["worker", "Bad_Profile"],
    ):
        with pytest.raises(SystemExit):
            launch.parse_args(bad)

    train_remote = launch.parse_args(["train", "exp", "--remote"])
    assert (train_remote.command, train_remote.name, train_remote.remote) == ("train", "exp", True)

    worker_pos = launch.parse_args(["worker", "ipavlov"])
    assert (worker_pos.command, worker_pos.profile_name) == ("worker", "ipavlov")

    worker_flag = launch.parse_args(["worker", "--profile", "h200"])
    assert (worker_flag.command, worker_flag.profile_name) == ("worker", "h200")


def test_train_smith_agent_main_resolves_experiment_once(tmp_path, monkeypatch):
    from examples.swe_smith import train_smith_agent

    cfg_path = _write_experiment(tmp_path, max_train_instances=None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_smith_agent.py",
            "--config",
            str(cfg_path),
            "--agl-base-url",
            "http://127.0.0.1:8080",
            "--agl-key",
            "k",
            "agents.orchestrator.max_turns=9",
        ],
    )
    resolved = resolve_experiment(cfg_path, ["agents.orchestrator.max_turns=9"])
    assert resolved.agents.orchestrator.max_turns == 9

    fake_cfg = OmegaConf.create({"ok": True})
    with (
        patch.object(
            train_smith_agent,
            "load_experiment_config",
            wraps=train_smith_agent.load_experiment_config,
        ) as spy,
        patch.object(train_smith_agent, "build_config", return_value=fake_cfg) as mock_build_config,
        patch.object(train_smith_agent, "train") as mock_train,
    ):
        train_smith_agent.main()
    assert spy.call_count == 1
    assert isinstance(mock_build_config.call_args.kwargs["config_path"], DictConfig)
    mock_train.assert_called_once()
    parsed = AgentSettings.model_validate_json(mock_train.call_args.kwargs["agent_config_json"])
    assert parsed.orchestrator.max_turns == 9


def _qwen35_template():
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=False, lstrip_blocks=False)
    env.filters["tojson"] = lambda x: json.dumps(x, ensure_ascii=False)
    env.filters["items"] = lambda d: list(d.items())
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    template_path = Path(__file__).resolve().parents[2] / "examples/swe_smith/qwen35_chat_template.jinja"
    return env.from_string(template_path.read_text(encoding="utf-8"))


def test_qwen35_chat_template_is_prefix_preserving():
    tmpl = _qwen35_template()
    tools = [{"type": "function", "function": {"name": "assign_task", "parameters": {"type": "object"}}}]

    msgs = [
        {"role": "system", "content": "Sys"},
        {"role": "user", "content": "Issue"},
        {
            "role": "assistant",
            "reasoning_content": "First step\n\n",
            "content": None,
            "tool_calls": [
                {"function": {"name": "assign_task", "arguments": {"profile": "research", "prompt": "Find bug"}}}
            ],
        },
        {"role": "tool", "content": '{"agent_name": "research_1", "status": "completed"}'},
        {
            "role": "assistant",
            "reasoning": None,
            "content": None,
            "tool_calls": [{"function": {"name": "assign_task", "arguments": {"profile": "fix", "prompt": "Fix bug"}}}],
        },
        {"role": "tool", "content": '{"agent_name": "fix_2", "status": "completed"}'},
        {"role": "user", "content": "Retry one tool call."},
        {"role": "assistant", "reasoning_content": "Final\n", "content": "Done\n"},
    ]

    p0 = tmpl.render(messages=msgs[:2], tools=tools, add_generation_prompt=True)
    ctx0 = (
        p0
        + "First step\n\n</think>\n\n<tool_call>\n<function=assign_task>\n"
        + "<parameter=profile>\nresearch\n</parameter>\n"
        + "<parameter=prompt>\nFind bug\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
    )
    p1 = tmpl.render(messages=msgs[:4], tools=tools, add_generation_prompt=True)
    assert p1.startswith(ctx0)
    ctx1 = (
        p1
        + "</think>\n\n<tool_call>\n<function=assign_task>\n"
        + "<parameter=profile>\nfix\n</parameter>\n"
        + "<parameter=prompt>\nFix bug\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
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


def test_qwen35_chat_template_parameter_newline_handling():
    tmpl = _qwen35_template()

    msgs_trailing_nl = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "echo hello\n"}}}],
        }
    ]
    rendered_trailing = tmpl.render(messages=msgs_trailing_nl, add_generation_prompt=False)
    assert "<parameter=command>\necho hello\n</parameter>\n" in rendered_trailing
    assert "<parameter=command>\necho hello\n\n</parameter>\n" not in rendered_trailing

    msgs_no_nl = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "echo hello"}}}],
        }
    ]
    rendered_no_nl = tmpl.render(messages=msgs_no_nl, add_generation_prompt=False)
    assert "<parameter=command>\necho hello\n</parameter>\n" in rendered_no_nl

    msgs_various_types = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "function": {
                        "name": "complex_tool",
                        "arguments": {
                            "crlf_cmd": "echo win\r\n",
                            "count": 42,
                            "options": {"verbose": True},
                        },
                    }
                }
            ],
        }
    ]
    rendered_various = tmpl.render(messages=msgs_various_types, add_generation_prompt=False)
    assert "<parameter=crlf_cmd>\necho win\r\n</parameter>\n" in rendered_various
    assert "<parameter=count>\n42\n</parameter>\n" in rendered_various
    assert '<parameter=options>\n{"verbose": true}\n</parameter>\n' in rendered_various
