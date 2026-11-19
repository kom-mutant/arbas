"""Unified launcher and process supervisor for ARBAS training and K8s controller."""

from __future__ import annotations

import argparse
import ctypes
import http.client
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, TextIO
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from omegaconf import DictConfig

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXAMPLE_DIR / "training.yaml"
_LOW_DISK_WARN_BYTES = 16 * 1024**3
_PRINT_LOCK = threading.Lock()


def healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/healthz", timeout=2) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException):
        return False


def stop_children(children: Path) -> None:
    """Kill and reap any surviving descendants via Linux PR_SET_CHILD_SUBREAPER."""
    deadline = time.monotonic() + 10
    while True:
        try:
            while os.waitpid(-1, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            return
        for pid in children.read_text().split():
            with suppress(ProcessLookupError):
                os.kill(int(pid), signal.SIGKILL)
        if time.monotonic() >= deadline:
            raise RuntimeError("Trainer processes did not exit after SIGKILL.")
        time.sleep(0.05)


def copy_output(stream: TextIO, log: TextIO) -> None:
    key = os.environ.get("AGL_KEY", "")
    last_progress = None
    for line in stream:
        if "AglRolloutManager: completed=" in line:
            if line == last_progress:
                continue
            last_progress = line
        if key and key != "dummy":
            line = line.replace(key, "[redacted]")
        log.write(line)
        with _PRINT_LOCK:
            print(line, end="", flush=True)


def _stop_process(process: subprocess.Popen[str] | None, *, timeout: float = 5.0) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_logged_command(args: list[str], log_path: Path) -> int:
    def stop_command(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    previous_signals = {
        signum: signal.signal(signum, stop_command) for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        with (
            log_path.open("a", encoding="utf-8", buffering=1) as log,
            subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace"
            ) as process,
        ):
            try:
                assert process.stdout is not None
                copy_output(process.stdout, log)
                status = process.wait()
                return status if status >= 0 else 128 - status
            finally:
                _stop_process(process)
    finally:
        for signum, handler in previous_signals.items():
            signal.signal(signum, handler)


def _validate_run_name(name: str) -> None:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", name):
        raise SystemExit("ERROR: use up to 48 lowercase letters, digits and hyphens for the experiment name.")


def allocate_run_dir(root: Path, name: str) -> Path:
    _validate_run_name(name)
    stamp = datetime.now(ZoneInfo("Europe/Moscow")).strftime("%m%d-%H")
    base = f"{name}-{stamp}"
    suffix = 1
    while True:
        candidate = root / (base if suffix == 1 else f"{base}-{suffix}")
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            suffix += 1


def prepare_run(source: Path, root: Path, requested_name: str) -> tuple[Path, DictConfig]:
    from omegaconf import OmegaConf

    from .hierarchical.config import AgentSettings
    from .pull_images import experiment_images
    from .train_smith_agent import load_experiment_config

    if not source.expanduser().resolve().is_file():
        raise SystemExit(f"ERROR: config file does not exist: {source}")
    config = load_experiment_config(source)
    name = requested_name or str(OmegaConf.select(config, "experiment.name") or "")
    _validate_run_name(name)
    latest = root / "latest"
    if latest.exists() and not latest.is_symlink():
        raise SystemExit(f"ERROR: {latest} already exists and is not a symlink.")
    OmegaConf.update(config, "experiment.name", name, merge=True)

    for field in ("train_files", "val_files"):
        raw = OmegaConf.select(config, f"data.{field}")
        if not raw:
            raise SystemExit(f"ERROR: data.{field} is required in {source}")
        path = Path(str(raw))
        if not path.is_file():
            raise SystemExit(f"ERROR: {field} does not exist: {path}")

    agents = OmegaConf.to_container(config.agents, resolve=True)
    AgentSettings.model_validate(agents)

    images = experiment_images(config)
    run_dir = allocate_run_dir(root, name)
    OmegaConf.save(config, run_dir / "training.yaml")

    manifest = run_dir / ".active.json"
    manifest.write_text(json.dumps({"run_id": run_dir.name, "images": images}), encoding="utf-8")
    manifest.replace(root / "active.json")

    link = run_dir / ".latest"
    link.symlink_to(run_dir.name)
    link.replace(latest)
    return run_dir, config


def _scratch_dir() -> Path:
    return Path(
        os.environ.get("AGL_TMPDIR")
        or os.environ.get("SLURM_JOB_SCRATCH")
        or os.environ.get("TMPDIR")
        or "/tmp"
    ).expanduser()


def warn_if_low_scratch(scratch_parent: Path) -> None:
    if not scratch_parent.is_dir():
        raise SystemExit(f"ERROR: temporary directory does not exist: {scratch_parent}")
    free_bytes = shutil.disk_usage(scratch_parent).free
    if free_bytes < _LOW_DISK_WARN_BYTES:
        free_gib = free_bytes / 1024**3
        print(
            f"WARNING: low free space at {scratch_parent} ({free_gib:.1f} GiB free); "
            "set AGL_TMPDIR to move temporary files.",
            file=sys.stderr,
        )


def _server_command(config: DictConfig, port: int) -> list[str]:
    from omegaconf import OmegaConf

    actor = config.actor_rollout_ref
    rollout = actor.rollout
    include_experts = str(
        bool(
            OmegaConf.select(rollout, "include_routed_experts", default=False)
            or OmegaConf.select(config, "experiment.include_routed_experts", default=False)
        )
    ).lower()
    bind = os.environ.get("AGL_SERVER_BIND", "127.0.0.1")
    return [
        "agl-server",
        f"port={port}",
        f"host={bind}",
        "key=${oc.env:AGL_KEY}",
        f"default_proxy.model_name={actor.model.path}",
        f"default_proxy.train.temperature={rollout.temperature}",
        f"default_proxy.val.temperature={rollout.val_kwargs.temperature}",
        f"default_proxy.include_routed_experts={include_experts}",
        "hydra.run.dir=${oc.env:AGL_RUN_DIR}",
        "hydra.output_subdir=null",
        "hydra.job_logging.root.handlers=[console]",
    ]


def _supervise_training(
    *,
    trainer_args: list[str],
    trainer_log: Path,
    server_cmd: list[str] | None,
    server_log: Path | None,
    local_server_url: str,
    trainer_env: dict[str, str] | None = None,
) -> int:
    children: Path | None = None
    if sys.platform == "linux":
        children = Path("/proc/self/task") / str(threading.get_native_id()) / "children"
        children.read_text()
        libc = ctypes.CDLL(None, use_errno=True)
        pr_set_child_subreaper = 36
        if libc.prctl(pr_set_child_subreaper, 1, 0, 0, 0) != 0:
            raise SystemExit(f"Cannot supervise trainer processes: {os.strerror(ctypes.get_errno())}")

    stop_signal: int | None = None

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stop_signal
        stop_signal = signum

    previous_signals = {
        signum: signal.signal(signum, request_stop) for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }

    with TemporaryDirectory(prefix="agl-ipc-", dir=_scratch_dir(), ignore_cleanup_errors=True) as ipc_dir:
        proc_env = {**os.environ, **(trainer_env or {}), "VERL_WEIGHT_TRANSFER_IPC_DIR": ipc_dir}
        server_proc: subprocess.Popen[str] | None = None
        server_logger: threading.Thread | None = None
        server_log_file: TextIO | None = None
        trainer_proc: subprocess.Popen[str] | None = None
        trainer_logger: threading.Thread | None = None
        trainer_log_file: TextIO | None = None
        log_errors: list[OSError] = []
        server_died = False

        def stream_to_log(stream: TextIO, handle: TextIO) -> None:
            try:
                copy_output(stream, handle)
            except OSError as exc:
                log_errors.append(exc)

        try:
            if server_cmd is not None and server_log is not None:
                server_log_file = server_log.open("a", encoding="utf-8", buffering=1)
                server_proc = subprocess.Popen(
                    server_cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                )
                assert server_proc.stdout is not None
                server_logger = threading.Thread(
                    target=stream_to_log, args=(server_proc.stdout, server_log_file), daemon=True
                )
                server_logger.start()
                for _ in range(30):
                    if stop_signal is not None or healthy(local_server_url) or server_proc.poll() is not None:
                        break
                    time.sleep(1)
                if stop_signal is None and (server_proc.poll() is not None or not healthy(local_server_url)):
                    raise SystemExit(f"ERROR: server did not start. See {server_log}.")
                if stop_signal is None:
                    print("On the Mac: bash examples/swe_smith/run.sh controller [machine]", flush=True)

            if stop_signal is None:
                trainer_log_file = trainer_log.open("a", encoding="utf-8", buffering=1)
                trainer_proc = subprocess.Popen(
                    [sys.executable, "-u", *trainer_args],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                    env=proc_env,
                    start_new_session=True,
                )
                assert trainer_proc.stdout is not None
                trainer_logger = threading.Thread(
                    target=stream_to_log, args=(trainer_proc.stdout, trainer_log_file), daemon=True
                )
                trainer_logger.start()

                while trainer_proc.poll() is None and stop_signal is None and not log_errors:
                    if server_proc is not None and server_proc.poll() is not None:
                        server_died = True
                        break
                    time.sleep(0.1)

                if stop_signal is not None or server_died:
                    sig = stop_signal if stop_signal is not None else signal.SIGTERM
                    trainer_proc.send_signal(sig)
                    with suppress(subprocess.TimeoutExpired):
                        trainer_proc.wait(timeout=5)
        finally:
            _stop_process(trainer_proc)
            _stop_process(server_proc)
            if children is not None:
                stop_children(children)

            if trainer_logger is not None:
                trainer_logger.join(timeout=5)
            if trainer_proc is not None and trainer_proc.stdout is not None:
                trainer_proc.stdout.close()
            if trainer_log_file is not None:
                trainer_log_file.close()

            if server_logger is not None:
                server_logger.join(timeout=5)
            if server_proc is not None and server_proc.stdout is not None:
                server_proc.stdout.close()
            if server_log_file is not None:
                server_log_file.close()

            for signum, handler in previous_signals.items():
                signal.signal(signum, handler)

        if trainer_logger is not None and trainer_logger.is_alive():
            raise RuntimeError("Trainer output did not close after process cleanup.")
        if server_logger is not None and server_logger.is_alive():
            raise RuntimeError("Server output did not close after process cleanup.")
        if server_died:
            raise SystemExit(f"ERROR: server stopped during training. See {server_log}.")
        if log_errors:
            raise log_errors[0]
        if stop_signal is not None:
            return 128 + stop_signal
        assert trainer_proc is not None
        return trainer_proc.returncode if trainer_proc.returncode >= 0 else 128 - trainer_proc.returncode


def run_train(role: str, target: str, config_source: Path) -> int:
    from omegaconf import OmegaConf

    if not os.environ.get("AGL_KEY"):
        os.environ["AGL_KEY"] = "dummy"
    port = int(os.environ.get("AGL_SERVER_PORT", "8080"))
    local_server_url = f"http://127.0.0.1:{port}"
    output_root = Path(os.environ.get("AGL_OUTPUT_DIR") or (Path.cwd() / "outputs/swe-smith"))
    output_root.mkdir(parents=True, exist_ok=True)

    server_cmd: list[str] | None = None
    server_log: Path | None = None
    if role in ("train", "server"):
        if healthy(local_server_url):
            raise SystemExit(f"ERROR: a server is already running on port {port}.")
        run_dir, config = prepare_run(config_source, output_root, target)
        os.environ["AGL_RUN_DIR"] = str(run_dir)
        os.environ["AGL_RUN_NAME"] = run_dir.name
        config_path = run_dir / "training.yaml"
        server_log = run_dir / f"{run_dir.name}-server.log"
        server_cmd = _server_command(config, port)
        print(f"Experiment: {run_dir.name}")
        print(f"Server log: {server_log}")
        if role == "server":
            return run_logged_command(server_cmd, server_log)
    else:
        if not healthy(local_server_url):
            raise SystemExit("ERROR: start the server first, or use 'run.sh train'.")
        active_file = output_root / "active.json"
        if not active_file.is_file():
            raise SystemExit(
                f"ERROR: {active_file} not found; start the server with 'run.sh server' or 'run.sh train'."
            )
        active = json.loads(active_file.read_text(encoding="utf-8"))
        run_name = str(active["run_id"])
        run_dir = output_root / run_name
        os.environ["AGL_RUN_DIR"] = str(run_dir)
        os.environ["AGL_RUN_NAME"] = run_name
        config_path = run_dir / "training.yaml"
        if not config_path.is_file():
            raise SystemExit(f"ERROR: training config not found at {config_path}.")
        config = OmegaConf.load(config_path)

    trainer_log = run_dir / f"{run_dir.name}-trainer.log"
    tensorboard_dir = Path(os.environ.get("TENSORBOARD_DIR") or (run_dir / "tensorboard"))
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    scratch_parent = _scratch_dir()
    warn_if_low_scratch(scratch_parent)

    with TemporaryDirectory(prefix="agl-tmp.", dir=scratch_parent, ignore_cleanup_errors=True) as run_tmp:
        print(f"Trainer log: {trainer_log}", flush=True)
        trainer_args = [
            str(EXAMPLE_DIR / "train_smith_agent.py"),
            "--agl-base-url",
            local_server_url,
            "--config",
            str(config_path),
        ]
        return _supervise_training(
            trainer_args=trainer_args,
            trainer_log=trainer_log,
            server_cmd=server_cmd,
            server_log=server_log,
            local_server_url=local_server_url,
            trainer_env={
                "TMPDIR": run_tmp,
                "RAY_TMPDIR": run_tmp,
                "TENSORBOARD_DIR": str(tensorboard_dir),
            },
        )


def _kubectl_apply(namespace: str | None, create_args: list[str]) -> None:
    prefix = ["kubectl", *(["-n", namespace] if namespace else [])]
    rendered = subprocess.run(
        [*prefix, *create_args, "--dry-run=client", "-o", "yaml"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([*prefix, "apply", "-f", "-"], input=rendered.stdout, check=True, text=True)


def _prepare_remote_images(ssh_target: str, tunnel_control: Path | None, remote_active: str) -> str:
    cmd = [
        "ssh",
        *(["-S", str(tunnel_control)] if tunnel_control else []),
        ssh_target,
        f"cat {shlex.quote(remote_active)}",
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    active = json.loads(raw)
    run_id = active["run_id"]
    if not isinstance(run_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id):
        raise SystemExit("ERROR: invalid run_id in active.json")
    images = active["images"]
    print("Preparing images for the current experiment")
    pull_cmd = [sys.executable, str(EXAMPLE_DIR / "pull_images.py")]
    for image in images:
        pull_cmd.extend(["--image", str(image)])
    env = os.environ.copy()
    env.pop("OPENROUTER_API_KEY", None)
    env.pop("POLZA_API_KEY", None)
    env.pop("AGL_KEY", None)
    subprocess.run(pull_cmd, check=True, env=env)
    return run_id


def _openai_image_name(image: str) -> str:
    if __package__:
        from .pull_images import openai_image_name
    else:
        from pull_images import openai_image_name
    return openai_image_name(image)


def verify_container_health(agent_url: str, network: str, image: str) -> None:
    probe_script = f"import urllib.request; urllib.request.urlopen('{agent_url}/healthz', timeout=5)"
    probe_cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        network,
        image,
        "python",
        "-c",
        probe_script,
    ]
    print(f"Checking container network connectivity ({network} -> {agent_url}/healthz)...", flush=True)
    try:
        proc = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=15)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise SystemExit(
            f"ERROR: container network check failed: {exc}\n"
            f"Could not reach {agent_url}/healthz from inside container with --network={network}."
        ) from exc
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip()
        raise SystemExit(
            f"ERROR: container cannot reach server at {agent_url}/healthz using network '{network}'.\n"
            f"Output from container: {err}\n\n"
            "In rootless Docker, --network=host attaches to RootlessKit's network namespace,\n"
            "which has its own loopback. If the server is on the host loopback, 127.0.0.1 cannot\n"
            "be reached via 127.0.0.1 from inside the container.\n\n"
            "To fix this, specify how containers should reach the host:\n"
            "- Set AGL_SERVER_AGENT_URL=http://10.0.2.2:<port> (for slirp4netns default gateway)\n"
            "- Or set AGL_SERVER_AGENT_URL=http://<host-ip>:<port>\n"
            "- Or configure AGL_DOCKER_NETWORK=<network_name>\n"
            "See: https://docs.docker.com/engine/security/rootless/troubleshoot/#--nethost-doesnt-listen-ports-on-the-host-network-namespace"
        )
    print("Container network check passed.", flush=True)


def run_controller(runner: str = "k8s") -> int:
    if not os.environ.get("AGL_KEY"):
        os.environ["AGL_KEY"] = "dummy"

    output_root = Path(os.environ.get("AGL_OUTPUT_DIR") or (Path.cwd() / "outputs/swe-smith"))
    output_root.mkdir(parents=True, exist_ok=True)

    if runner == "local":
        port = os.environ.get("AGL_SERVER_PORT") or "8080"
        public_host = os.environ.get("AGL_SERVER_PUBLIC_HOST") or "127.0.0.1"
        server_url = f"http://{public_host}:{port}"
        agent_url = os.environ.get("AGL_SERVER_AGENT_URL") or server_url
        network = os.environ.get("AGL_DOCKER_NETWORK", "host")

        for _ in range(10):
            if healthy(server_url):
                break
            time.sleep(0.1)
        if not healthy(server_url):
            raise SystemExit(f"ERROR: server is not reachable at {server_url}.")

        active_file = output_root / "active.json"
        if active_file.is_file():
            active = json.loads(active_file.read_text(encoding="utf-8"))
            run_name = str(active["run_id"])
            run_dir = output_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            if images := active.get("images"):
                print("Preparing images for the current experiment", flush=True)
                pull_cmd = [sys.executable, str(EXAMPLE_DIR / "pull_images.py")]
                for img in images:
                    pull_cmd.extend(["--image", str(img)])
                env = os.environ.copy()
                env.pop("OPENROUTER_API_KEY", None)
                env.pop("AGL_KEY", None)
                subprocess.run(pull_cmd, check=True, env=env)
                test_image = _openai_image_name(str(images[0]))
                verify_container_health(agent_url, network, test_image)
        else:
            run_dir = allocate_run_dir(output_root, "controller")
            run_name = run_dir.name

        os.environ["AGL_RUN_DIR"] = str(run_dir)
        os.environ["AGL_RUN_NAME"] = run_name
        controller_log = run_dir / f"{run_name}-controller.log"
        print(f"Controller log: {controller_log}", flush=True)

        return run_logged_command(
            [
                "agl-controller",
                "runner_type=local",
                f"agl_server.url={server_url}",
                f"agl_server.agent_url={agent_url}",
                "agl_server.key=${oc.env:AGL_KEY}",
                "hydra.job_logging.root.level=WARNING",
                f"local_runner.maximum_size={os.environ.get('AGL_MAX_LOCAL_JOBS', '32')}",
                f"local_runner.poll_interval={os.environ.get('AGL_LOCAL_POLL_INTERVAL', '1')}",
                "hydra.run.dir=${oc.env:AGL_RUN_DIR}",
                "hydra.output_subdir=null",
                "hydra.job_logging.root.handlers=[console]",
            ],
            controller_log,
        )

    ssh_target = os.environ.get("SSH_TARGET") or os.environ.get("AGL_SSH_TARGET", "")
    port = os.environ.get("AGL_SERVER_PORT") or ("18080" if ssh_target else "8080")
    remote_port = os.environ.get("AGL_SSH_REMOTE_PORT") or "8080"
    ssh_bind = os.environ.get("AGL_SSH_BIND") or ("0.0.0.0" if ssh_target else "127.0.0.1")
    public_host = os.environ.get("AGL_SERVER_PUBLIC_HOST") or "127.0.0.1"
    server_url = f"http://{public_host}:{port}"
    namespace = os.environ.get("AGL_NAMESPACE") or "default"
    default_agent_url = (
        f"http://host.docker.internal:{port}"
        if public_host in ("127.0.0.1", "localhost") or public_host.startswith("127.")
        else server_url
    )
    agent_url = os.environ.get("AGL_SERVER_AGENT_URL") or default_agent_url

    if raw_output := os.environ.get("AGL_REMOTE_OUTPUT_DIR"):
        remote_active = f"{raw_output.rstrip('/')}/active.json"
    elif remote_repo := os.environ.get("REMOTE_REPO_DIR"):
        remote_active = f"{remote_repo.rstrip('/')}/outputs/swe-smith/active.json"
    else:
        remote_active = ""
    if remote_active and (not ssh_target or not remote_active.startswith("/")):
        raise SystemExit("ERROR: automatic image preparation needs SSH_TARGET and an absolute REMOTE_REPO_DIR.")

    tunnel_dir: Path | None = None
    tunnel_control: Path | None = None

    try:
        if ssh_target and not healthy(server_url):
            tunnel_dir = Path(tempfile.mkdtemp(prefix="arbas-ssh."))
            tunnel_control = tunnel_dir / "control"
            print(f"Starting SSH tunnel to {ssh_target}")
            subprocess.run(
                [
                    "ssh",
                    "-M",
                    "-S",
                    str(tunnel_control),
                    "-fnNT",
                    "-o",
                    "ExitOnForwardFailure=yes",
                    "-L",
                    f"{ssh_bind}:{port}:127.0.0.1:{remote_port}",
                    ssh_target,
                ],
                check=True,
            )
            for _ in range(50):
                if healthy(server_url):
                    break
                time.sleep(0.1)

        if not healthy(server_url):
            raise SystemExit(f"ERROR: server is not reachable at {server_url}.")

        _kubectl_apply(None, ["create", "namespace", namespace])
        _kubectl_apply(
            namespace,
            ["create", "configmap", "swe-smith-agent-scripts", f"--from-file={EXAMPLE_DIR / 'hierarchical'}"],
        )
        for provider in ("openrouter", "polza"):
            if api_key := os.environ.get(f"{provider.upper()}_API_KEY"):
                with TemporaryDirectory(prefix="arbas-secret.") as secret_dir:
                    secret_file = Path(secret_dir) / "api-key"
                    secret_file.write_text(api_key, encoding="utf-8")
                    _kubectl_apply(
                        namespace,
                        ["create", "secret", "generic", f"swe-smith-{provider}", f"--from-file=api-key={secret_file}"],
                    )

        if remote_active:
            run_name = _prepare_remote_images(ssh_target, tunnel_control, remote_active)
            run_dir = output_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = allocate_run_dir(output_root, "controller")
            run_name = run_dir.name

        os.environ["AGL_RUN_DIR"] = str(run_dir)
        controller_log = run_dir / f"{run_name}-controller.log"
        print(f"Controller log: {controller_log}", flush=True)
        return run_logged_command(
            [
                "agl-controller",
                "runner_type=k8s",
                f"agl_server.url={server_url}",
                f"agl_server.agent_url={agent_url}",
                "agl_server.key=${oc.env:AGL_KEY}",
                "hydra.job_logging.root.level=WARNING",
                f"k8s_runner.namespace={namespace}",
                f"k8s_runner.ttl_after_finished={os.environ.get('AGL_JOB_TTL', '3600')}",
                "hydra.run.dir=${oc.env:AGL_RUN_DIR}",
                "hydra.output_subdir=null",
                "hydra.job_logging.root.handlers=[console]",
            ],
            controller_log,
        )
    finally:
        if tunnel_control is not None:
            subprocess.run(
                ["ssh", "-S", str(tunnel_control), "-O", "exit", ssh_target],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        if tunnel_dir is not None:
            shutil.rmtree(tunnel_dir, ignore_errors=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run.sh",
        description="Launch ARBAS training on GPU or controller.",
    )
    parser.add_argument("role", choices=["train", "server", "trainer", "controller"])
    parser.add_argument(
        "target",
        nargs="?",
        default="",
        help="Experiment name (train/server) or runner/profile (controller)",
    )
    parser.add_argument(
        "machine",
        nargs="?",
        default="",
        help="Optional machine profile for controller (when target is a runner)",
    )
    parser.add_argument(
        "--runner",
        choices=["local", "k8s"],
        default=None,
        help="Controller runner mode (local or k8s)",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to experiment YAML (default: training.yaml)")
    args = parser.parse_args(argv)
    if args.role == "trainer" and args.target:
        parser.error("trainer does not take a positional argument")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.role == "controller":
        runner = args.runner or os.environ.get("AGL_RUNNER")
        if not runner:
            if args.target in ("local", "k8s"):
                runner = args.target
            else:
                runner = "k8s"
        return run_controller(runner=runner)
    env_cfg = os.environ.get("AGL_TRAIN_CONFIG")
    config_source = args.config or (Path(env_cfg) if env_cfg else DEFAULT_CONFIG)
    return run_train(args.role, args.target, config_source)


if __name__ == "__main__":
    raise SystemExit(main())
