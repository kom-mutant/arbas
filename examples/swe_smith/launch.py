"""Launch ARBAS training: ``train`` on a GPU machine, ``worker`` on a separate Docker machine."""

from __future__ import annotations

import argparse
import ctypes
import http.client
import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import sysconfig
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, TextIO
from zoneinfo import ZoneInfo

from .hierarchical.config import SandboxSettings, parse_memory_bytes
from .hierarchical.sandbox import remove_run_containers
from .pull_images import prepare_images

if TYPE_CHECKING:
    from omegaconf import DictConfig

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXAMPLE_DIR / "training.yaml"
DEFAULT_MAX_WORKERS = 8
_LOW_DISK_WARN_BYTES = 16 * 1024**3
_MAX_UNIX_SOCKET_BYTES = 107
_RAY_SOCKET_SUFFIX = Path("session_2000-01-01_00-00-00_000000_4194304") / "sockets" / "plasma_store"
_ZMQ_SOCKET_NAME = "rl-colocate-zmq-0000000000000000-replica-999-rank-99.sock"
_PROFILE_NAME_RE = re.compile(r"[a-z0-9-]+")
_BASH_NO_COPY_ENV = {"_", "SHLVL", "OLDPWD", "PWD"}


# --- process supervision -----------------------------------------------------------------------


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
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            ) as process,
        ):
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    print(line, end="", flush=True)
                status = process.wait()
                return status if status >= 0 else 128 - status
            finally:
                _stop_process(process)
    finally:
        for signum, handler in previous_signals.items():
            signal.signal(signum, handler)


# --- run & runtime directories -----------------------------------------------------------------


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


def prepare_run(
    source: Path, root: Path, requested_name: str, *, publish_active: bool = True
) -> tuple[Path, DictConfig, list[str], SandboxSettings]:
    from omegaconf import OmegaConf

    from .experiment import experiment_images, load_dataset, load_experiment_config
    from .hierarchical.config import AgentSettings

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
    settings = AgentSettings.model_validate(agents)

    try:
        load_dataset(Path(str(config.data.val_files)))
        images = experiment_images(config)
    except ValueError as exc:
        raise SystemExit(f"ERROR: invalid dataset in {source}: {exc}") from exc

    run_dir = allocate_run_dir(root, name)
    OmegaConf.save(config, run_dir / "training.yaml")

    if publish_active:
        manifest = run_dir / ".active.json"
        manifest.write_text(
            json.dumps(
                {
                    "run_id": run_dir.name,
                    "images": images,
                    "max_workers": settings.sandbox.max_workers,
                    "cpus": settings.sandbox.cpus,
                    "memory": settings.sandbox.memory,
                }
            ),
            encoding="utf-8",
        )
        manifest.replace(root / "active.json")

    link = run_dir / ".latest"
    link.symlink_to(run_dir.name)
    link.replace(latest)
    return run_dir, config, images, settings.sandbox


def _scratch_dir() -> Path:
    return Path(
        os.environ.get("AGL_TMPDIR") or os.environ.get("SLURM_JOB_SCRATCH") or os.environ.get("TMPDIR") or "/tmp"
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


def _sockets_fit_unix_limit(run_tmp: Path, ipc_dir: Path) -> bool:
    ray_socket = run_tmp.absolute() / _RAY_SOCKET_SUFFIX
    zmq_socket = ipc_dir.absolute() / _ZMQ_SOCKET_NAME
    return (
        len(os.fsencode(ray_socket)) <= _MAX_UNIX_SOCKET_BYTES
        and len(os.fsencode(zmq_socket)) <= _MAX_UNIX_SOCKET_BYTES
    )


@contextmanager
def allocate_runtime_dirs(scratch_parent: Path | None = None) -> Iterator[tuple[Path, Path]]:
    parent = scratch_parent or _scratch_dir()
    warn_if_low_scratch(parent)
    with ExitStack() as stack:
        run_tmp_real = Path(
            stack.enter_context(TemporaryDirectory(prefix="agl-tmp.", dir=parent, ignore_cleanup_errors=True))
        )
        (run_tmp_real / "ipc").mkdir()
        run_tmp = run_tmp_real
        ipc_dir = run_tmp_real / "ipc"
        if sys.platform == "linux" and not _sockets_fit_unix_limit(run_tmp.absolute(), ipc_dir.absolute()):
            # Ray and ZMQ append session/socket names to TMPDIR/IPC dir; AF_UNIX paths are capped at 107 bytes.
            alias_root = Path(stack.enter_context(TemporaryDirectory(prefix="agl.", dir="/tmp")))
            short_tmp = alias_root / "r"
            short_tmp.symlink_to(run_tmp_real.absolute(), target_is_directory=True)
            run_tmp = short_tmp
            ipc_dir = short_tmp / "ipc"
        yield run_tmp, ipc_dir


# --- GPU machine: gateway + trainer (+ local workers) -----------------------------------------


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


def _local_controller_command(server_url: str, max_workers: int) -> list[str]:
    """The ``agl-controller`` that spawns one host-side rollout process per queued rollout."""
    return [
        "agl-controller",
        "runner_type=local",
        f"agl_server.url={server_url}",
        "agl_server.key=${oc.env:AGL_KEY}",
        "hydra.job_logging.root.level=WARNING",
        f"local_runner.maximum_size={max_workers}",
        f"local_runner.poll_interval={os.environ.get('AGL_LOCAL_POLL_INTERVAL', '1')}",
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
    controller_cmd: list[str] | None = None,
) -> int:
    """Run gateway, optional controller and trainer; stop them all (and sandbox containers) on any exit."""
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

    with ExitStack() as stack:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            stack.callback(signal.signal, signum, signal.signal(signum, request_stop))

        proc_env = {**os.environ, **(trainer_env or {})}
        server_proc: subprocess.Popen[str] | None = None
        trainer_proc: subprocess.Popen[str] | None = None
        controller_proc: subprocess.Popen[str] | None = None
        controller_log = trainer_log.with_name(f"{trainer_log.parent.name}-controller.log")
        trainer_logger: threading.Thread | None = None
        log_error: OSError | None = None
        failed_service_log: Path | None = None
        cleanup_error: Exception | None = None

        service_env = {**os.environ, "PYTHONUNBUFFERED": "1"}

        def start_file_logged(command: list[str], path: Path) -> subprocess.Popen[str]:
            handle = stack.enter_context(path.open("a", encoding="utf-8", buffering=1))
            return subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=service_env,
                start_new_session=True,
            )

        def tee_trainer(stream: TextIO, handle: TextIO) -> None:
            nonlocal log_error
            try:
                for line in stream:
                    handle.write(line)
                    print(line, end="", flush=True)
            except OSError as exc:
                log_error = exc

        try:
            if server_cmd is not None and server_log is not None:
                server_proc = start_file_logged(server_cmd, server_log)
                for _ in range(30):
                    if stop_signal is not None or healthy(local_server_url) or server_proc.poll() is not None:
                        break
                    time.sleep(1)
                if stop_signal is None and (server_proc.poll() is not None or not healthy(local_server_url)):
                    raise SystemExit(f"ERROR: server did not start. See {server_log}.")
                if stop_signal is None and controller_cmd is None:
                    print("On the Docker machine: bash examples/swe_smith/run.sh worker [profile]", flush=True)

            if stop_signal is None and controller_cmd is not None:
                print(f"Controller log: {controller_log}", flush=True)
                controller_proc = start_file_logged(controller_cmd, controller_log)

            if stop_signal is None:
                trainer_handle = stack.enter_context(trainer_log.open("a", encoding="utf-8", buffering=1))
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
                stack.callback(trainer_proc.stdout.close)
                trainer_logger = threading.Thread(
                    target=tee_trainer, args=(trainer_proc.stdout, trainer_handle), daemon=True
                )
                trainer_logger.start()

                while trainer_proc.poll() is None and stop_signal is None and log_error is None:
                    if server_proc is not None and server_proc.poll() is not None:
                        failed_service_log = server_log
                        break
                    if controller_proc is not None and controller_proc.poll() is not None:
                        failed_service_log = controller_log
                        break
                    time.sleep(0.1)

                if stop_signal is not None or failed_service_log is not None:
                    sig = stop_signal if stop_signal is not None else signal.SIGTERM
                    trainer_proc.send_signal(sig)
                    with suppress(subprocess.TimeoutExpired):
                        trainer_proc.wait(timeout=5)
        finally:
            _stop_process(trainer_proc)
            _stop_process(controller_proc)
            _stop_process(server_proc)
            if children is not None:
                stop_children(children)
            if controller_proc is not None:
                try:
                    remove_run_containers(trainer_log.parent)
                except (OSError, subprocess.SubprocessError) as exc:
                    cleanup_error = exc
                    print(f"ERROR: could not clean up containers for {trainer_log.parent}: {exc}", file=sys.stderr)

            if trainer_logger is not None:
                trainer_logger.join(timeout=5)

        if trainer_logger is not None and trainer_logger.is_alive():
            raise RuntimeError(f"Process output did not close after cleanup. See {trainer_log}.")
        if failed_service_log is not None:
            service = "server" if failed_service_log == server_log else "controller"
            raise SystemExit(f"ERROR: {service} stopped during training. See {failed_service_log}.")
        if log_error is not None:
            raise log_error
        if stop_signal is not None:
            return 128 + stop_signal
        assert trainer_proc is not None
        if cleanup_error is not None and trainer_proc.returncode == 0:
            return 1
        return trainer_proc.returncode if trainer_proc.returncode >= 0 else 128 - trainer_proc.returncode


def _resolve_triton_cpath() -> str:
    include_dir = Path(sysconfig.get_path("include") or "")
    if not os.access(include_dir / "Python.h", os.R_OK):
        raise SystemExit(
            f"ERROR: Triton needs {include_dir / 'Python.h'}. "
            "Set up the training environment with examples/swe_smith/setup_training.sh."
        )
    existing = [entry for entry in os.environ.get("CPATH", "").split(os.pathsep) if entry]
    if str(include_dir) not in existing:
        existing.insert(0, str(include_dir))
    return os.pathsep.join(existing)


def _http_url(host: str, port: int) -> str:
    formatted = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{formatted}:{port}"


def _allocate_ephemeral_port(bind_host: str) -> int:
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    with socket.socket(family) as listener:
        listener.bind((bind_host, 0))
        return int(listener.getsockname()[1])


def _configured_int(variable: str, *, minimum: int, maximum: int | None = None) -> int | None:
    raw = os.environ.get(variable, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        value = minimum - 1
    if value < minimum or (maximum is not None and value > maximum):
        expected = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
        raise SystemExit(f"ERROR: invalid {variable}={raw!r}; expected {expected}.")
    return value


def _configured_port(variable: str) -> int | None:
    return _configured_int(variable, minimum=1, maximum=65535)


def _warn_if_docker_limits_unsupported() -> None:
    try:
        proc = subprocess.run(["docker", "info", "--format", "{{json .}}"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return
    if proc.returncode != 0 or not proc.stdout.strip():
        return
    try:
        info = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return
    if not isinstance(info, dict):
        return
    missing = [label for key, label in (("MemoryLimit", "memory"), ("CpuCfsQuota", "cpu")) if info.get(key) is False]
    if missing:
        print(
            f"WARNING: Docker daemon reports disabled cgroup limits ({', '.join(missing)}); "
            "per-container limits will not be enforced (on rootless Docker enable systemd cgroup v2 delegation).",
            file=sys.stderr,
        )


def _budget_summary(max_workers: int, cpus: float, memory: str) -> str:
    total_ram_gib = max_workers * parse_memory_bytes(memory) / 1024**3
    return (
        f"per container cpus={cpus:g} memory={memory} (peak {max_workers * cpus:g} CPUs, {total_ram_gib:.1f} GiB RAM)"
    )


def _require_worker_deps() -> None:
    missing = [pkg for pkg in ("openai", "numpy") if importlib.util.find_spec(pkg) is None]
    if missing:
        raise SystemExit(
            f"ERROR: rollout worker dependencies missing ({', '.join(missing)}); "
            'install them with: uv pip install -e ".[worker]"'
        )


def _resolve_worker_budget(*, default_max_workers: int, cpus: float | None = None, memory: str | None = None) -> int:
    max_workers = _configured_int("AGL_MAX_LOCAL_JOBS", minimum=1) or default_max_workers
    if cpus is not None and memory is not None:
        print(f"Container budget: max_workers={max_workers}, {_budget_summary(max_workers, cpus, memory)}", flush=True)
    else:
        print(f"Container budget: max_workers={max_workers}", flush=True)
    return max_workers


def run_train(name: str, config_source: Path, *, remote: bool = False) -> int:
    """Run gateway and trainer. Rollouts run here too, unless ``remote`` leaves them to a ``worker``."""
    if not os.environ.get("AGL_KEY"):
        os.environ["AGL_KEY"] = "dummy"
    cpath = _resolve_triton_cpath() if sys.platform == "linux" else None
    bind_host = os.environ.get("AGL_SERVER_BIND", "127.0.0.1")
    client_host = "127.0.0.1" if bind_host in ("0.0.0.0", "::", "") else bind_host
    port = _configured_port("AGL_SERVER_PORT") or (8080 if remote else None)
    server_url = _http_url(client_host, port) if port is not None else ""
    if port is not None and healthy(server_url):
        raise SystemExit(f"ERROR: a server is already running on port {port}.")
    output_root = Path(os.environ.get("AGL_OUTPUT_DIR") or (Path.cwd() / "outputs/swe-smith"))
    output_root.mkdir(parents=True, exist_ok=True)

    run_dir, config, images, sandbox_cfg = prepare_run(config_source, output_root, name, publish_active=remote)
    os.environ["AGL_RUN_DIR"] = str(run_dir)
    os.environ["AGL_RUN_NAME"] = run_dir.name
    server_log = run_dir / f"{run_dir.name}-server.log"
    print(f"Experiment: {run_dir.name}")
    print(f"Server log: {server_log}")

    controller_cmd = None
    if not remote:
        _require_worker_deps()
        prepare_images(images)
        _warn_if_docker_limits_unsupported()
        max_workers = _resolve_worker_budget(
            default_max_workers=sandbox_cfg.max_workers, cpus=sandbox_cfg.cpus, memory=sandbox_cfg.memory
        )
        if port is None:
            port = _allocate_ephemeral_port(bind_host)
            server_url = _http_url(client_host, port)
            if healthy(server_url):
                raise SystemExit(f"ERROR: a server is already running on port {port}.")
        controller_cmd = _local_controller_command(server_url, max_workers)
    else:
        summary = _budget_summary(sandbox_cfg.max_workers, sandbox_cfg.cpus, sandbox_cfg.memory)
        print(f"Sandbox limits: {summary} (default max_workers={sandbox_cfg.max_workers})", flush=True)
    assert port is not None

    trainer_log = run_dir / f"{run_dir.name}-trainer.log"
    tensorboard_dir = Path(os.environ.get("TENSORBOARD_DIR") or (run_dir / "tensorboard"))
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    with allocate_runtime_dirs() as (run_tmp, ipc_dir):
        trainer_env = {
            "TMPDIR": str(run_tmp),
            "RAY_TMPDIR": str(run_tmp),
            "VERL_WEIGHT_TRANSFER_IPC_DIR": str(ipc_dir),
            "TENSORBOARD_DIR": str(tensorboard_dir),
        }
        if cpath:
            trainer_env["CPATH"] = cpath
        print(f"Trainer log: {trainer_log}", flush=True)
        trainer_args = [
            "-m",
            "examples.swe_smith.train_smith_agent",
            "--agl-base-url",
            server_url,
            "--config",
            str(run_dir / "training.yaml"),
        ]
        return _supervise_training(
            trainer_args=trainer_args,
            trainer_log=trainer_log,
            server_cmd=_server_command(config, port),
            server_log=server_log,
            local_server_url=server_url,
            controller_cmd=controller_cmd,
            trainer_env=trainer_env,
        )


# --- Docker machine: SSH tunnel + rollout workers ----------------------------------------------


@contextmanager
def _ssh_tunnel(target: str, local_port: int, remote_port: int) -> Iterator[Path]:
    """Forward ``127.0.0.1:local_port`` to the gateway on ``target``. Yields the SSH control socket."""
    with TemporaryDirectory(prefix="arbas-ssh.") as tunnel_dir:
        control = Path(tunnel_dir) / "control"
        print(f"Starting SSH tunnel to {target}")
        subprocess.run(
            [
                "ssh", "-M", "-S", str(control), "-fnNT", "-o", "ExitOnForwardFailure=yes",
                "-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}", target,
            ],
            check=True,
        )  # fmt: skip
        try:
            yield control
        finally:
            subprocess.run(
                ["ssh", "-S", str(control), "-O", "exit", target],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )


def _read_remote_active(ssh_target: str, control: Path | None, remote_active: str) -> dict:
    cmd = ["ssh", *(["-S", str(control)] if control else []), ssh_target, f"cat {shlex.quote(remote_active)}"]
    active = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    if not isinstance(active["run_id"], str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", active["run_id"]):
        raise SystemExit("ERROR: invalid run_id in active.json")
    return active


def run_worker() -> int:
    """Run rollouts for a ``train --remote`` gateway, tunnelling to it over SSH when ``SSH_TARGET`` is set."""
    _require_worker_deps()
    if not os.environ.get("AGL_KEY"):
        os.environ["AGL_KEY"] = "dummy"
    ssh_target = os.environ.get("SSH_TARGET") or os.environ.get("AGL_SSH_TARGET", "")
    port = _configured_port("AGL_SERVER_PORT") or (18080 if ssh_target else 8080)
    remote_port = _configured_port("AGL_SSH_REMOTE_PORT") or 8080
    server_url = _http_url("127.0.0.1", port)
    output_root = Path(os.environ.get("AGL_OUTPUT_DIR") or (Path.cwd() / "outputs/swe-smith"))
    output_root.mkdir(parents=True, exist_ok=True)

    if raw_output := os.environ.get("AGL_REMOTE_OUTPUT_DIR"):
        remote_active = f"{raw_output.rstrip('/')}/active.json"
    elif remote_repo := os.environ.get("REMOTE_REPO_DIR"):
        remote_active = f"{remote_repo.rstrip('/')}/outputs/swe-smith/active.json"
    else:
        remote_active = ""
    if remote_active and (not ssh_target or not remote_active.startswith("/")):
        raise SystemExit("ERROR: automatic image preparation needs SSH_TARGET and an absolute REMOTE_REPO_DIR.")

    with ExitStack() as stack:
        control = None
        if ssh_target and not healthy(server_url):
            control = stack.enter_context(_ssh_tunnel(ssh_target, port, remote_port))
            for _ in range(50):
                if healthy(server_url):
                    break
                time.sleep(0.1)
        if not healthy(server_url):
            raise SystemExit(f"ERROR: server is not reachable at {server_url}.")

        local_active = output_root / "active.json"
        if remote_active:
            active = _read_remote_active(ssh_target, control, remote_active)
        elif local_active.is_file():
            active = json.loads(local_active.read_text(encoding="utf-8"))
        else:
            active = None
        if active is not None:
            run_dir = output_root / str(active["run_id"])
            run_dir.mkdir(parents=True, exist_ok=True)
            if active.get("images"):
                prepare_images(active["images"])
            raw_max = active.get("max_workers")
            default_max = int(raw_max) if isinstance(raw_max, int) and raw_max >= 1 else DEFAULT_MAX_WORKERS
            cpus = float(active["cpus"]) if active.get("cpus") is not None else None
            memory = str(active["memory"]) if active.get("memory") is not None else None
        else:
            run_dir = allocate_run_dir(output_root, "worker")
            default_max, cpus, memory = DEFAULT_MAX_WORKERS, None, None

        _warn_if_docker_limits_unsupported()
        max_workers = _resolve_worker_budget(default_max_workers=default_max, cpus=cpus, memory=memory)

        os.environ["AGL_RUN_DIR"] = str(run_dir)
        os.environ["AGL_RUN_NAME"] = run_dir.name
        worker_log = run_dir / f"{run_dir.name}-worker.log"
        print(f"Worker log: {worker_log}", flush=True)
        try:
            return run_logged_command(_local_controller_command(server_url, max_workers), worker_log)
        finally:
            try:
                remove_run_containers(run_dir)
            except (OSError, subprocess.SubprocessError) as exc:
                print(f"ERROR: could not clean up containers for {run_dir}: {exc}", file=sys.stderr)


# --- command line ------------------------------------------------------------------------------


def _load_profile_env(profile: str | None) -> None:
    allocated_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

    if slurm_cache := os.environ.get("SLURM_CACHE_DIR"):
        os.environ["HF_HOME"] = f"{slurm_cache}/huggingface"

    machine_file = os.environ.get("ARBAS_MACHINE_FILE")
    if machine_file:
        env_file = Path(machine_file).expanduser()
    else:
        env_file = Path.home() / ".config/arbas" / f"{profile or 'machine'}.env"
    if not env_file.is_file():
        if machine_file or profile:
            raise SystemExit(f"ERROR: machine settings not found at {env_file}.")
        return

    proc = subprocess.run(
        ["bash", "-eua", "-c", 'source "$1" >&2; command env -0', "arbas-env", str(env_file.absolute())],
        capture_output=True,
    )
    if proc.returncode:
        detail = os.fsdecode(proc.stderr).strip() or f"exit {proc.returncode}"
        raise SystemExit(f"ERROR: failed to load {env_file}: {detail}")
    loaded = dict(entry.split("=", 1) for entry in os.fsdecode(proc.stdout).split("\0") if entry)
    for key in os.environ.keys() - loaded.keys() - _BASH_NO_COPY_ENV:
        del os.environ[key]
    os.environ.update({key: value for key, value in loaded.items() if key not in _BASH_NO_COPY_ENV})

    if allocated_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = allocated_devices


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="run.sh", description="Launch ARBAS training.")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Run the gateway and trainer, with rollout workers on this machine.")
    train.add_argument("name", nargs="?", default="", help="Experiment name (default: experiment.name from the YAML)")
    train.add_argument("--config", type=Path, default=None, help="Experiment YAML (default: training.yaml)")
    train.add_argument("--profile", default=None, help="Machine profile from ~/.config/arbas/<name>.env")
    train.add_argument(
        "--remote", action="store_true", help="Leave rollouts to a 'worker' on another machine that has Docker."
    )

    worker = commands.add_parser("worker", help="Run rollouts for a 'train --remote' gateway.")
    worker.add_argument("profile_pos", nargs="?", default="", metavar="profile", help="Machine profile name")
    worker.add_argument("--profile", default=None, help="Machine profile from ~/.config/arbas/<name>.env")

    args = parser.parse_args(argv)
    positional = getattr(args, "profile_pos", "")
    if args.profile is not None and positional:
        parser.error("specify profile either positionally or with --profile, not both")
    args.profile_name = args.profile if args.profile is not None else (positional or None)
    if args.profile_name is not None and not _PROFILE_NAME_RE.fullmatch(args.profile_name):
        parser.error("invalid profile name; use lowercase letters, digits and hyphens")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _load_profile_env(args.profile_name)
    if args.command == "worker":
        return run_worker()
    env_config = os.environ.get("AGL_TRAIN_CONFIG")
    config_source = args.config or (Path(env_config) if env_config else DEFAULT_CONFIG)
    return run_train(args.name, config_source, remote=args.remote)


if __name__ == "__main__":
    raise SystemExit(main())
