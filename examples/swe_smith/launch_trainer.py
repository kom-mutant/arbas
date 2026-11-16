"""Log the gateway and supervise the Linux trainer and its child processes."""

import ctypes
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TextIO


def stop_children(children: Path) -> None:
    # Orphaned descendants become our children, even if they called setsid().
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
    key = os.environ["AGL_KEY"]
    last_progress = None
    for line in stream:
        if "AglRolloutManager: completed=" in line:
            if line == last_progress:
                continue
            last_progress = line
        line = line.replace(key, "[redacted]")
        log.write(line)
        print(line, end="", flush=True)


def run_logged_command(args: list[str], log_path: str) -> int:
    def stop_command(signum, _frame) -> None:
        raise SystemExit(128 + signum)

    with (
        open(log_path, "a", encoding="utf-8", buffering=1) as log,
        subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace") as process,
    ):
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, stop_command)
        try:
            copy_output(process.stdout, log)
            status = process.wait()
            return status if status >= 0 else 128 - status
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()


def run_trainer(args: list[str], log_path: str) -> int:
    children = Path("/proc/self/task") / str(threading.get_native_id()) / "children"
    children.read_text()  # Check that the kernel exposes child PIDs before starting training.
    libc = ctypes.CDLL(None, use_errno=True)
    PR_SET_CHILD_SUBREAPER = 36
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        raise SystemExit(f"Cannot supervise trainer processes: {os.strerror(ctypes.get_errno())}")

    stop_signal = None

    def request_stop(signum, _frame):
        nonlocal stop_signal
        stop_signal = signum

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, request_stop)

    with open(log_path, "a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            [sys.executable, "-u", *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            start_new_session=True,
        )
        log_errors = []

        def read_trainer_output():
            try:
                copy_output(process.stdout, log)
            except OSError as exc:
                log_errors.append(exc)

        logger = threading.Thread(target=read_trainer_output, daemon=True)
        try:
            logger.start()
            while process.poll() is None and stop_signal is None and not log_errors:
                time.sleep(0.1)
            if stop_signal is not None:
                process.send_signal(stop_signal)
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=5)
        finally:
            stop_children(children)

        logger.join(timeout=5)
        if logger.is_alive():
            raise RuntimeError("Trainer output did not close after process cleanup.")
        process.stdout.close()
        if log_errors:
            raise log_errors[0]

    if stop_signal is not None:
        return 128 + stop_signal
    return process.returncode if process.returncode >= 0 else 128 - process.returncode


def check_smoke(path: str) -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(path, size_guidance={"scalars": 0}).Reload()
    steps = defaultdict(dict)
    for tag in events.Tags()["scalars"]:
        for event in events.Scalars(tag):
            steps[event.step][tag] = event.value
    steps = {step: metrics for step, metrics in steps.items() if metrics.get("training/n_rollouts", 0) > 0}
    if not steps:
        raise SystemExit(f"ERROR: trainer exited without a GRPO step in TensorBoard. See {path}")
    for step, metrics in steps.items():
        if (
            metrics.get("training/n_rollouts_w_reward") != metrics["training/n_rollouts"]
            or metrics.get("training/n_sample_trained", 0) <= 0
            or any(f"timing_s/{stage}" not in metrics for stage in ("update_actor", "update_weights"))
        ):
            raise SystemExit(f"ERROR: incomplete GRPO step {step}. See {path}")
        groups = int(metrics.get("training/n_groups", 0))
        nonzero = groups - int(metrics.get("training/n_zero_adv_groups", groups))
        print(f"GRPO step {step}: nonzero advantage groups {nonzero}/{groups}. TensorBoard: {path}")


def main() -> int:
    if sys.argv[1] == "--log-command":
        _, log_path, *command = sys.argv[1:]
        return run_logged_command(command, log_path)

    log_path, smoke_tmp, *trainer_args = sys.argv[1:]
    try:
        with TemporaryDirectory(prefix="agl-ipc-", dir="/tmp") as ipc_dir:
            os.environ["VERL_WEIGHT_TRANSFER_IPC_DIR"] = ipc_dir
            status = run_trainer(trainer_args, log_path)
        if status == 0 and smoke_tmp:
            check_smoke(os.environ["TENSORBOARD_DIR"])
        return status
    finally:
        if smoke_tmp:
            shutil.rmtree(smoke_tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
