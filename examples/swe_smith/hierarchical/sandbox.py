"""Docker sandbox holding the task repository that subagents edit and hidden tests grade.

The agent runs on the host; ``ContainerSandbox`` starts a stock SWE-smith image with
``--network=none``, CPU/RAM limits, and no secrets, executing commands via ``docker exec``.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple

from .config import normalize_memory_limit

RUN_LABEL = "arbas.run"
TESTBED = "/testbed"

_CONTAINER_GIT_DIR = "/tmp/.arbas-base.git"
_CMD_ENV = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}
_PROTECTED_NAMES = frozenset(
    {"conftest.py", "pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml", "sitecustomize.py", "usercustomize.py"}
)

_PYTEST_RUNNER = """import shlex
import sys
import pytest

# Load installed pytest before making project modules importable.
sys.path.insert(0, sys.argv.pop(1))

@pytest.hookimpl(optionalhook=True)
def pytest_xdist_setupnodes(specs):
    for spec in specs:
        spec.python = shlex.quote(sys.executable) + " -I -B"

raise SystemExit(pytest.main(sys.argv[1:], plugins=[sys.modules[__name__]]))
"""

_STATUS_RE = re.compile(
    r"(?:^|\s)(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+)|(\S+)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)(?:\s|$)"
)
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

_OVERLONG_WARNING = (
    "The output of your last command was too long.\n"
    "Please try a different command that produces less output.\n"
    "If you're looking at a file you can use head, tail or sed to view a smaller "
    "number of lines selectively.\n"
    "If you're using grep or find and it produced too much output, use a more "
    "selective search pattern.\n"
    "If you really need the full output, redirect it to a file and search within it."
)


class Evaluation(NamedTuple):
    reward: float
    resolved: bool
    reason: str
    timed_out: bool
    partial_reward: float = 0.0


_REAPER_SCRIPT = (
    "cat >/dev/null; "
    'err=$(docker rm --force "$1" 2>&1 >/dev/null) && exit 0; '
    'case "$err" in *"No such container"*|*"already in progress"*) exit 0;; esac; '
    'printf "%s\\n" "$err" >&2; exit 1'
)


def _is_benign_rm_race(stderr: str) -> bool:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    return bool(lines) and all("No such container" in line or "already in progress" in line for line in lines)


def remove_run_containers(run_dir: Path) -> None:
    """Remove every sandbox container started for ``run_dir``."""
    listed = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"label={RUN_LABEL}={run_dir.resolve()}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if containers := listed.stdout.split():
        removed = subprocess.run(
            ["docker", "rm", "--force", *containers],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if removed.returncode != 0 and not _is_benign_rm_race(removed.stderr):
            raise subprocess.CalledProcessError(removed.returncode, removed.args, removed.stdout, removed.stderr)


def render_observation(returncode: int, output: str, obs_cap: int) -> str:
    """Format a command result, eliding overlong outputs to a head+tail window."""
    if len(output) < obs_cap:
        return f"<returncode>{returncode}</returncode>\n<output>\n{output}</output>"
    half = obs_cap // 2
    return (
        f"<returncode>{returncode}</returncode>\n"
        f"<warning>\n{_OVERLONG_WARNING}\n</warning>\n"
        f"<output_head>\n{output[:half]}\n</output_head>\n"
        f"<elided_chars>\n{len(output) - obs_cap} characters elided\n</elided_chars>\n"
        f"<output_tail>\n{output[-half:]}\n</output_tail>"
    )


def parse_test_statuses(test_output: str) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for line in _ANSI_ESCAPE_RE.sub("", test_output).splitlines():
        match = _STATUS_RE.search(line)
        if not match:
            continue
        if match.group(1):
            statuses[match.group(2)] = match.group(1)
        else:
            statuses[match.group(3)] = match.group(4)
    return statuses


def select_eval_nodes(eval_meta: dict[str, Any], *, f2p_only: bool) -> tuple[list[str], list[str]]:
    """Return ``(FAIL_TO_PASS, PASS_TO_PASS)``; ``f2p_only`` keeps only PASS_TO_PASS in FAIL_TO_PASS files."""
    fail_to_pass = list(eval_meta.get("FAIL_TO_PASS", []))
    pass_to_pass = list(eval_meta.get("PASS_TO_PASS", []))
    if f2p_only:
        files = {node.split("::", 1)[0] for node in fail_to_pass}
        pass_to_pass = [node for node in pass_to_pass if node.split("::", 1)[0] in files]
    return fail_to_pass, pass_to_pass


def _grade(
    output: str, rc: int, fail_to_pass: list[str], pass_to_pass: list[str], timeout: int, f2p_only: bool
) -> Evaluation:
    timed_out = rc == 124
    statuses = parse_test_statuses(output)
    f2p_pass = sum(statuses.get(test) in ("PASSED", "XFAIL") for test in fail_to_pass)
    p2p_ok = sum(statuses.get(test) in ("PASSED", "XFAIL") for test in pass_to_pass)
    resolved = rc == 0 and f2p_pass == len(fail_to_pass) and p2p_ok == len(pass_to_pass)
    partial_reward = float(resolved)
    if rc == 1 and p2p_ok == len(pass_to_pass):
        f2p_failed = sum(statuses.get(test) == "FAILED" for test in fail_to_pass)
        if f2p_failed and f2p_pass + f2p_failed == len(fail_to_pass):
            partial_reward = f2p_pass / len(fail_to_pass)
    prefix = f"EVAL TIMEOUT after {timeout}s — " if timed_out else ""
    suffix = " (f2p_only)" if f2p_only else ""
    reason = (
        f"{prefix}FAIL_TO_PASS {f2p_pass}/{len(fail_to_pass)} passed, "
        f"PASS_TO_PASS {p2p_ok}/{len(pass_to_pass)} ok{suffix}"
    )
    return Evaluation(float(resolved), resolved, reason, timed_out, partial_reward)


def _is_protected(path: str) -> bool:
    return PurePosixPath(path).name in _PROTECTED_NAMES or path.endswith(".pth")


def _format_cpus(cpus: float) -> str:
    return str(int(cpus)) if cpus.is_integer() else str(cpus)


class ContainerSandbox:
    """Throwaway SWE-smith container: ``--network=none``, CPU/RAM capped, no secrets or bind mounts."""

    root = TESTBED
    git_dir = _CONTAINER_GIT_DIR
    python = "python"
    shell: tuple[str, ...] = ("bash", "-lc")  # login shell activates the image's conda env

    def __init__(
        self,
        image: str,
        name: str,
        *,
        run_dir: Path,
        cpus: float = 1.0,
        memory: str = "6g",
        ttl_seconds: int = 7200,
    ) -> None:
        if cpus <= 0:
            raise ValueError("cpus must be positive")
        self.image = image
        self.name = name
        self.run_dir = run_dir
        self.cpus = float(cpus)
        self.memory = normalize_memory_limit(memory)
        self.ttl_seconds = ttl_seconds
        self._tests = b""
        self._configs = b""
        self._reaper: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> ContainerSandbox:
        # Start the reaper in its own session BEFORE `docker run` so even an immediate
        # SIGKILL from LocalReconciler removes the container right away.
        reaper = subprocess.Popen(
            ["sh", "-c", _REAPER_SCRIPT, "arbas-reap", self.name],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            subprocess.run(
                [
                    "docker", "run", "--detach", "--rm", "--init", "--pull=never", "--network=none",
                    "--cpus", _format_cpus(self.cpus),
                    "--memory", self.memory,
                    "--memory-swap", self.memory,
                    "--name", self.name,
                    "--label", f"{RUN_LABEL}={self.run_dir.resolve()}",
                    self.image, "sleep", str(self.ttl_seconds),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )  # fmt: skip
        except BaseException:
            self._close_reaper(reaper)
            raise
        self._reaper = reaper
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        reaper = self._reaper
        self._reaper = None
        if reaper is not None:
            status = self._close_reaper(reaper)
            if status and exc_type is None:
                raise RuntimeError(f"Failed to remove Docker container {self.name}: exit {status}")

    @staticmethod
    def _close_reaper(reaper: subprocess.Popen[bytes]) -> int:
        assert reaper.stdin is not None
        reaper.stdin.close()
        try:
            return reaper.wait(timeout=30)
        except subprocess.TimeoutExpired:
            reaper.kill()
            reaper.wait()
            return 124

    def _exec(
        self, argv: list[str], *, stdin: bytes | None = None, timeout: float
    ) -> subprocess.CompletedProcess[bytes]:
        env = [arg for key, value in _CMD_ENV.items() for arg in ("-e", f"{key}={value}")]
        command = [
            "docker", "exec", *(["-i"] if stdin is not None else []), "-w", self.root, *env, self.name,
            "timeout", "--kill-after=5", str(timeout), *argv,
        ]  # fmt: skip
        return subprocess.run(
            command,
            input=stdin,
            stdin=None if stdin is not None else subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout + 15,
        )

    def _run(self, argv: list[str], *, stdin: bytes | None = None, timeout: float = 120) -> bytes:
        proc = self._exec(argv, stdin=stdin, timeout=timeout)
        if proc.returncode != 0:
            detail = (proc.stdout + proc.stderr).decode(errors="replace").strip()
            raise RuntimeError(f"{shlex.join(argv)} failed with exit {proc.returncode}: {detail}")
        return proc.stdout

    def _git(self, *args: str, baseline: bool = False, timeout: float = 120) -> bytes:
        command = ["git", "-c", "safe.directory=*", "-c", "user.name=arbas", "-c", "user.email=arbas@local"]
        if baseline:
            command[1:1] = ["--git-dir", self.git_dir, "--work-tree", self.root]
        return self._run([*command, *args], timeout=timeout)

    def _archive(self, rev: str, paths: list[str]) -> bytes:
        if not paths:
            return b""
        listed = self._git("ls-tree", "-r", "-z", "--name-only", rev, "--", *paths)
        existing = [path for path in listed.decode().split("\0") if path]
        return self._git("archive", "--format=tar", rev, "--", *existing) if existing else b""

    def setup(self, instance_id: str, test_nodes: list[str]) -> None:
        """Check out the task branch, set aside hidden tests in host RAM, and delete ``.git``."""
        self._git("checkout", instance_id)
        test_files = sorted({node.split("::", 1)[0] for node in test_nodes if node})
        self._tests = self._archive("HEAD~1", test_files)
        tracked = self._git("ls-files", "-z").decode().split("\0")
        self._configs = self._archive("HEAD", [path for path in tracked if path and _is_protected(path)])

        self._run(["rm", "-rf", f"{self.root}/.git", self.git_dir])
        self._git("init", "-q", baseline=True)
        self._git("add", "-A", baseline=True)
        self._git("commit", "-q", "-m", "baseline", "--allow-empty", baseline=True)

    def run_bash(self, command: str, timeout: float) -> tuple[str, int]:
        started = time.monotonic()
        try:
            proc = self._exec([*self.shell, command], timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            output = ((exc.stdout or b"") + (exc.stderr or b"")).decode(errors="replace")
            return f"{output}\n[timed out after {timeout}s]", 124
        except OSError as exc:
            return f"[failed to start: {exc}]", 1
        output = (proc.stdout + proc.stderr).decode(errors="replace")
        if proc.returncode == 124 or (proc.returncode == 137 and time.monotonic() - started >= timeout):
            return f"{output}\n[timed out after {timeout}s]", 124
        return output, proc.returncode

    def capture_patch(self) -> str:
        """Return ``git diff`` against the baseline commit, raising if Git fails."""
        self._git("add", "--intent-to-add", "--", ".", baseline=True, timeout=60)
        return self._git("-c", "core.fileMode=false", "diff", "HEAD", baseline=True, timeout=60).decode(
            errors="replace"
        )

    def _restore_harness(self) -> None:
        names = [arg for name in sorted(_PROTECTED_NAMES) for arg in ("-name", name, "-o")]
        self._run(["find", self.root, "-type", "f", "(", *names, "-name", "*.pth", ")", "-delete"], timeout=60)
        for archive in (self._configs, self._tests):
            if archive:
                self._run(["tar", "-xmf", "-", "-C", self.root], stdin=archive, timeout=60)

    def evaluate(self, eval_meta: dict[str, Any], timeout: int, *, f2p_only: bool = True) -> Evaluation:
        fail_to_pass, pass_to_pass = select_eval_nodes(eval_meta, f2p_only=f2p_only)
        if not fail_to_pass:
            return Evaluation(0.0, False, "no FAIL_TO_PASS in eval meta", False)
        self._restore_harness()
        has_xdist = self.run_bash(f"{self.python} -I -B -c 'import xdist'", 30)[1] == 0
        flags = "-n4" if has_xdist else "-p no:xdist"
        nodes = " ".join(shlex.quote(node) for node in fail_to_pass + pass_to_pass)
        output, rc = self.run_bash(
            f"{self.python} -I -B -c {shlex.quote(_PYTEST_RUNNER)} {shlex.quote(self.root)} "
            f"--color=no -rA -p no:cacheprovider {flags} {nodes}",
            timeout,
        )
        return _grade(output, rc, fail_to_pass, pass_to_pass, timeout, f2p_only)
