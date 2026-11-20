from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, NamedTuple

log = logging.getLogger(__name__)

_PYTEST_RUNNER = """import shlex
import sys
import pytest

# Load installed pytest before making project modules importable.
sys.path.insert(0, sys.argv.pop(1))

@pytest.hookimpl(optionalhook=True)
def pytest_xdist_setupnodes(specs):
    for spec in specs:
        spec.python = shlex.quote(sys.executable) + " -I"

raise SystemExit(pytest.main(sys.argv[1:], plugins=[sys.modules[__name__]]))
"""

TESTBED = "/testbed"

_HIDDEN_GIT_DIR = os.environ.get("SMITH_HIDDEN_GIT_DIR", "/opt/agl_tmp")

_STATUS_RE = re.compile(
    r"(?:^|\s)(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+)|(\S+)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)(?:\s|$)"
)

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

_GIT_INVOKE_RE = re.compile(
    r"(?:^|[\n;`(]|&&|\|\|?|\$\()\s*(?:\w+=\S+\s+)*(?:[\w./-]*/)?git(?:-[a-z]+)?(?=\s|$|;|&|\|)",
    re.I,
)

_GIT_ACCESS_RE = re.compile(r'--git-dir|--work-tree|(?:^|[\s=:"\'/])\.git(?:/|\b)', re.I)

_NET_FETCH_RE = re.compile(
    r"(?:^|[\n;`(]|&&|\|\|?|\$\()\s*(?:\w+=\S+\s+)*(?:[\w./-]*/)?"
    r"(?:curl|wget|httpie|(?-i:http|https)|aria2c|scp|sftp|rsync|nc|ncat|netcat|telnet)"
    r"(?=\s|$|;|&|\|)",
    re.I,
)

_PKG_INSTALL_RE = re.compile(
    r"(?:^|[\n;`(]|&&|\|\|?|\$\()\s*(?:\w+=\S+\s+)*(?:"
    r"(?:[\w./-]*/)?(?:pip|pip3|conda|mamba|easy_install|uv)\b[^\n;&|]*?\binstall\b"
    r"|(?:[\w./-]*/)?python[0-9.]*\s+-m\s+pip\b[^\n;&|]*?\binstall\b)",
    re.I,
)

_PY_NET_RE = re.compile(
    r"urllib\.request|\burlopen\b|\brequests\.(?:get|post|put|head|Session)\b|"
    r"\bhttpx\.|\bsocket\.(?:socket|create_connection)\b|\burllib3\b",
    re.I,
)

_TEST_TAMPER_RE = re.compile(
    r'(?:>>?|\btee\b(?:\s+-a)?\s+)\s*[\'"]?[^\s\'"|;&<>]*'
    r"(?:conftest\.py|pytest\.ini|tox\.ini|sitecustomize\.py|usercustomize\.py|"
    r'setup\.cfg|pyproject\.toml|\.pth)(?=[\'"\s;&|]|$)',
    re.I,
)

_CMD_ENV = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}

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


def render_observation(returncode: int, output: str, obs_cap: int) -> str:
    """Format a command result, eliding overlong outputs to a head+tail window."""
    if len(output) < obs_cap:
        return f"<returncode>{returncode}</returncode>\n<output>\n{output}</output>"
    half = obs_cap // 2
    elided = len(output) - obs_cap
    return (
        f"<returncode>{returncode}</returncode>\n"
        f"<warning>\n{_OVERLONG_WARNING}\n</warning>\n"
        f"<output_head>\n{output[:half]}\n</output_head>\n"
        f"<elided_chars>\n{elided} characters elided\n</elided_chars>\n"
        f"<output_tail>\n{output[-half:]}\n</output_tail>"
    )


def _forbidden_action(action: str) -> str | None:
    """Return a rejection reason if the action tries to bypass /testbed constraints."""
    if _GIT_INVOKE_RE.search(action):
        return (
            "git is disabled in this environment; do not use it for any "
            "purpose. Inspect and edit the source files under /testbed "
            "directly (cat, grep, sed, python) to fix the bug."
        )
    if _GIT_ACCESS_RE.search(action) or _HIDDEN_GIT_DIR in action:
        return (
            "Accessing the git metadata directory is not allowed. Work only "
            "with the source files under /testbed; do not read .git."
        )
    if _NET_FETCH_RE.search(action) or _PY_NET_RE.search(action):
        return (
            "Network access is disabled. Do not fetch code from the internet "
            "(curl, wget, urllib, requests, etc.); all dependencies are "
            "already installed. Solve the bug using only the source files "
            "already present under /testbed."
        )
    if _PKG_INSTALL_RE.search(action):
        return (
            "Installing packages is not allowed. Everything needed to run the "
            "code and its tests is already installed. Fix the bug by editing "
            "the source under /testbed; do not install anything."
        )
    if _TEST_TAMPER_RE.search(action):
        return (
            "Modifying test-harness or config files (conftest.py, pytest.ini, "
            "tox.ini, setup.cfg, pyproject.toml, sitecustomize.py, .pth) is "
            "not allowed. Fix the bug in the source under /testbed instead."
        )
    return None


def _agent_env() -> dict[str, str]:
    """Environment for agent shell commands with secrets and hidden git path stripped."""
    hidden = {
        "SMITH_HIDDEN_GIT_DIR",
        "AGL_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "POLZA_API_KEY",
        "ORCHESTRATOR_API_KEY",
        "SUBAGENT_API_KEY",
        "JUDGE_API_KEY",
        "AGL_EVAL_META",
        "AGL_TASK_INPUT",
    }
    env = {key: value for key, value in os.environ.items() if key not in hidden and _HIDDEN_GIT_DIR not in value}
    env.update(_CMD_ENV)
    return env


def _run(command: str, timeout: float) -> tuple[str, int]:
    try:
        proc = subprocess.Popen(
            ["bash", "-c", command],
            cwd=TESTBED,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_agent_env(),
            start_new_session=True,
        )
        timed_out = False
        try:
            proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
        output = (stdout or "") + (stderr or "")
        return (output + f"\n[timed out after {timeout}s]", 124) if timed_out else (output, proc.returncode)
    except Exception as exc:
        return f"[failed to start: {exc}]", 1


def _git_dir() -> str:
    return _HIDDEN_GIT_DIR if os.path.isdir(_HIDDEN_GIT_DIR) else os.path.join(TESTBED, ".git")


def _git_base() -> list[str]:
    if os.path.isdir(_HIDDEN_GIT_DIR):
        return ["git", "--git-dir", _HIDDEN_GIT_DIR, "--work-tree", TESTBED, "-c", "safe.directory=*"]
    return ["git"]


def _remove_stale_git_index_lock() -> None:
    lock_path = os.path.join(_git_dir(), "index.lock")
    try:
        os.unlink(lock_path)
        log.warning("removed stale git index lock: %s", lock_path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        log.warning("failed to remove stale git index lock %s: %s", lock_path, exc)


def _proc_output(proc: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(part for part in (proc.stdout, proc.stderr) if part).strip()


def _git_retry(
    args: list[str],
    *,
    timeout: int = 120,
    attempts: int = 3,
) -> subprocess.CompletedProcess[str] | None:
    last: subprocess.CompletedProcess[str] | None = None
    for i in range(attempts):
        _remove_stale_git_index_lock()
        try:
            proc = subprocess.run([*_git_base(), *args], cwd=TESTBED, capture_output=True, text=True, timeout=timeout)
            if proc.returncode == 0:
                return proc
            last = proc
            log.warning(
                "git %s rc=%d (attempt %d/%d): %s",
                args[0],
                proc.returncode,
                i + 1,
                attempts,
                _proc_output(proc)[:200],
            )
        except subprocess.TimeoutExpired:
            last = None
            _remove_stale_git_index_lock()
            log.warning("git %s timed out after %ds (attempt %d/%d)", args[0], timeout, i + 1, attempts)
        if i < attempts - 1:
            time.sleep(min(2**i, 8))
    return last


def checkout_bug_commit(instance_id: str) -> None:
    """Checkout the SWE-smith instance branch (<instance_id>) in /testbed."""
    proc = _git_retry(["checkout", instance_id], timeout=120)
    if proc is None or proc.returncode != 0:
        reason = "timeout" if proc is None else _proc_output(proc)
        raise SystemExit(f"git checkout {instance_id} failed: {reason}")
    log.info("checked out SWE-smith bug branch in /testbed: %s", instance_id)


def relocate_git() -> None:
    """Move /testbed/.git out of the agent's working tree after checkout."""
    src = os.path.join(TESTBED, ".git")
    if not os.path.isdir(src):
        log.warning("relocate_git: %s missing; git-history leak NOT closed", src)
        return
    if os.path.exists(_HIDDEN_GIT_DIR):
        shutil.rmtree(_HIDDEN_GIT_DIR, ignore_errors=True)
    parent = os.path.dirname(_HIDDEN_GIT_DIR) or "/"
    os.makedirs(parent, exist_ok=True)
    shutil.move(src, _HIDDEN_GIT_DIR)
    log.info("relocated .git -> %s", _HIDDEN_GIT_DIR)


def restore_f2p_tests(test_nodes: list[str]) -> None:
    """Restore FAIL_TO_PASS test files from HEAD~1 while keeping the agent's source edits."""
    paths = sorted({node.split("::", 1)[0] for node in test_nodes if node})
    if not paths:
        return
    proc = _git_retry(["checkout", "HEAD~1", "--", *paths], timeout=120)
    if proc is None or proc.returncode != 0:
        log.warning("restore_f2p_tests failed: %s", getattr(proc, "stderr", "timeout"))


def capture_patch() -> str:
    try:
        subprocess.run(
            [*_git_base(), "add", "--intent-to-add", "--", "."],
            cwd=TESTBED,
            capture_output=True,
            check=True,
            timeout=60,
        )
        proc = subprocess.run(
            [*_git_base(), "-c", "core.fileMode=false", "diff", "HEAD"],
            cwd=TESTBED,
            capture_output=True,
            check=True,
            text=True,
            timeout=60,
        )
        return proc.stdout
    except Exception as exc:
        log.error("patch capture failed: %s", exc)
        return ""


def parse_test_statuses(test_output: str) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for line in _ANSI_ESCAPE_RE.sub("", test_output).splitlines():
        match = _STATUS_RE.search(line)
        if not match:
            continue
        if match.group(1):
            status, node = match.group(1), match.group(2)
        else:
            node, status = match.group(3), match.group(4)
        statuses[node] = status
    return statuses


def _shq(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def evaluate(eval_meta: dict[str, Any], timeout: int, f2p_only: bool = True) -> Evaluation:
    fail_to_pass = list(eval_meta.get("FAIL_TO_PASS", []))
    pass_to_pass = list(eval_meta.get("PASS_TO_PASS", []))
    if not fail_to_pass:
        return Evaluation(0.0, False, "no FAIL_TO_PASS in eval meta", False)

    if f2p_only:
        f2p_files = sorted({test.split("::", 1)[0] for test in fail_to_pass})
        pass_to_pass = [test for test in pass_to_pass if test.split("::", 1)[0] in f2p_files]

    nodes = fail_to_pass + pass_to_pass
    restore_f2p_tests(nodes)
    # Cap xdist workers at 4 to avoid OOM in resource-limited pods.
    python = _shq(sys.executable)
    xdist_flag = "-n4" if _run(f"{python} -I -c 'import xdist'", 30)[1] == 0 else "-p no:xdist"
    output, rc = _run(
        f"{python} -I -c {_shq(_PYTEST_RUNNER)} {_shq(TESTBED)} --color=no -rA -p no:cacheprovider "
        + xdist_flag
        + " "
        + " ".join(map(_shq, nodes)),
        timeout,
    )
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
