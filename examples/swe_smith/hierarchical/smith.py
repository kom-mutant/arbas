from __future__ import annotations

import http.client
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

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
    r"(?:curl|wget|httpie|http|https|aria2c|scp|sftp|rsync|nc|ncat|netcat|telnet)"
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


def render_observation(returncode: int, output: str, obs_cap: int) -> str:
    """Format a command result the way mini-swe-agent does.

    Output shorter than ``obs_cap`` is shown verbatim; longer output is elided to
    a head+tail window with a warning so one noisy command cannot blow up context.
    """
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
    """Return a rejection reason if the action cheats instead of solving the bug.

    Blocks four reward-hacking routes so the agent must fix the bug from the
    local /testbed source only:
      - git (relocate_git() moved the repo; history holds the fix + deleted tests)
      - network fetches (curl/wget/python-http) that download the upstream code
      - package installs (pip/conda) that pull the target package's correct source
      - writing test-harness files (conftest/pytest.ini/sitecustomize) to force PASS
    Returns None for allowed actions. Network/install/tamper blocks are a code
    backstop; the authoritative fix is a default-deny egress NetworkPolicy.
    """
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
    """Environment for agent shell commands.

    Hides the relocated git directory from the agent: drops SMITH_HIDDEN_GIT_DIR
    and any variable whose value leaks the path (so ``env`` reveals nothing about
    where .git went), then applies the quiet-pager/progress overrides.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "SMITH_HIDDEN_GIT_DIR",
            "AGL_KEY",
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
            "ORCHESTRATOR_API_KEY",
            "SUBAGENT_API_KEY",
            "JUDGE_API_KEY",
            "AGL_EVAL_META",
            "AGL_TASK_INPUT",
        }
        and _HIDDEN_GIT_DIR not in value
    }
    env.update(_CMD_ENV)
    return env


def _run(command: str, timeout: int) -> tuple[str, int]:
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
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
        output = stdout + stderr
        return (output + f"\n[timed out after {timeout}s]", 124) if timed_out else (output, proc.returncode)
    except Exception as exc:
        return f"[failed to start: {exc}]", 1


def post_event(event_type: str, data: dict[str, Any], retry: bool = False) -> None:
    event_url = os.environ.get("AGL_EVENT_URL")
    if not event_url:
        log.warning("AGL_EVENT_URL not set; skip %s event", event_type)
        return
    body = json.dumps({"event_type": event_type, "data": data}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    agl_key = os.environ.get("AGL_KEY") or os.environ.get("OPENAI_API_KEY")
    if agl_key:
        headers["Authorization"] = f"Bearer {agl_key}"
    request = urllib.request.Request(event_url, data=body, headers=headers, method="POST")
    attempts = 4 if retry else 1
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                response.read()
            return
        except (urllib.error.URLError, TimeoutError) as exc:
            fatal_http = isinstance(exc, urllib.error.HTTPError) and 400 <= exc.code < 500 and exc.code != 429
            if fatal_http or attempt == attempts:
                log.warning("failed to post %s event: %s", event_type, exc)
                return
            backoff = 2 ** attempt
            log.warning(
                "failed to post %s event (attempt %d): %s; retrying in %ds",
                event_type,
                attempt,
                exc,
                backoff,
            )
            time.sleep(backoff)


def fetch_eval_meta() -> dict[str, Any]:
    event_url = os.environ.get("AGL_EVENT_URL", "")
    match = re.match(r"^(?P<base>.*)/api/rollouts/(?P<rid>[^/]+)/attempt/", event_url)
    if not match:
        log.error("cannot derive rollout id from AGL_EVENT_URL=%r; eval meta unavailable", event_url)
        return {}
    url = f"{match.group('base')}/api/rollouts/{match.group('rid')}"
    headers = {}
    agl_key = os.environ.get("AGL_KEY") or os.environ.get("OPENAI_API_KEY")
    if agl_key:
        headers["Authorization"] = f"Bearer {agl_key}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    payload = None
    attempts = 4
    for i in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.HTTPException,
            json.JSONDecodeError,
        ) as exc:
            log.warning("fetch eval meta failed (attempt %d/%d): %s", i + 1, attempts, exc)
            if i < attempts - 1:
                time.sleep(min(2**i, 8))
    if payload is None:
        log.error("failed to fetch eval meta from server after %d attempts", attempts)
        return {}
    inp = (payload.get("rollout") or {}).get("input") or {}
    return {
        "instance_id": inp.get("instance_id", ""),
        "repo": inp.get("repo", ""),
        "FAIL_TO_PASS": inp.get("FAIL_TO_PASS", []),
        "PASS_TO_PASS": inp.get("PASS_TO_PASS", []),
    }


def _git_dir() -> str:
    """Path to the live git directory (relocated once the agent loop is set up)."""
    return _HIDDEN_GIT_DIR if os.path.isdir(_HIDDEN_GIT_DIR) else os.path.join(TESTBED, ".git")


def _git_base() -> list[str]:
    """Base argv for harness git calls.

    Before relocation .git is in-tree, so a bare ``git`` (cwd=/testbed) works.
    After relocate_git() moves it out of the worktree, every harness git call
    must target it explicitly with --git-dir/--work-tree; safe.directory avoids
    git's dubious-ownership refusal now that git-dir and work-tree differ.
    """
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


def _proc_output(proc: subprocess.CompletedProcess) -> str:
    return "\n".join(part for part in (proc.stdout, proc.stderr) if part).strip()


def _git_retry(args: list[str], *, timeout: int = 120, attempts: int = 3) -> subprocess.CompletedProcess | None:
    """Run a git command in /testbed with retries on timeout/failure.

    Under high pod concurrency the node is heavily oversubscribed and IO-bound
    git operations intermittently time out; a few retries with backoff clears
    the transient failures. Returns the CompletedProcess of the last attempt
    (caller checks returncode), or None if every attempt raised.
    """
    last: subprocess.CompletedProcess | None = None
    for i in range(attempts):
        _remove_stale_git_index_lock()
        try:
            proc = subprocess.run([*_git_base(), *args], cwd=TESTBED, capture_output=True, text=True, timeout=timeout)
            if proc.returncode == 0:
                return proc
            last = proc
            log.warning(
                "git %s rc=%d (attempt %d/%d): %s", args[0], proc.returncode, i + 1, attempts, _proc_output(proc)[:200]
            )
        except subprocess.TimeoutExpired:
            last = None
            _remove_stale_git_index_lock()
            log.warning("git %s timed out after %ds (attempt %d/%d)", args[0], timeout, i + 1, attempts)
        if i < attempts - 1:
            time.sleep(min(2**i, 8))
    return last


def checkout_bug_commit(instance_id: str) -> None:
    """Bring the injected bug into /testbed using SWE-smith's official checkout.

    SWE-smith builds each instance as a branch `<instance_id>` of two
    commits on top of `main`: `Bug Patch` (injects the bug) then `Remove F2P
    Tests` (deletes the FAIL_TO_PASS test files). The image defaults to clean
    `main`, so without this the agent sees fixed code and evaluate() false-passes.
    """
    proc = _git_retry(["checkout", instance_id], timeout=120)
    if proc is None or proc.returncode != 0:
        reason = "timeout" if proc is None else _proc_output(proc)
        raise SystemExit(f"git checkout {instance_id} failed: {reason}")
    log.info("checked out SWE-smith bug branch in /testbed: %s", instance_id)


def relocate_git() -> None:
    """Move /testbed/.git out of the agent's working tree (call after checkout).

    This closes the reward-hacking leak where the agent recovers the injected-bug
    fix (and the deleted FAIL_TO_PASS tests) from git history via
    `git checkout <pre-bug-sha> -- <src>` / `git show`/`git log -p`. Once .git is
    gone from /testbed the worktree is no longer a repo, so the agent's git calls
    fail; the harness still reaches history through _git_base() (--git-dir). A
    same-filesystem move is an O(1) rename (no blob IO); a cross-fs move falls
    back to a copy. The agent is also forbidden from invoking git or reading the
    relocated directory (see _forbidden_action).
    """
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
    """Restore the test files deleted by `Remove F2P Tests`, keeping the agent's fix.

    The branch HEAD has the FAIL_TO_PASS test files deleted; the parent commit
    (`HEAD~1`, the `Bug Patch` commit) still has them. We restore
    just those test paths from `~1` into the working tree -- this overwrites only
    the test files, leaving the agent's source edits intact -- so evaluate() can
    run the real FAIL_TO_PASS/PASS_TO_PASS suite against the agent's fix.
    """
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
        m = _STATUS_RE.search(line)
        if not m:
            continue
        if m.group(1):
            status, node = m.group(1), m.group(2)
        else:
            node, status = m.group(3), m.group(4)
        statuses[node] = status
    return statuses


def evaluate(eval_meta: dict[str, Any], timeout: int, f2p_only: bool = True) -> tuple[float, bool, str, bool]:
    fail_to_pass = list(eval_meta.get("FAIL_TO_PASS", []))
    pass_to_pass = list(eval_meta.get("PASS_TO_PASS", []))
    if not fail_to_pass:
        return 0.0, False, "no FAIL_TO_PASS in eval meta", False

    # f2p_only (SWE-smith --f2p_only): grade only on the test FILES holding the
    # F2P tests. F2P kept whole; P2P filtered to those same files (dropping the
    # unrelated cross-file P2P that cause eval timeouts). Resolve = all F2P + P2P pass.
    if f2p_only:
        f2p_files = sorted({t.split("::", 1)[0] for t in fail_to_pass})
        pass_to_pass = [t for t in pass_to_pass if t.split("::", 1)[0] in f2p_files]

    nodes = fail_to_pass + pass_to_pass
    restore_f2p_tests(nodes)
    # Cap pytest at 4 workers: the project's own `-n auto` reads HOST cores (dozens),
    # ignores the pod cgroup, and spawns ~70 workers that OOMKill the 4Gi pod. `-n`
    # needs xdist, so probe first and fall back to serial `-p no:xdist` when absent.
    python = _shq(sys.executable)
    xdist_flag = "-n4" if _run(f"{python} -I -c 'import xdist'", 30)[1] == 0 else "-p no:xdist"
    output, rc = _run(
        f"{python} -I -c {_shq(_PYTEST_RUNNER)} {_shq(TESTBED)} --color=no -rA -p no:cacheprovider "
        + xdist_flag
        + " "
        + " ".join(map(_shq, nodes)),
        timeout,
    )
    # rc 124 == subprocess.TimeoutExpired (see _run); pytest itself never exits 124.
    timed_out = rc == 124
    statuses = parse_test_statuses(output)
    f2p_pass = [t for t in fail_to_pass if statuses.get(t) in ("PASSED", "XFAIL")]
    # only count P2P reporting PASSED/XFAIL — a missing status means it never ran.
    p2p_ok = [t for t in pass_to_pass if statuses.get(t) in ("PASSED", "XFAIL")]
    resolved = rc == 0 and len(f2p_pass) == len(fail_to_pass) and len(p2p_ok) == len(pass_to_pass)
    prefix = f"EVAL TIMEOUT after {timeout}s — " if timed_out else ""
    suffix = " (f2p_only)" if f2p_only else ""
    reason = (
        f"{prefix}FAIL_TO_PASS {len(f2p_pass)}/{len(fail_to_pass)} passed, "
        f"PASS_TO_PASS {len(p2p_ok)}/{len(pass_to_pass)} ok{suffix}"
    )
    return (1.0 if resolved else 0.0), resolved, reason, timed_out


def _shq(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def length_penalized_reward(
    reward: float, n_turns: int, max_turns: int, *, t0: int, lam: float, is_train: bool
) -> float:
    """Apply the long-turn penalty (plan A) to a SOLVED *training* trajectory's reward.

    The penalty applies **only** when ``is_train`` is True and ``reward >= 1.0``
    (solved). This keeps VALIDATION reward the true, unshaped metric (validation
    drives checkpoint selection, so it must never be reshaped), and leaves
    unsolved rollouts full exploration room. The penalty is a linear soft ramp
    over the turn budget::

        reward = 1.0 - lam * clip((n_turns - t0) / (max_turns - t0), 0, 1)

    A train solve within ``t0`` turns keeps reward ``1.0``; one that runs to the
    cap gets ``1 - lam``. Validation rewards and non-solved rewards are returned
    unchanged.
    """
    if not is_train or reward < 1.0 or n_turns <= t0:
        return reward
    span = max(1, max_turns - t0)
    frac = min((n_turns - t0) / span, 1.0)
    return 1.0 - lam * frac


def prompt_length_penalty(
    reward: float,
    max_prompt_tokens: int,
    *,
    soft_start: int,
    hard_cap: int,
    max_pen: float,
    is_train: bool,
    solved: bool,
) -> float:
    """Penalizes context bloat: the longest single-turn prompt of a rollout is its
    true upper bound on context pressure (each turn's prompt embeds all prior
    history). Gated identically to the turn penalty — only ``is_train`` and
    ``solved`` — so VALIDATION reward stays the unshaped metric and unsolved
    rollouts are untouched. ``solved`` is the *raw* solved status (not the running
    reward), so this composes with the turn penalty even after it lowered the
    reward below 1.0. Soft linear ramp over the context band::

        penalty = max_pen * clip((max_prompt_tokens - soft_start) /
                                 (hard_cap - soft_start), 0, 1)
        reward -= penalty

    A solve whose longest prompt is <= ``soft_start`` keeps its reward; at or
    above ``hard_cap`` it loses the full ``max_pen``. Stacks with (subtracts on
    top of) the turn penalty.
    """
    if not is_train or not solved or max_prompt_tokens <= soft_start:
        return reward
    if max_prompt_tokens >= hard_cap:
        return reward - max_pen
    span = max(1, hard_cap - soft_start)
    frac = min((max_prompt_tokens - soft_start) / span, 1.0)
    return reward - max_pen * frac
