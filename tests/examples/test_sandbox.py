"""Tests for ``ContainerSandbox``.

Docker CLI assembly and container lifecycle are verified with a fake ``docker`` CLI;
repository setup, hidden-test restoration, patch capture, and pytest grading are exercised
end-to-end on real git repositories via ``LocalTestSandbox`` (which overrides only ``_exec``).
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from omegaconf import OmegaConf

from agentlightning.client import AgentLightningAsyncClient
from agentlightning.controller.local_reconciler import LocalReconciler
from agentlightning.schemas import Rollout, RolloutConfig, RolloutLifecycleStatus, RolloutLocalConfig, RolloutState
from examples.swe_smith.hierarchical import sandbox as sandbox_module
from examples.swe_smith.hierarchical.sandbox import (
    ContainerSandbox,
    Evaluation,
    parse_test_statuses,
    remove_run_containers,
    render_observation,
    select_eval_nodes,
)
from tests.examples.conftest import LocalTestSandbox

# --------------------------------------------------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------------------------------------------------


def test_render_observation_short_output_is_returned_whole():
    assert render_observation(0, "hello\n", obs_cap=100) == "<returncode>0</returncode>\n<output>\nhello\n</output>"


def test_render_observation_elides_the_middle_of_long_output():
    output = "A" * 10 + "M" * 30 + "Z" * 10
    rendered = render_observation(3, output, obs_cap=20)
    assert rendered.startswith("<returncode>3</returncode>\n<warning>\n")
    assert f"<output_head>\n{'A' * 10}\n</output_head>" in rendered
    assert f"<output_tail>\n{'Z' * 10}\n</output_tail>" in rendered
    assert "<elided_chars>\n30 characters elided\n</elided_chars>" in rendered
    assert "M" not in rendered


def test_parse_test_statuses_understands_both_formats_and_strips_ansi():
    output = "\n".join(
        [
            "PASSED a.py::t1",
            "a.py::t2 FAILED",
            "\x1b[32mPASSED\x1b[0m b.py::t3",
            "FAILED b.py::t4 - AssertionError: assert 1 == 2",
            "b.py::t5 \x1b[31mERROR\x1b[0m",
            "SKIPPED [1] b.py:3: not today",
            "=== 2 passed in 0.01s ===",
        ]
    )
    assert parse_test_statuses(output) == {
        "a.py::t1": "PASSED",
        "a.py::t2": "FAILED",
        "b.py::t3": "PASSED",
        "b.py::t4": "FAILED",
        "b.py::t5": "ERROR",
        "[1]": "SKIPPED",
    }


def test_select_eval_nodes_f2p_only_keeps_pass_to_pass_in_fail_to_pass_files():
    meta = {"FAIL_TO_PASS": ["a.py::x"], "PASS_TO_PASS": ["a.py::y", "b.py::z"]}
    assert select_eval_nodes(meta, f2p_only=True) == (["a.py::x"], ["a.py::y"])
    assert select_eval_nodes(meta, f2p_only=False) == (["a.py::x"], ["a.py::y", "b.py::z"])
    assert select_eval_nodes({}, f2p_only=True) == ([], [])


# --------------------------------------------------------------------------------------------------------------------
# Docker: fake subprocess.run
# --------------------------------------------------------------------------------------------------------------------


class FakeRun:
    """Stand-in for ``subprocess.run`` that records calls and replays queued results or exceptions."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.results: list[Any] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        self.calls.append((argv, kwargs))
        result = self.results.pop(0) if self.results else subprocess.CompletedProcess(argv, 0, b"", b"")
        if isinstance(result, BaseException):
            raise result
        return result

    @property
    def argvs(self) -> list[list[str]]:
        return [argv for argv, _ in self.calls]


@pytest.fixture
def fake_run(monkeypatch) -> FakeRun:
    fake = FakeRun()
    monkeypatch.setattr(sandbox_module.subprocess, "run", fake)
    return fake


def test_remove_run_containers_removes_listed_containers_and_ignores_rm_races(fake_run, tmp_path):
    fake_run.results = [
        subprocess.CompletedProcess([], 0, "c1\nc2\n", ""),
        subprocess.CompletedProcess([], 0, "", ""),
    ]
    remove_run_containers(tmp_path)
    assert fake_run.argvs == [
        ["docker", "ps", "-aq", "--filter", f"label=arbas.run={tmp_path.resolve()}"],
        ["docker", "rm", "--force", "c1", "c2"],
    ]

    fake_run.calls.clear()
    fake_run.results = [
        subprocess.CompletedProcess([], 0, "c1\n", ""),
        subprocess.CompletedProcess([], 1, "", "Error response from daemon: No such container: c1\n"),
    ]
    remove_run_containers(tmp_path)

    fake_run.calls.clear()
    fake_run.results = [
        subprocess.CompletedProcess([], 0, "c1\n", ""),
        subprocess.CompletedProcess([], 1, "", "Cannot connect to the Docker daemon\n"),
    ]
    with pytest.raises(subprocess.CalledProcessError):
        remove_run_containers(tmp_path)


def test_remove_run_containers_does_nothing_when_none_listed(fake_run, tmp_path):
    fake_run.results = [subprocess.CompletedProcess([], 0, "\n", "")]
    remove_run_containers(tmp_path)
    assert [argv[1] for argv in fake_run.argvs] == ["ps"]


@pytest.fixture
def fake_reaper(monkeypatch):
    created: list[MagicMock] = []

    def make_popen(*args, **kwargs):
        proc = MagicMock()
        proc.args = args
        proc.kwargs = kwargs
        proc.stdin = MagicMock()
        proc.wait.return_value = 0
        created.append(proc)
        return proc

    monkeypatch.setattr(sandbox_module.subprocess, "Popen", make_popen)
    return created


@pytest.fixture
def container(tmp_path) -> ContainerSandbox:
    return ContainerSandbox("swe/image:latest", "agl-rollout-1-2", run_dir=tmp_path, ttl_seconds=600)


def test_container_start_is_isolated_enforces_limits_and_attaches_reaper(fake_run, fake_reaper, container, tmp_path):
    with container as entered:
        assert entered is container
    ((start, kwargs),) = fake_run.calls
    assert start == [
        "docker", "run", "--detach", "--rm", "--init", "--pull=never", "--network=none",
        "--cpus", "1",
        "--memory", "6g",
        "--memory-swap", "6g",
        "--name", "agl-rollout-1-2",
        "--label", f"arbas.run={tmp_path.resolve()}",
        "swe/image:latest", "sleep", "600",
    ]  # fmt: skip
    assert kwargs["check"] is True
    assert not {"-v", "--volume", "--mount", "--env", "-e", "--env-file"} & set(start)
    assert not any(":arbas" in arg for arg in start)

    (reaper,) = fake_reaper
    assert reaper.args[0] == [
        "sh",
        "-c",
        sandbox_module._REAPER_SCRIPT,
        "arbas-reap",
        "agl-rollout-1-2",
    ]
    assert reaper.kwargs["stdin"] is subprocess.PIPE
    assert reaper.kwargs["start_new_session"] is True
    reaper.stdin.close.assert_called_once()
    reaper.wait.assert_called_once_with(timeout=30)


def test_container_resource_limits_validate_and_normalize(fake_run, fake_reaper, tmp_path):
    with ContainerSandbox("swe/image:latest", "agl-rollout-custom", run_dir=tmp_path, cpus=1.5, memory="512Mi"):
        pass
    argv = fake_run.argvs[0]
    assert argv[argv.index("--cpus") + 1] == "1.5"
    assert argv[argv.index("--memory") + 1] == "512m"
    assert argv[argv.index("--memory-swap") + 1] == "512m"

    for kwargs in ({"cpus": 0}, {"cpus": -1}, {"memory": "0g"}, {"memory": "abc"}):
        with pytest.raises(ValueError):
            ContainerSandbox("swe/image:latest", "bad", run_dir=tmp_path, **kwargs)


def test_container_cleans_up_if_reaper_fails_to_start_or_exit(fake_run, fake_reaper, container, monkeypatch):
    with pytest.raises(RuntimeError, match="boom"), container:
        raise RuntimeError("boom")
    (reaper,) = fake_reaper
    reaper.stdin.close.assert_called_once()
    reaper.wait.assert_called_once_with(timeout=30)

    def failing_reaper(*_a, **_kw):
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.wait.return_value = 1
        return proc

    monkeypatch.setattr(sandbox_module.subprocess, "Popen", failing_reaper)
    with pytest.raises(RuntimeError, match="Failed to remove Docker container agl-rollout-1-2: exit 1"), container:
        pass

    fake_run.calls.clear()
    reaper_proc = MagicMock()
    reaper_proc.stdin = MagicMock()
    reaper_proc.wait.return_value = 0
    monkeypatch.setattr(sandbox_module.subprocess, "Popen", MagicMock(return_value=reaper_proc))
    fake_run.results = [subprocess.CalledProcessError(125, ["docker", "run"])]
    with pytest.raises(subprocess.CalledProcessError), container:
        pass
    reaper_proc.stdin.close.assert_called_once()
    reaper_proc.wait.assert_called_once_with(timeout=30)


def test_run_bash_executes_through_docker_exec_with_in_container_timeout(fake_run, container):
    fake_run.results = [subprocess.CompletedProcess([], 0, b"out\n", b"err\n")]
    assert container.run_bash("echo hi", 3) == ("out\nerr\n", 0)
    ((argv, kwargs),) = fake_run.calls
    name_at = argv.index("agl-rollout-1-2")
    assert argv[:4] == ["docker", "exec", "-w", "/testbed"]
    env_flags = argv[4:name_at]
    assert env_flags[::2] == ["-e"] * (len(env_flags) // 2)
    assert "PAGER=cat" in env_flags
    assert argv[name_at:] == ["agl-rollout-1-2", "timeout", "--kill-after=5", "3", "bash", "-lc", "echo hi"]
    assert kwargs["input"] is None
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["timeout"] == 18  # host-side backstop: in-container timeout + 15s


def test_exec_with_stdin_adds_interactive_flag_and_pipes_input(fake_run, container):
    container._exec(["tar", "-xf", "-"], stdin=b"payload", timeout=2)
    ((argv, kwargs),) = fake_run.calls
    assert argv[:5] == ["docker", "exec", "-i", "-w", "/testbed"]
    assert argv[-3:] == ["tar", "-xf", "-"]
    assert kwargs["input"] == b"payload"
    assert kwargs["stdin"] is None


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.TimeoutExpired(["docker"], 18, b"partial ", b"output"),
        subprocess.CompletedProcess([], 124, b"partial ", b"output"),
    ],
    ids=["host-timeout", "timeout-exit-124"],
)
def test_run_bash_reports_timeouts_as_124(fake_run, container, failure):
    fake_run.results = [failure]
    assert container.run_bash("sleep 99", 3) == ("partial output\n[timed out after 3s]", 124)


def test_run_bash_treats_sigkill_after_timeout_as_124_not_earlier(fake_run, container, monkeypatch):
    times = iter([10.0, 13.1, 20.0, 20.5])
    monkeypatch.setattr(sandbox_module.time, "monotonic", lambda: next(times))
    fake_run.results = [
        subprocess.CompletedProcess([], 137, b"hung ", b""),
        subprocess.CompletedProcess([], 137, b"oom", b""),
    ]
    assert container.run_bash("sleep 99", 3) == ("hung \n[timed out after 3s]", 124)
    assert container.run_bash("oom-cmd", 3) == ("oom", 137)


def test_run_bash_reports_start_failure(fake_run, container):
    fake_run.results = [FileNotFoundError("docker")]
    output, rc = container.run_bash("true", 3)
    assert rc == 1
    assert output.startswith("[failed to start: ")
    assert "docker" in output


# --------------------------------------------------------------------------------------------------------------------
# ContainerSandbox git/pytest workflow on a real git repository (via LocalTestSandbox._exec)
# --------------------------------------------------------------------------------------------------------------------

GOOD_CALC = "def add(a, b):\n    return a + b\n"
BUGGY_CALC = "def add(a, b):\n    return a - b\n"
HALF_FIXED_CALC = "def add(a, b):\n    return a + b if a > 0 else a - b\n"
TEST_CALC = (
    "from calc import add\n\n"
    "def test_add():\n    assert add(2, 3) == 5\n\n"
    "def test_add_negative():\n    assert add(-1, -1) == -2\n"
)
CONFTEST = "import pytest\n\n@pytest.fixture\ndef three():\n    return 3\n"
TEST_CALC_WITH_FIXTURE = "from calc import add\n\ndef test_add(three):\n    assert add(2, three) == 5\n"
F2P = ["test_calc.py::test_add"]
EVAL_META = {"FAIL_TO_PASS": F2P, "PASS_TO_PASS": []}
CHEAT_CONFTESTS = {
    "passes-everything": (
        "import pytest\n\n"
        "@pytest.hookimpl(hookwrapper=True)\n"
        "def pytest_runtest_makereport(item, call):\n"
        "    outcome = yield\n"
        "    outcome.get_result().outcome = 'passed'\n"
    ),
    "deselects-everything": "def pytest_collection_modifyitems(items):\n    items.clear()\n",
}


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return done.stdout


def build_task_repo(repo: Path, *, test_source: str, extra_files: dict[str, str] | None = None) -> None:
    """``main`` has the good code; ``fixture-instance`` = buggy code commit + a commit removing the tests."""
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "config", "user.name", "Offline fixture")
    files = {"calc.py": GOOD_CALC, "test_calc.py": test_source, **(extra_files or {})}
    for name, text in files.items():
        (repo / name).write_text(text)
    git(repo, "add", ".")
    git(repo, "commit", "-m", "Clean main")
    git(repo, "checkout", "-b", "fixture-instance")
    (repo / "calc.py").write_text(BUGGY_CALC)
    git(repo, "commit", "-am", "Bug Patch")
    git(repo, "rm", "test_calc.py")
    git(repo, "commit", "-m", "Remove F2P Tests")
    git(repo, "checkout", "main")


def make_sandbox(tmp_path: Path, f2p: list[str], **repo_kwargs: Any) -> LocalTestSandbox:
    workdir = tmp_path / "work"
    build_task_repo(workdir, **repo_kwargs)
    box = LocalTestSandbox(workdir, git_dir=tmp_path / "base.git")
    box.setup("fixture-instance", f2p)
    return box


def workdir_of(box: LocalTestSandbox) -> Path:
    return Path(box.root)


def fix(box: LocalTestSandbox, source: str = GOOD_CALC) -> None:
    (workdir_of(box) / "calc.py").write_text(source)


@pytest.fixture
def box(tmp_path) -> LocalTestSandbox:
    return make_sandbox(tmp_path, F2P, test_source=TEST_CALC)


def test_setup_erases_history_and_hides_tests(box):
    workdir = workdir_of(box)
    assert not (workdir / ".git").exists()
    assert not (workdir / "test_calc.py").exists()
    assert (workdir / "calc.py").read_text() == BUGGY_CALC
    assert git(workdir, "--git-dir", box.git_dir, "rev-list", "--count", "HEAD").strip() == "1"
    subjects = git(workdir, "--git-dir", box.git_dir, "log", "--all", "--format=%s")
    assert subjects.split() == ["baseline"]
    assert "Bug Patch" not in subjects
    assert "Clean main" not in subjects


def test_capture_patch_is_empty_without_edits_and_raises_on_git_failure(box, tmp_path):
    assert box.capture_patch() == ""
    uninitialized = LocalTestSandbox(tmp_path / "missing-baseline")
    (tmp_path / "missing-baseline").mkdir()
    with pytest.raises(RuntimeError, match="failed with exit"):
        uninitialized.capture_patch()


def test_capture_patch_contains_only_agent_edits(box):
    output, rc = box.run_bash("sed -i.bak 's/a - b/a + b/' calc.py && rm calc.py.bak && echo hi > new.py", 30)
    assert rc == 0, output
    patch = box.capture_patch()
    assert "-    return a - b" in patch
    assert "+    return a + b" in patch
    assert "+++ b/new.py" in patch
    assert "test_calc.py" not in patch
    assert ".bak" not in patch


def test_evaluate_resolves_after_fix_and_restores_hidden_tests(box):
    fix(box)
    assert "test_calc.py" not in box.capture_patch()
    result = box.evaluate(EVAL_META, 60)
    assert result.resolved is True
    assert (result.reward, result.partial_reward, result.timed_out) == (1.0, 1.0, False)
    assert result.reason == "FAIL_TO_PASS 1/1 passed, PASS_TO_PASS 0/0 ok (f2p_only)"
    assert (workdir_of(box) / "test_calc.py").read_text() == TEST_CALC


def test_evaluate_without_fix_fails(box):
    result = box.evaluate(EVAL_META, 60)
    assert (result.reward, result.resolved, result.partial_reward, result.timed_out) == (0.0, False, 0.0, False)
    assert result.reason.startswith("FAIL_TO_PASS 0/1 passed")


def test_evaluate_syntax_error_scores_zero(box):
    fix(box, "def add(:\n")
    result = box.evaluate(EVAL_META, 60)
    assert (result.reward, result.resolved, result.timed_out) == (0.0, False, False)


def test_evaluate_without_fail_to_pass_is_rejected(box):
    assert box.evaluate({"FAIL_TO_PASS": []}, 60) == Evaluation(0.0, False, "no FAIL_TO_PASS in eval meta", False)


def test_evaluate_partial_credit_for_half_fixed_f2p(tmp_path):
    f2p = ["test_calc.py::test_add", "test_calc.py::test_add_negative"]
    box = make_sandbox(tmp_path, f2p, test_source=TEST_CALC)
    fix(box, HALF_FIXED_CALC)
    result = box.evaluate({"FAIL_TO_PASS": f2p, "PASS_TO_PASS": []}, 60)
    assert (result.reward, result.resolved) == (0.0, False)
    assert result.partial_reward == 0.5
    assert result.reason.startswith("FAIL_TO_PASS 1/2 passed")


def test_evaluate_f2p_only_ignores_pass_to_pass_in_other_files(box):
    fix(box)
    meta = {"FAIL_TO_PASS": F2P, "PASS_TO_PASS": ["missing_file.py::test_x"]}
    assert box.evaluate(meta, 60, f2p_only=True).resolved is True
    strict = box.evaluate(meta, 60, f2p_only=False)
    assert strict.resolved is False
    assert strict.reason.endswith("PASS_TO_PASS 0/1 ok")


@pytest.mark.parametrize("fixed", [True, False], ids=["fixed", "unfixed"])
@pytest.mark.parametrize("cheat", CHEAT_CONFTESTS.values(), ids=CHEAT_CONFTESTS.keys())
def test_planted_conftest_does_not_change_the_outcome(box, cheat, fixed):
    workdir = workdir_of(box)
    (workdir / "tests").mkdir()
    for conftest in (workdir / "conftest.py", workdir / "tests" / "conftest.py"):
        conftest.write_text(cheat)
    if fixed:
        fix(box)
    assert box.evaluate(EVAL_META, 60).resolved is fixed
    assert not (workdir / "conftest.py").exists()
    assert not (workdir / "tests" / "conftest.py").exists()


def test_planted_pytest_module_does_not_shadow_the_real_pytest(box):
    (workdir_of(box) / "pytest.py").write_text("raise SystemExit(0)\n")
    assert box.evaluate(EVAL_META, 60).resolved is False
    fix(box)
    assert box.evaluate(EVAL_META, 60).resolved is True


def test_modified_task_conftest_is_restored_from_the_task_commit(tmp_path):
    box = make_sandbox(tmp_path, F2P, test_source=TEST_CALC_WITH_FIXTURE, extra_files={"conftest.py": CONFTEST})
    workdir = workdir_of(box)
    assert (workdir / "conftest.py").read_text() == CONFTEST
    fix(box)
    (workdir / "conftest.py").write_text("import pytest\n\n@pytest.fixture\ndef three():\n    return 99\n")
    assert box.evaluate(EVAL_META, 60).resolved is True
    assert (workdir / "conftest.py").read_text() == CONFTEST


@pytest.fixture
def bare(tmp_path) -> LocalTestSandbox:
    workdir = tmp_path / "work"
    workdir.mkdir()
    return LocalTestSandbox(workdir)


def test_run_bash_returns_output_and_exit_code(bare):
    assert bare.run_bash("echo out; echo err >&2; exit 7", 30) == ("out\nerr\n", 7)


def test_run_bash_timeout_returns_124_and_kills_background_children(bare):
    output, rc = bare.run_bash("(sleep 0.3; touch orphan) & wait", 0.03)
    assert rc == 124
    assert output.endswith("[timed out after 0.03s]")
    time.sleep(0.6)
    assert not (workdir_of(bare) / "orphan").exists()


def test_run_bash_kills_background_children_after_normal_exit(bare):
    assert bare.run_bash("(sleep 0.3; touch orphan) >/dev/null 2>&1 &", 30)[1] == 0
    time.sleep(0.6)
    assert not (workdir_of(bare) / "orphan").exists()


# --------------------------------------------------------------------------------------------------------------------
# Per-attempt SIGKILL & LocalReconciler timeout integration (real child processes, fake docker CLI)
# --------------------------------------------------------------------------------------------------------------------


def _install_fake_docker_cli(tmp_path: Path) -> tuple[Path, Path, Path]:
    started = tmp_path / "started"
    removed = tmp_path / "removed"
    bin_dir = tmp_path / "bin"
    started.mkdir()
    removed.mkdir()
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"started = Path({str(started)!r})\n"
        f"removed = Path({str(removed)!r})\n"
        "args = sys.argv[1:]\n"
        "if args[0] == 'run':\n"
        "    name = args[args.index('--name') + 1]\n"
        "    (started / name).write_text('\\n'.join(args), encoding='utf-8')\n"
        "elif args[0] == 'rm':\n"
        "    for item in args[1:]:\n"
        "        if not item.startswith('-'):\n"
        "            (removed / item).write_text('removed', encoding='utf-8')\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    return bin_dir, started, removed


def _wait_for(predicate, timeout: float = 4.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met before timeout")


def test_sigkill_of_timed_out_worker_removes_only_its_container(tmp_path):
    bin_dir, started, removed = _install_fake_docker_cli(tmp_path)
    repo_root = Path(__file__).resolve().parents[2]
    stop_file = tmp_path / "stop"
    script = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from examples.swe_smith.hierarchical.sandbox import ContainerSandbox\n"
        "name = sys.argv[1]\n"
        "stop = Path(sys.argv[2])\n"
        "with ContainerSandbox('swe/img', name, run_dir=stop.parent, ttl_seconds=3600):\n"
        "    while not stop.exists():\n"
        "        time.sleep(0.02)\n"
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": str(repo_root),
    }
    w1 = subprocess.Popen(
        [sys.executable, "-c", script, "agl-rollout-1-0", str(stop_file)], env=env, start_new_session=True
    )
    w2 = subprocess.Popen(
        [sys.executable, "-c", script, "agl-rollout-2-0", str(stop_file)], env=env, start_new_session=True
    )
    try:
        _wait_for(lambda: (started / "agl-rollout-1-0").exists() and (started / "agl-rollout-2-0").exists())
        os.killpg(w1.pid, signal.SIGKILL)
        w1.wait(timeout=5)
        _wait_for(lambda: (removed / "agl-rollout-1-0").exists())
        assert not (removed / "agl-rollout-2-0").exists()
        assert w2.poll() is None

        stop_file.write_text("done", encoding="utf-8")
        assert w2.wait(timeout=5) == 0
        assert (removed / "agl-rollout-2-0").exists()
    finally:
        stop_file.write_text("done", encoding="utf-8")
        for proc in (w1, w2):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)


@pytest.mark.asyncio
async def test_local_reconciler_timeout_removes_container_while_sibling_continues(tmp_path, monkeypatch):
    bin_dir, started, removed = _install_fake_docker_cli(tmp_path)
    repo_root = Path(__file__).resolve().parents[2]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (tmp_path / "timeout_test_worker.py").write_text(
        "import os, time\n"
        "from pathlib import Path\n"
        "from examples.swe_smith.hierarchical.sandbox import ContainerSandbox\n"
        "class Worker:\n"
        "    def run(self):\n"
        "        name = f\"agl-rollout-{os.environ['AGL_ROLLOUT_ID']}-{os.environ['AGL_ATTEMPT_ID']}\"\n"
        "        with ContainerSandbox('swe/img', name, run_dir=Path(os.environ['AGL_RUN_DIR']), ttl_seconds=3600):\n"
        "            time.sleep(30 if os.environ['TASK_MODE'] == 'hang' else 1.8)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PYTHONPATH", f"{tmp_path}{os.pathsep}{repo_root}")
    monkeypatch.setenv("AGL_RUN_DIR", str(run_dir))

    rollouts = {
        rid: Rollout(
            rollout_id=rid,
            input={"mode": mode},
            config=RolloutConfig(
                timeout_seconds=timeout_s,
                local=RolloutLocalConfig(
                    agent_class="timeout_test_worker:Worker",
                    env_map={"TASK_MODE": "input.mode"},
                ),
            ),
            status=RolloutLifecycleStatus(state=RolloutState.QUEUING, created_at=1.0, updated_at=1.0),
        )
        for rid, mode, timeout_s in (("r1", "hang", 1), ("r2", "quick", 10))
    }

    api = AsyncMock(spec=AgentLightningAsyncClient)

    async def fake_get(_path, **_kwargs):
        active = [
            r.model_dump(mode="json")
            for r in rollouts.values()
            if r.status.state in (RolloutState.QUEUING, RolloutState.RUNNING)
        ]
        return httpx.Response(200, json=active, request=httpx.Request("GET", "http://server/api/rollouts"))

    async def fake_patch(path, *, json):
        rid = path.rsplit("/", 1)[-1]
        for key, value in json["status"].items():
            setattr(rollouts[rid].status, key, RolloutState(value) if key == "state" else value)
        return httpx.Response(200, json={}, request=httpx.Request("PATCH", f"http://server{path}"))

    api.get.side_effect = fake_get
    api.patch.side_effect = fake_patch
    reconciler = LocalReconciler(
        api,
        OmegaConf.create(
            {
                "runner_type": "local",
                "agl_server": {"url": "http://127.0.0.1:8080", "key": "k"},
                "local_runner": {"maximum_size": 2, "poll_interval": 0.02},
            }
        ),
    )

    try:
        await reconciler._reconcile_once()
        _wait_for(lambda: (started / "agl-rollout-r1-0").exists() and (started / "agl-rollout-r2-0").exists())

        deadline = time.monotonic() + 3.0
        while rollouts["r1"].status.state != RolloutState.FAILED and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
            await reconciler._reconcile_once()
        assert rollouts["r1"].status.state == RolloutState.FAILED
        assert "timed out" in (rollouts["r1"].status.error_message or "")
        _wait_for(lambda: (removed / "agl-rollout-r1-0").exists())
        assert rollouts["r2"].status.state == RolloutState.RUNNING
        assert not (removed / "agl-rollout-r2-0").exists()

        deadline = time.monotonic() + 5.0
        while rollouts["r2"].status.state != RolloutState.SUCCEEDED and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
            await reconciler._reconcile_once()
        assert rollouts["r2"].status.state == RolloutState.SUCCEEDED
        _wait_for(lambda: (removed / "agl-rollout-r2-0").exists())
    finally:
        await reconciler._shutdown()
