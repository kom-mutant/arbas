import http.client
import json
import subprocess
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
from omegaconf import OmegaConf
from openai import APIError, OpenAI, RateLimitError
from openai.types.chat import ChatCompletion
from pydantic import ValidationError

from examples.swe_smith.hierarchical import run, smith
from examples.swe_smith.hierarchical.agents import ORCHESTRATOR_TOOLS, Orchestrator, SubAgent, _unparsed_call
from examples.swe_smith.hierarchical.config import AgentSettings, load_config
from examples.swe_smith.hierarchical.oracle import LinUCB, ModelRouter
from examples.swe_smith.hierarchical.types import (
    AgentRunResult,
    AgentSpec,
    AssignmentResult,
    ModelSettings,
    TaskAssignment,
    TokenUsage,
)
from examples.swe_smith.train_smith_agent import build_agent_config, load_experiment_config

TRAINING_YAML = Path(__file__).resolve().parents[2] / "examples/swe_smith/training.yaml"


def reply(text=None, tool=None, arguments=None, *, finish=None, **fields):
    calls = (
        []
        if tool is None
        else [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": tool, "arguments": json.dumps(arguments or {})},
            }
        ]
    )
    return ChatCompletion.model_validate(
        {
            "id": "offline",
            "object": "chat.completion",
            "created": 0,
            "model": "offline",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish or ("tool_calls" if calls else "stop"),
                    "message": {"role": "assistant", "content": text, "tool_calls": calls, **fields},
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        }
    )


class Client:
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs)))
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def base_agent_settings_dict() -> dict:
    config = load_experiment_config(TRAINING_YAML)
    return OmegaConf.to_container(config.agents, resolve=True)


@pytest.fixture
def config(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(smith, "TESTBED", str(repo))
    monkeypatch.setattr(smith, "_HIDDEN_GIT_DIR", str(tmp_path / "hidden-git"))
    agent_cfg = base_agent_settings_dict()
    agent_cfg["orchestrator"]["model"] = "trainable"
    agent_cfg["orchestrator"]["max_turns"] = 100
    agent_cfg["subagent"]["base_url"] = "https://workers.invalid/v1"
    agent_cfg["subagent"]["model"] = "worker"
    agent_cfg["oracle"]["models"] = []
    agent_cfg["oracle"]["learn"] = True
    settings = {
        "ARBAS_AGENT_CONFIG": AgentSettings.model_validate(agent_cfg).model_dump_json(),
        "AGL_OPENAI_BASE_URL": "https://gateway.invalid/v1",
        "AGL_KEY": "gateway-only",
        "OPENROUTER_API_KEY": "worker-only",
    }
    with patch.dict("os.environ", settings, clear=True):
        return load_config()


@pytest.fixture
def task(config):
    repo = Path(smith.TESTBED)

    def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    git("init", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Offline fixture")
    (repo / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (repo / "test_calc.py").write_text("from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    git("add", ".")
    git("commit", "-m", "Clean main")
    git("checkout", "-b", "fixture-instance")
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    git("commit", "-am", "Bug Patch")
    git("rm", "test_calc.py")
    git("commit", "-m", "Remove F2P Tests")
    git("checkout", "main")
    return {
        "instance_id": "fixture-instance",
        "repo": "offline/fixture",
        "FAIL_TO_PASS": ["test_calc.py::test_add"],
        "PASS_TO_PASS": [],
    }


def test_sequential_assignments_share_files_but_not_histories(config):
    worker = Client(
        [
            reply(tool="bash", arguments={"command": "printf shared > module.py && printf PRIVATE"}),
            reply("Created module.py"),
            reply(tool="bash", arguments={"command": "cat module.py"}),
            reply("Verified shared"),
            reply("Fresh follow-up on the first worker"),
        ]
    )
    root = Client(
        [
            reply(tool="create_subagent", arguments={"name": "fix"}),
            reply(tool="assign_task", arguments={"agent": "fix_1", "prompt": "Create file", "context": "FIRST"}),
            reply(tool="create_subagent", arguments={"name": "verify"}),
            reply(tool="assign_task", arguments={"agent": "verify_2", "prompt": "Verify", "context": "SECOND"}),
            reply(tool="assign_task", arguments={"agent": "fix_1", "prompt": "Follow-up", "context": "THIRD"}),
            reply("Done"),
        ]
    )
    settings = [
        ModelSettings(model="small", temperature=0.7, max_tokens=4096, reasoning=None),
        ModelSettings(model="large", temperature=0.7, max_tokens=4096, reasoning={"effort": "low"}),
        ModelSettings(model="small", temperature=0.7, max_tokens=4096, reasoning=None),
    ]
    router = SimpleNamespace(select=MagicMock(side_effect=settings))
    result = Orchestrator(config, root, worker, router=router).run("Original issue")
    assert router.select.call_count == 3
    assert isinstance(result.orchestrator, AgentRunResult)
    assert not hasattr(result.orchestrator, "assignment")
    assert not hasattr(result.orchestrator, "patch_before")
    assert [r.model for r in result.assignments] == ["small", "large", "small"]
    assert [r.agent_name for r in result.assignments] == ["fix_1", "verify_2", "fix_1"]
    assert [r.usage.total_tokens for r in result.assignments] == [240, 240, 120]
    assert result.orchestrator.usage.total_tokens == 720
    assert len(worker.requests[2]["messages"]) == 2
    assert "FIRST" not in json.dumps(worker.requests[2]["messages"])
    assert "SECOND" in worker.requests[2]["messages"][1]["content"]
    assert "Original issue" in worker.requests[2]["messages"][1]["content"]
    assert len(worker.requests[4]["messages"]) == 2
    assert "FIRST" not in json.dumps(worker.requests[4]["messages"])
    assert "SECOND" not in json.dumps(worker.requests[4]["messages"])
    assert worker.requests[2]["extra_body"] == {"reasoning": {"effort": "low"}}
    assert "PRIVATE" not in json.dumps(root.requests[2]["messages"])
    assert "Created module.py" in root.requests[2]["messages"][-1]["content"]
    assert "shared" in result.assignments[1].messages[-2]["content"]
    assert {t["function"]["name"] for t in root.requests[0]["tools"]} == {"create_subagent", "assign_task"}


def test_invalid_tools_and_arguments_are_feedback_not_execution(config):
    client = Client(
        [
            reply(tool="bash", arguments={"command": "touch unexpected"}),
            reply(tool="create_subagent", arguments={"name": "unknown"}),
            reply(tool="assign_task", arguments={"agent": "missing", "prompt": "Fix"}),
            reply(tool="assign_task", arguments={"agent": "missing", "prompt": ""}),
            reply("Cannot delegate"),
        ]
    )
    result = Orchestrator(config, client, Client([])).run("Issue")
    assert not (Path(smith.TESTBED) / "unexpected").exists()
    assert result.assignments == []
    assert "Available tools" in client.requests[1]["messages"][-1]["content"]
    assert "Invalid arguments" in client.requests[2]["messages"][-1]["content"]
    assert "Unknown agent" in client.requests[3]["messages"][-1]["content"]
    assert "Invalid arguments" in client.requests[4]["messages"][-1]["content"]
    assert result.orchestrator.tool_errors == {"unknown_tool": 1, "invalid_arguments": 2, "unknown_agent": 1}


def test_orchestrator_retries_tool_call_returned_as_text(config):
    root = Client(
        [
            reply(tool="create_subagent", arguments={"name": "research"}),
            reply(tool="assign_task", arguments={"agent": "research", "prompt": "Inspect"}),
            reply('<think>Delegate</think>\n<tool_call>\n{"name": "assign_task",'),
            reply(tool="assign_task", arguments={"agent": "research_1", "prompt": "Inspect"}),
            reply("Done"),
        ]
    )
    worker = Client([reply("Inspected")])

    result = Orchestrator(config, root, worker).run("Issue")

    assert "Available: research_1" in root.requests[2]["messages"][-1]["content"]
    assert "Retry one tool call" in root.requests[3]["messages"][-1]["content"]
    assert [assignment.agent_name for assignment in result.assignments] == ["research_1"]
    assert len(worker.requests) == 1
    assert result.orchestrator.summary == "Done"
    assert result.orchestrator.tool_errors == {"unknown_agent": 1, "unrecognized_call": 1}


def test_worker_retries_unparsed_tool_call(config):
    client = Client([reply("Attempting Bash", finish="tool_calls"), reply("Done")])
    worker = SubAgent(AgentSpec(name="fix_1", profile="fix", system_prompt="Fix"), client, config)
    result = worker.execute("Issue", TaskAssignment(agent_name="fix_1", prompt="Fix"), config.subagent)

    assert "Retry one tool call" in client.requests[1]["messages"][-1]["content"]
    assert result.summary == "Done"


@pytest.mark.parametrize(
    ("text", "finish", "tools", "expected"),
    [
        ("Attempting", "tool_calls", ("bash",), True),
        ('<tool_call>{"name": "bash"}</tool_call>', "stop", ("bash",), True),
        ("<function=assign_task>\n<parameter=agent>fix_1</parameter>", "stop", ("assign_task",), True),
        ('create_subagent(name="research")', "stop", ("create_subagent", "assign_task"), True),
        ('Next:\nassign_task(agent="fix_1", prompt="x")', "stop", ("create_subagent", "assign_task"), True),
        ('I will call bash(command="ls") now.', "stop", ("bash",), True),
        ('<think>Maybe bash(command="ls")</think>\nFixed calc.py.', "stop", ("bash",), False),
        ('Maybe bash(command="ls")</think>\nFixed calc.py.', "stop", ("bash",), False),
        ('Done</think>\nbash(command="ls")', "stop", ("bash",), True),
        ("Used create_subagent and assign_task to fix the bug.", "stop", ("create_subagent", "assign_task"), False),
        ("The helper `bash(cmd)` was checked.", "stop", ("bash",), False),
        ("my_bash(x) works properly.", "stop", ("bash",), False),
    ],
)
def test_unparsed_call_detector_distinguishes_text_calls_from_reports(text, finish, tools, expected):
    choice = reply(text, finish=finish).choices[0]
    assert _unparsed_call(choice, tools) is expected


@pytest.mark.parametrize("actor", ["orchestrator", "worker"])
def test_length_cutoff_never_executes_tools(config, actor):
    client = Client([reply(tool="bash", arguments={"command": "touch unexpected"}, finish="length")])
    if actor == "orchestrator":
        result = Orchestrator(config, client, Client([])).run("Issue").orchestrator
    else:
        worker = SubAgent(AgentSpec(name="fix_1", profile="fix", system_prompt="Fix"), client, config)
        result = worker.execute("Issue", TaskAssignment(agent_name="fix_1", prompt="Fix"), config.subagent)
    assert result.status == "failed"
    assert result.usage.total_tokens == 120
    assert not (Path(smith.TESTBED) / "unexpected").exists()


def test_api_failure_keeps_command_evidence_and_tokens(config):
    error = APIError("offline", httpx.Request("POST", "https://workers.invalid"), body=None)
    client = Client([reply(tool="bash", arguments={"command": "printf partial > partial.py && cat partial.py"}), error])
    worker = SubAgent(AgentSpec(name="fix_1", profile="fix", system_prompt="Fix"), client, config)
    result = worker.execute("Issue", TaskAssignment(agent_name="fix_1", prompt="Fix"), config.subagent)
    assert result.status == "failed"
    assert result.usage.total_tokens == 120
    assert "partial" in result.summary
    assert (Path(smith.TESTBED) / "partial.py").read_text() == "partial"


def test_router_programming_error_is_not_hidden(config):
    root = Client(
        [
            reply(tool="create_subagent", arguments={"name": "fix"}),
            reply(tool="assign_task", arguments={"agent": "fix_1", "prompt": "Fix"}),
        ]
    )
    router = SimpleNamespace(select=MagicMock(side_effect=ValueError("router bug")))
    with pytest.raises(ValueError, match="router bug"):
        Orchestrator(config, root, Client([]), router=router).run("Issue")


def test_gateway_pause_waits_without_consuming_a_step(config):
    request = httpx.Request("POST", "https://gateway.invalid")
    paused = RateLimitError("gateway paused", response=httpx.Response(429, request=request), body=None)
    client = Client([paused, reply("Done", reasoning_content="provider reasoning")])
    with patch("examples.swe_smith.hierarchical.agents.time.sleep"):
        result = Orchestrator(config, client, Client([])).run("Issue").orchestrator
    assert result.steps_taken == 1
    assert result.usage.total_tokens == 120
    assert result.messages[-1]["reasoning_content"] == "provider reasoning"


def test_shell_timeout_stops_background_children_and_hides_keys(config, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-router-value")
    monkeypatch.setenv("POLZA_API_KEY", "secret-polza-value")
    monkeypatch.setenv("AGL_KEY", "secret-gateway-value")
    monkeypatch.setenv("JUDGE_API_KEY", "secret-judge-value")
    output, rc = smith._run("env", 2)
    assert rc == 0
    assert "secret-router-value" not in output
    assert "POLZA_API_KEY=" not in output
    assert "secret-polza-value" not in output
    assert "secret-gateway-value" not in output
    assert "secret-judge-value" not in output
    _, rc = smith._run("(sleep 0.3; touch orphan) & wait", 0.03)
    assert rc == 124
    time.sleep(0.4)
    assert not (Path(smith.TESTBED) / "orphan").exists()


@pytest.mark.parametrize(
    ("error", "attempts"),
    [
        (urllib.error.HTTPError("https://events.invalid", 401, "Unauthorized", {}, None), 1),
        (urllib.error.URLError("offline"), 4),
        (http.client.RemoteDisconnected("connection closed"), 4),
        (ConnectionResetError("connection reset"), 4),
    ],
)
def test_reward_retries_are_finite(error, attempts):
    with (
        patch.dict("os.environ", {"AGL_EVENT_URL": "https://events.invalid"}),
        patch.object(run.urllib.request, "urlopen", side_effect=error) as send,
        patch.object(run.time, "sleep"),
    ):
        run.post_event("reward", {"value": 0}, retry=True)
    assert send.call_count == attempts


@pytest.mark.parametrize(
    ("router_enabled", "feedback_mode", "fixes_bug", "judge_valid"),
    [
        (False, "semantic", True, True),
        (True, "episode", True, True),
        (True, "semantic", True, True),
        (True, "semantic", False, True),
        (True, "semantic", True, False),
    ],
)
def test_main_with_real_sdk_prepares_fixes_grades_and_publishes(
    task, config, tmp_path, monkeypatch, router_enabled, feedback_mode, fixes_bug, judge_valid
):
    config.oracle.feedback = feedback_mode
    config.judge.base_url = "https://judge.invalid/v1"
    config.judge_api_key = "judge-only"
    config.judge.model = "judge-model"
    config.judge.temperature = 0
    config.judge.max_tokens = 512
    config.judge.reasoning = None
    semantic = router_enabled and feedback_mode == "semantic"
    if router_enabled:
        config.oracle.models = ["test/small", "test/large"]
        config.oracle.checkpoint = str(tmp_path / "router.pkl")
    task_prompt = "Fix addition" if fixes_bug else "Locate the arithmetic mistake and explain it"
    command = "sed -i.bak 's/a - b/a + b/' calc.py && rm calc.py.bak" if fixes_bug else "cat calc.py"
    score = 0.75 if fixes_bug else 1.0
    commands = [
        reply(
            tool="bash",
            arguments={"command": command + (" && printf helper > helper.py" if fixes_bug else "")},
        ),
        reply("Fixed calc.py" if fixes_bug else "Found subtraction in calc.py; no fix requested"),
    ]
    replies = {
        "gateway.invalid": [
            reply(tool="create_subagent", arguments={"name": "fix"}),
            reply(tool="assign_task", arguments={"agent": "fix_1", "prompt": task_prompt}),
            reply("Fixed"),
        ],
        "workers.invalid": commands,
        "judge.invalid": [
            reply(
                json.dumps({"score": score, "reason": "Evidence supports the assigned work"}) if judge_valid else "bad"
            )
        ],
    }
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.host in {"workers.invalid", "judge.invalid"}:
            assert not (Path(smith.TESTBED) / "test_calc.py").exists()
        if request.url.host == "judge.invalid":
            body = json.loads(request.content)
            assert body["model"] == "judge-model"
            assert "tools" not in body
            evidence = json.loads(body["messages"][1]["content"])
            assert evidence["assignment"] == task_prompt
            assert "Modify source only" in evidence["worker_instructions"]
            assert "calc.py" in evidence["command_observations"]
            if fixes_bug:
                assert "helper.py" in evidence["patch_after"]
            else:
                assert evidence["patch_after"] == ""
            assert evidence["patch_before"] == ""
        return httpx.Response(200, json=replies[request.url.host].pop(0).model_dump(mode="json"))

    def make_client(**kwargs):
        return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)))

    def publish(*args, **kwargs):
        if semantic:
            assert replies["judge.invalid"] == []
        if router_enabled:
            assert LinUCB.load(config.oracle.checkpoint).n_update == int(not semantic or judge_valid)

    monkeypatch.setenv("AGL_TASK_INPUT", "Fix addition")
    monkeypatch.setenv("AGL_EVAL_META", json.dumps(task))
    monkeypatch.setenv("HIER_OUTPUT_DIR", str(tmp_path / "results"))
    with (
        patch.object(run, "load_config", return_value=config),
        patch.object(run, "OpenAI", side_effect=make_client),
        patch.object(ModelRouter, "select", autospec=True, wraps=ModelRouter.select) as choose,
        patch.object(run, "post_event", side_effect=publish) as send,
    ):
        assert run.main() == 0
    if router_enabled:
        assert choose.call_count == 1
        assert choose.call_args.args[1] == "Fix addition"
    saved_text = (tmp_path / "results/fixture-instance.json").read_text()
    saved = json.loads(saved_text)
    assert saved["reward"] == int(fixes_bug)
    assert saved["timings_s"]["rollout_wall"] >= saved["timings_s"]["evaluation"] >= 0
    assert (saved["timings_s"]["judge"] is not None) == semantic
    if fixes_bug:
        assert "helper.py" in saved["patch"]
    else:
        assert saved["patch"] == ""
    assert "test_calc.py" not in saved["patch"]
    assert (Path(smith.TESTBED) / "test_calc.py").is_file()
    assert not (Path(smith.TESTBED) / ".git").exists()
    assert [r.url.host for r in requests] == [
        "gateway.invalid",
        "gateway.invalid",
        "workers.invalid",
        "workers.invalid",
        "gateway.invalid",
    ] + (["judge.invalid"] if semantic else [])
    for request in requests:
        expected = {
            "workers.invalid": config.subagent_api_key,
            "gateway.invalid": config.orchestrator_api_key,
            "judge.invalid": config.judge_api_key,
        }[request.url.host]
        assert request.headers["authorization"] == f"Bearer {expected}"
    assert config.orchestrator_api_key not in saved_text
    assert config.subagent_api_key not in saved_text
    assert config.judge_api_key not in saved_text
    assert [c.args[0] for c in send.call_args_list] == ["agent_output", "reward"]
    assert send.call_args_list[1].args[1]["value"] == int(fixes_bug)
    if router_enabled:
        decision = saved["oracle"]["decisions"][0]
        worker_requests = [json.loads(r.content) for r in requests if r.url.host == "workers.invalid"]
        assert {r["model"] for r in worker_requests} == {decision["model"]}
        expected_score = (score if judge_valid else None) if semantic else int(fixes_bug)
        assert decision["score"] == expected_score
        assert decision["cost"] == 40
        if expected_score is None:
            assert decision["reward"] is None
            assert saved["assignments"][0]["judgment"]["error"]
        else:
            assert decision["reward"] == pytest.approx(0.8 * expected_score - 0.2 * 40 / 10000)
        assert decision["learned"] == (not semantic or judge_valid)
        assert LinUCB.load(config.oracle.checkpoint).n_update == int(not semantic or judge_valid)
    else:
        assert "oracle" not in saved


@pytest.mark.parametrize("failure_stage", ["judge", "feedback"])
def test_main_publishes_and_saves_even_when_judge_or_feedback_raises(
    task, config, tmp_path, monkeypatch, failure_stage
):
    config.oracle.models = ["test/small", "test/large"]
    config.oracle.checkpoint = str(tmp_path / "router.pkl")
    config.oracle.feedback = "semantic"
    config.judge.base_url = "https://judge.invalid/v1"
    config.judge_api_key = "judge-only"

    replies = {
        "gateway.invalid": [
            reply(tool="create_subagent", arguments={"name": "fix"}),
            reply(tool="assign_task", arguments={"agent": "fix_1", "prompt": "Fix addition"}),
            reply("Fixed"),
        ],
        "workers.invalid": [
            reply(tool="bash", arguments={"command": "sed -i.bak 's/a - b/a + b/' calc.py && rm calc.py.bak"}),
            reply("Fixed calc.py"),
        ],
    }

    def respond(request):
        return httpx.Response(200, json=replies[request.url.host].pop(0).model_dump(mode="json"))

    def make_client(**kwargs):
        return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)))

    monkeypatch.setenv("AGL_TASK_INPUT", "Fix addition")
    monkeypatch.setenv("AGL_EVAL_META", json.dumps(task))
    monkeypatch.setenv("HIER_OUTPUT_DIR", str(tmp_path / "results"))

    judge_patch = (
        patch.object(run, "judge_assignments", side_effect=RuntimeError("judge down"))
        if failure_stage == "judge"
        else patch.object(run, "judge_assignments", return_value=[1.0])
    )
    feedback_patch = (
        patch.object(ModelRouter, "feedback", side_effect=OSError("disk full"))
        if failure_stage == "feedback"
        else patch.object(ModelRouter, "feedback", wraps=ModelRouter.feedback, autospec=True)
    )

    with (
        patch.object(run, "load_config", return_value=config),
        patch.object(run, "OpenAI", side_effect=make_client),
        judge_patch,
        feedback_patch,
        patch.object(run, "post_event") as send,
        pytest.raises(RuntimeError if failure_stage == "judge" else OSError),
    ):
        run.main()

    assert [c.args[0] for c in send.call_args_list] == ["agent_output", "reward"]
    assert send.call_args_list[1].args[1]["value"] == 1.0
    assert send.call_args_list[1].args[1]["resolved"] is True
    saved = json.loads((tmp_path / "results/fixture-instance.json").read_text())
    assert saved["reward"] == 1.0
    assert saved["resolved"] is True


def test_unfixed_source_is_not_rewarded_and_pytest_cannot_be_shadowed(task):
    smith.checkout_bug_commit(task["instance_id"])
    smith.relocate_git()
    (Path(smith.TESTBED) / "pytest.py").write_text("print('PASSED test_calc.py::test_add')\n")
    ev = smith.evaluate(task, 20)
    assert ev.reward == 0.0
    assert not ev.resolved


def test_syntax_error_returns_zero_reward_without_raising(task):
    smith.checkout_bug_commit(task["instance_id"])
    smith.relocate_git()
    (Path(smith.TESTBED) / "calc.py").write_text("def add(:\n")
    ev = smith.evaluate(task, 20)
    assert ev.reward == 0.0
    assert ev.resolved is False
    assert ev.timed_out is False


@pytest.mark.parametrize("host, expected_key", [("openrouter.ai", "router"), ("polza.ai", "polza")])
def test_env_config_keeps_worker_credentials_separate(host, expected_key):
    agent_cfg = base_agent_settings_dict()
    agent_cfg["subagent"]["base_url"] = f"https://{host}/api/v1"
    agent_cfg["subagent"]["reasoning"] = {"effort": "low"}
    agent_cfg["oracle"]["models"] = ["test/small", "test/large"]
    agent_cfg["oracle"]["learn"] = False
    with patch.dict(
        "os.environ",
        {
            "ARBAS_AGENT_CONFIG": AgentSettings.model_validate(agent_cfg).model_dump_json(),
            "AGL_OPENAI_BASE_URL": "https://gateway.invalid/mode/val/openai/v1",
            "AGL_KEY": "gateway",
            "OPENROUTER_API_KEY": "router",
            "POLZA_API_KEY": "polza",
        },
        clear=True,
    ):
        config = load_config()
    assert config.orchestrator_api_key == "gateway"
    assert config.subagent_api_key == expected_key
    assert config.subagent.reasoning == {"effort": "low"}
    assert config.orchestrator.reasoning is None
    assert config.oracle.models == ["test/small", "test/large"]
    assert config.oracle.learn is False
    assert config.mode == "val"
    config.orchestrator_base_url = "https://direct.invalid/v1"
    assert config.mode == "inference"


def test_polza_endpoint_requires_polza_key():
    agent_cfg = base_agent_settings_dict()
    agent_cfg["subagent"]["base_url"] = "https://polza.ai/api/v1"
    env = {
        "ARBAS_AGENT_CONFIG": AgentSettings.model_validate(agent_cfg).model_dump_json(),
        "OPENROUTER_API_KEY": "router",
    }
    with patch.dict("os.environ", env, clear=True), pytest.raises(ValueError, match="Set POLZA_API_KEY"):
        load_config()


def test_build_agent_config_resolves_interpolations_and_overrides():
    encoded = build_agent_config(
        TRAINING_YAML,
        "exp-1117-19",
        config_overrides=[
            "actor_rollout_ref.rollout.temperature=0.42",
            "agents.orchestrator.max_turns=10",
        ],
    )
    parsed = AgentSettings.model_validate_json(encoded)
    assert parsed.orchestrator.temperature == 0.42
    assert parsed.orchestrator.max_turns == 10
    assert parsed.oracle.checkpoint == "/agl-logs/exp-1117-19/oracle/linucb.pkl"


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("subagent", "max_steps"),
        ("subagent", "model"),
        ("subagent", "max_tokens"),
        ("oracle", "models"),
        ("sandbox", "f2p_only"),
        ("orchestrator", "max_turns"),
    ],
)
def test_missing_agent_setting_fails_without_fallback(section, field):
    agent_cfg = base_agent_settings_dict()
    del agent_cfg[section][field]
    with pytest.raises(ValidationError, match=field):
        AgentSettings.model_validate(agent_cfg)


def test_extra_agent_setting_is_rejected_fail_fast(tmp_path):
    agent_cfg = base_agent_settings_dict()
    agent_cfg["oracle"]["typo_field"] = 123
    with pytest.raises(ValidationError, match="extra_forbidden"):
        AgentSettings.model_validate(agent_cfg)

    cfg = OmegaConf.to_container(load_experiment_config(TRAINING_YAML))
    cfg["agents"]["subagent"]["typo_field"] = True
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(OmegaConf.to_yaml(cfg))
    with pytest.raises(ValidationError, match="extra_forbidden"):
        build_agent_config(bad_yaml, "test-run")


def test_training_reward_uses_yaml_penalty_settings(config):
    config.orchestrator_base_url = "https://gateway.invalid/mode/train/openai/v1"
    result = AgentRunResult(
        agent_name="orchestrator",
        model="model",
        steps_taken=90,
        usage=TokenUsage(max_prompt_tokens=57000),
    )
    solved = smith.Evaluation(1.0, True, "", False, 1.0)
    unsolved = smith.Evaluation(0.0, False, "", False)
    assert run.training_reward(solved, result, config) == pytest.approx(0.9)
    assert run.training_reward(unsolved, result, config) == 0.0
    config.orchestrator_base_url = "https://gateway.invalid/mode/val/openai/v1"
    assert run.training_reward(solved, result, config) == 1.0


@pytest.mark.parametrize(
    ("failure", "regression", "exit_code", "expected"),
    [
        ("FAILED", "PASSED", 1, 0.75),
        ("FAILED", "FAILED", 1, 0.0),
        ("ERROR", "PASSED", 1, 0.0),
        ("FAILED", "PASSED", 124, 0.0),
    ],
)
def test_partial_reward_requires_completed_tests_without_regressions(config, failure, regression, exit_code, expected):
    fail_to_pass = [f"test_suite.py::test_fix_{i}" for i in range(4)]
    pass_to_pass = "test_suite.py::test_existing"
    output = "\n".join(
        [f"PASSED {node}" for node in fail_to_pass[:3]]
        + [f"{failure} {fail_to_pass[3]}", f"{regression} {pass_to_pass}"]
    )
    with (
        patch.object(smith, "restore_f2p_tests"),
        patch.object(smith, "_run", side_effect=[("", 1), (output, exit_code)]),
    ):
        evaluation = smith.evaluate({"FAIL_TO_PASS": fail_to_pass, "PASS_TO_PASS": [pass_to_pass]}, 20)

    result = AgentRunResult(agent_name="orchestrator", model="model")
    config.orchestrator_base_url = "https://gateway.invalid/mode/train/openai/v1"
    assert evaluation.resolved is False
    config.reward.partial_credit = False
    assert run.training_reward(evaluation, result, config) == 0.0
    config.reward.partial_credit = True
    assert run.training_reward(evaluation, result, config) == expected
    config.orchestrator_base_url = "https://gateway.invalid/mode/val/openai/v1"
    assert run.training_reward(evaluation, result, config) == 0.0


def test_oracle_routes_each_assignment_then_learns_from_delayed_score(config, tmp_path):
    config.oracle.models = ["test/small", "test/large"]
    config.oracle.checkpoint = str(tmp_path / "router.pkl")
    config.subagent.reasoning = {"effort": "low"}
    router = ModelRouter(config)
    root = Client(
        [
            reply(tool="create_subagent", arguments={"name": "fix"}),
            reply(tool="assign_task", arguments={"agent": "fix_1", "prompt": "Find bug", "context": "First"}),
            reply(tool="assign_task", arguments={"agent": "fix_1", "prompt": "Fix bug", "context": "Second"}),
            reply("Done"),
        ]
    )
    worker = Client([reply("Found"), reply("Fixed")])
    swarm = Orchestrator(config, root, worker, router).run("An original issue " * 800)
    decisions = router.report()["decisions"]
    assert len(decisions) == 2
    assert len({d["program_id"] for d in decisions}) == 2
    assert [d["agent_name"] for d in decisions] == ["fix_1", "fix_1"]
    assert [d["model"] for d in decisions] == [r.model for r in swarm.assignments]
    assert all(r["extra_body"] == {"reasoning": {"effort": "low"}} for r in worker.requests)
    assert decisions[0]["features"] != decisions[1]["features"]
    assert router.selector.n_update == 0
    assert LinUCB.load(config.oracle.checkpoint).n_update == 0

    router.feedback([1, 0], swarm.assignments)
    assert [d["score"] for d in router.report()["decisions"]] == [1, 0]
    assert router.report()["decisions"][0]["reward"] > 0
    assert router.report()["decisions"][1]["reward"] < 0
    resumed = ModelRouter(config)
    assert resumed.selector.n_select == 2
    assert resumed.selector.n_update == 2
    np.testing.assert_array_equal(resumed.selector.A, router.selector.A)
    np.testing.assert_array_equal(resumed.selector.b, router.selector.b)
    expected_b = np.zeros_like(resumed.selector.b)
    for x, d in router.choices:
        expected_b[resumed.selector.index[d["model"]]] += d["reward"] * x
    np.testing.assert_allclose(resumed.selector.b, expected_b)


@pytest.mark.parametrize("validation_url", [False, True])
def test_oracle_frozen_evaluation_does_not_change_checkpoint(config, tmp_path, validation_url):
    config.oracle.models = ["test/small", "test/large"]
    config.oracle.checkpoint = str(tmp_path / "router.pkl")
    LinUCB(config.oracle.models).save(config.oracle.checkpoint)
    before = Path(config.oracle.checkpoint).read_bytes()
    if validation_url:
        config.orchestrator_base_url = "https://gateway.invalid/mode/val/rollout/1/v1"
    else:
        config.oracle.learn = False
    router = ModelRouter(config)
    spec = AgentSpec(name="fix_1", profile="fix", system_prompt="Fix")
    settings = router.select("Issue", spec, TaskAssignment(agent_name=spec.name, prompt="Fix"), config.subagent)
    router.feedback([1], [AssignmentResult(agent_name=spec.name, model=settings.model)])
    assert router.selector.n_update == 0
    assert router.report()["decisions"][0]["learned"] is False
    assert Path(config.oracle.checkpoint).read_bytes() == before
    assert not Path(config.oracle.checkpoint).with_suffix(".lock").exists()


def test_oracle_shared_checkpoint_keeps_updates_from_other_episodes(config, tmp_path):
    config.oracle.models = ["test/small", "test/large"]
    config.oracle.checkpoint = str(tmp_path / "router.pkl")
    routers = [ModelRouter(config) for _ in range(4)]

    def episode(router):
        spec = AgentSpec(name="fix_1", profile="fix", system_prompt="Fix")
        settings = router.select("Issue", spec, TaskAssignment(agent_name=spec.name, prompt="Fix"), config.subagent)
        result = AssignmentResult(agent_name=spec.name, model=settings.model, usage=TokenUsage(completion_tokens=100))
        router.feedback([1], [result])

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(episode, routers))
    learned = LinUCB.load(config.oracle.checkpoint)
    assert learned.n_select == 4
    assert learned.n_update == 4
    assert learned.pulls.sum() == 4
    x = routers[0].choices[0][0]
    np.testing.assert_allclose(learned.A.sum(axis=0), 2 * np.eye(64) + 4 * np.outer(x, x))
    np.testing.assert_allclose(learned.b.sum(axis=0), 4 * (0.8 - 0.2 * 100 / 10000) * x)


def test_oracle_rejects_checkpoint_from_another_model_pool(config, tmp_path):
    config.oracle.checkpoint = str(tmp_path / "router.pkl")
    LinUCB(["old/model"]).save(config.oracle.checkpoint)
    config.oracle.models = ["test/small", "test/large"]
    with pytest.raises(ValueError, match="checkpoint pool"):
        ModelRouter(config)


@pytest.fixture
def assignment_result():
    return AssignmentResult(
        agent_name="worker_identity",
        model="worker_model_identity",
        assignment=TaskAssignment(
            agent_name="worker_identity", prompt="Find the arithmetic bug", context="Look at calc"
        ),
        summary="Found subtraction instead of addition",
        status="completed",
        messages=[{"role": "tool", "content": "Command: cat calc.py\ndef add(a,b): return a-b"}],
        patch_before="before assignment",
        patch_after="after assignment",
    )


def test_semantic_judge_gets_evidence_without_worker_model_or_identity(config, assignment_result):
    client = Client([reply('{"score":0.8,"reason":"The shown function contains the arithmetic error"}')])
    judgment = run.evaluate_assignment(client, "Fix addition", assignment_result, config)
    assert judgment.score == 0.8
    assert judgment.error is None
    request = client.requests[0]
    assert "tools" not in request
    assert request["temperature"] == 0
    assert request["response_format"] == {"type": "json_object"}
    evidence = json.loads(request["messages"][1]["content"])
    assert evidence["assignment"] == "Find the arithmetic bug"
    assert "cat calc.py" in evidence["command_observations"]
    assert evidence["patch_before"] == "before assignment"
    assert evidence["patch_after"] == "after assignment"
    assert "worker_identity" not in request["messages"][1]["content"]
    assert "worker_model_identity" not in request["messages"][1]["content"]
    assert assignment_result.messages == [{"role": "tool", "content": "Command: cat calc.py\ndef add(a,b): return a-b"}]


@pytest.mark.parametrize(
    "response",
    [
        reply("not JSON"),
        reply('{"score":2,"reason":"out of range"}'),
        reply('{"score":"NaN","reason":"not finite"}'),
        reply('{"score":0.8}'),
        reply('{"score":0.8,"reason":"cut off"}', finish="length"),
        APIError("judge offline", httpx.Request("POST", "https://judge.invalid"), body=None),
    ],
)
def test_invalid_judge_result_is_missing_feedback_not_zero(config, assignment_result, response):
    judgment = run.evaluate_assignment(Client([response]), "Fix addition", assignment_result, config)
    assert judgment.score is None
    assert judgment.error


def test_judge_receives_full_saved_observation(config, assignment_result):
    observation = "BEGIN " + "x" * 20000 + " END"
    assignment_result.messages[0]["content"] = observation
    client = Client([reply('{"score":0.3,"reason":"Evidence received"}')])
    run.evaluate_assignment(client, "Fix addition", assignment_result, config)
    evidence = json.loads(client.requests[0]["messages"][1]["content"])
    assert evidence["command_observations"] == observation


def test_oracle_skips_failed_judgments_but_updates_for_valid_zero(config, tmp_path):
    config.oracle.models = ["test/small"]
    config.oracle.checkpoint = str(tmp_path / "router.pkl")
    router = ModelRouter(config)
    spec = AgentSpec(name="fix_1", profile="fix", system_prompt="Fix")
    results = []
    for prompt in ["First", "Second", "Third"]:
        assignment = TaskAssignment(agent_name=spec.name, prompt=prompt)
        settings = router.select("Issue", spec, assignment, config.subagent)
        results.append(AssignmentResult(agent_name=spec.name, model=settings.model, assignment=assignment))
    before = Path(config.oracle.checkpoint).read_bytes()
    router.feedback([None, None, None], results)
    assert Path(config.oracle.checkpoint).read_bytes() == before
    assert router.selector.n_update == 0
    router.feedback([1, None, 0], results)
    assert router.selector.n_update == 2
    assert [d["learned"] for d in router.report()["decisions"]] == [True, False, True]
    assert [d["reward"] for d in router.report()["decisions"]] == [0.8, None, 0]
    assert router.report()["score_source"] == "semantic_judge"


def test_assignment_patches_are_captured_before_later_workers_change_source(task, config, tmp_path):
    config.oracle.models = ["test/small"]
    config.oracle.checkpoint = str(tmp_path / "router.pkl")
    smith.checkout_bug_commit(task["instance_id"])
    smith.relocate_git()
    root = Client(
        [
            reply(tool="create_subagent", arguments={"name": "fix"}),
            reply(tool="assign_task", arguments={"agent": "fix_1", "prompt": "First change"}),
            reply(tool="assign_task", arguments={"agent": "fix_1", "prompt": "Correct the change"}),
            reply("Done"),
        ]
    )
    worker = Client(
        [
            reply(tool="bash", arguments={"command": "sed -i.bak 's/a - b/a * b/' calc.py && rm calc.py.bak"}),
            reply("First change"),
            reply(tool="bash", arguments={"command": r"sed -i.bak 's/a \* b/a + b/' calc.py && rm calc.py.bak"}),
            reply("Corrected"),
        ]
    )
    results = Orchestrator(config, root, worker, ModelRouter(config)).run("Fix addition").assignments
    assert results[0].patch_before == ""
    assert "+    return a * b" in results[0].patch_after
    assert "+    return a + b" not in results[0].patch_after
    assert results[1].patch_before == results[0].patch_after
    assert "+    return a + b" in results[1].patch_after


def test_training_parser_follows_effective_model(tmp_path):
    pytest.importorskip("verl")
    from examples.swe_smith.train_smith_agent import build_config

    (tmp_path / "config.json").write_text('{"model_type": "qwen3_5"}')
    override = "actor_rollout_ref.model.path="
    local = build_config(config_overrides=[override + str(tmp_path)])
    other = build_config(config_overrides=[override + "Qwen/Qwen3-8B"])

    assert local.actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser == "qwen3_coder"
    assert Path(local.actor_rollout_ref.rollout.engine_kwargs.vllm.chat_template).name == "qwen35_chat_template.jinja"
    assert other.actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser == "hermes"


def test_http_comment_allowed_and_usage_handles_none():
    assert smith._forbidden_action('python3 -c "\n# Test (HTTP default)\nprint(1)\n"') is None
    assert smith._forbidden_action("http https://example.com/file.py") is not None
    usage = TokenUsage()
    usage.add(SimpleNamespace(prompt_tokens=None, completion_tokens=7, total_tokens=None))
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens, usage.max_prompt_tokens) == (0, 7, 7, 0)


def test_training_config_settings(tmp_path):
    pytest.importorskip("verl")

    from examples.swe_smith.train_smith_agent import DEFAULT_CONFIG, build_config

    settings = load_experiment_config(DEFAULT_CONFIG)
    config_path = tmp_path / "experiment.yaml"
    OmegaConf.save(settings, config_path)
    config = build_config(config_path=config_path)

    assert config.actor_rollout_ref.model.path == settings.actor_rollout_ref.model.path
    assert config.actor_rollout_ref.model.lora_rank == settings.actor_rollout_ref.model.lora_rank
    assert config.actor_rollout_ref.model.lora.merge == settings.actor_rollout_ref.model.lora.merge
    assert config.actor_rollout_ref.rollout.n == settings.actor_rollout_ref.rollout.n
    assert config.actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend == "triton"
    assert (
        config.actor_rollout_ref.rollout.gpu_memory_utilization
        == settings.actor_rollout_ref.rollout.gpu_memory_utilization
    )
    assert config.ray_kwargs.ray_init.address == "local"
    assert config.trainer.logger == settings.trainer.logger
    assert config.data.train_files == settings.data.train_files
    assert config.agentlightning.trace_aggregator.trajectory_max_response_length == config.data.max_response_length
    assert "agents" not in config
    assert "experiment" not in config
    with pytest.raises(ValueError, match=r"rollout\.name=vllm"):
        build_config(config_overrides=["actor_rollout_ref.rollout.name=sglang"])
    with pytest.raises(ValueError, match="gdn_prefill_backend=triton"):
        build_config(config_overrides=["actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=flashinfer"])
    with pytest.raises(ValueError, match="local Ray"):
        build_config(config_overrides=["ray_kwargs.ray_init.address=auto"])
    with pytest.raises(ValueError, match="MAX_JOBS must be 1 or 2"):
        build_config(config_overrides=["ray_kwargs.ray_init.runtime_env.env_vars.MAX_JOBS=47"])


def test_prepare_ray_preserves_worker_defaults_and_checks_socket_path(monkeypatch):
    pytest.importorskip("verl")
    pytest.importorskip("ray")
    from examples.swe_smith.train_smith_agent import build_config, prepare_ray

    defaults = {"VLLM_DISABLE_COMPILE_CACHE": "1", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"}
    monkeypatch.setattr("verl.trainer.constants_ppo.get_ppo_ray_runtime_env", lambda: {"env_vars": defaults})
    config = build_config(
        config_overrides=[
            "ray_kwargs.ray_init.runtime_env.env_vars.MAX_JOBS=1",
            "ray_kwargs.ray_init.runtime_env.env_vars.TEST_VAR=present",
        ]
    )
    monkeypatch.setenv("RAY_TMPDIR", "/tmp/agl-ray")
    monkeypatch.setenv("VERL_WEIGHT_TRANSFER_IPC_DIR", "/tmp/agl-ipc-test")
    prepare_ray(config)
    assert dict(config.ray_kwargs.ray_init.runtime_env.env_vars) == {
        **defaults,
        "MAX_JOBS": "1",
        "VLLM_USE_FLASHINFER_MOE_FP16": "0",
        "TEST_VAR": "present",
        "VERL_WEIGHT_TRANSFER_IPC_DIR": "/tmp/agl-ipc-test",
    }

    monkeypatch.setenv("RAY_TMPDIR", "/tmp/" + "long-" * 20)
    with pytest.raises(RuntimeError, match="Ray socket path is too long"):
        prepare_ray(config)


def test_extra_assignment_arguments_are_rejected(config):
    assignment_tool = next(tool for tool in ORCHESTRATOR_TOOLS if tool["function"]["name"] == "assign_task")
    assert assignment_tool["function"]["parameters"]["additionalProperties"] is False
    root = Client(
        [
            reply(tool="create_subagent", arguments={"name": "fix"}),
            reply(tool="assign_task", arguments={"agent": "fix_1", "prompt": "Fix", "prompt_2": "Lost instruction"}),
            reply("Done"),
        ]
    )
    worker = Client([])
    result = Orchestrator(config, root, worker).run("Issue")
    assert not result.assignments
    assert "extra_forbidden" in root.requests[2]["messages"][-1]["content"]
    assert result.orchestrator.tool_errors == {"invalid_arguments": 1}
    assert not worker.requests


def test_budget_reserves_report_without_extra_request(config):
    config.subagent.max_steps = 2
    client = Client([reply(tool="bash", arguments={"command": "printf evidence"}), reply("Found evidence; no fix yet")])
    worker = SubAgent(AgentSpec(name="research_1", profile="research", system_prompt="Research"), client, config)
    result = worker.execute("Issue", TaskAssignment(agent_name="research_1", prompt="Find bug"), config.subagent)
    assert result.status == "completed"
    assert result.summary == "Found evidence; no fix yet"
    assert result.steps_taken == 2
    assert "tools" not in client.requests[-1]
    assert "evidence" in json.dumps(client.requests[-1]["messages"])


def test_budget_reserves_report_empty_summary_sets_step_limit(config):
    config.subagent.max_steps = 2
    client = Client([reply(tool="bash", arguments={"command": "printf evidence"}), reply("")])
    worker = SubAgent(AgentSpec(name="research_1", profile="research", system_prompt="Research"), client, config)
    result = worker.execute("Issue", TaskAssignment(agent_name="research_1", prompt="Find bug"), config.subagent)
    assert result.status == "step_limit"
    assert "step limit reached" in result.summary
    assert result.steps_taken == 2


def test_final_report_does_not_execute_a_returned_tool_call(config):
    config.subagent.max_steps = 2
    client = Client(
        [
            reply(tool="bash", arguments={"command": "printf first"}),
            reply(tool="bash", arguments={"command": "touch unexpected"}),
        ]
    )
    worker = SubAgent(AgentSpec(name="fix_1", profile="fix", system_prompt="Fix"), client, config)
    result = worker.execute("Issue", TaskAssignment(agent_name="fix_1", prompt="Fix"), config.subagent)
    assert result.status == "step_limit"
    assert result.tool_errors == {"unexpected_call": 1}
    assert "Tool call returned during final report" in result.summary
    assert not (Path(smith.TESTBED) / "unexpected").exists()


def test_hierarchical_package_runs_standalone_as_in_k8s_pod(tmp_path):
    import shutil
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "examples/swe_smith/hierarchical"
    agl = tmp_path / "agl"
    shutil.copytree(src, agl / "hierarchical", ignore=shutil.ignore_patterns("__pycache__"))
    for py_file in (agl / "hierarchical").glob("*.py"):
        assert "from datetime import UTC" not in py_file.read_text(encoding="utf-8")

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path = [''] + [p for p in sys.path if p != {str(repo_root)!r}]; "
            "import hierarchical.run; print(hierarchical.run.main.__name__)",
        ],
        cwd=agl,
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "main"
