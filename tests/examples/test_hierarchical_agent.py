import json
import subprocess
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import numpy as np
import pytest
from openai import APIError, OpenAI, RateLimitError
from openai.types.chat import ChatCompletion

from examples.swe_smith.hierarchical import run, smith
from examples.swe_smith.hierarchical.agents import Orchestrator, SubAgent
from examples.swe_smith.hierarchical.config import Config, load_config, select_subagent_settings
from examples.swe_smith.hierarchical.oracle.bandit import LinUCB
from examples.swe_smith.hierarchical.routing import ModelRouter
from examples.swe_smith.hierarchical.types import AgentSpec, ModelSettings, TaskAssignment, TaskResult, TokenUsage


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


@pytest.fixture
def config(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(smith, "TESTBED", str(repo))
    monkeypatch.setattr(smith, "_HIDDEN_GIT_DIR", str(tmp_path / "hidden-git"))
    return Config(
        orchestrator_base_url="https://gateway.invalid/v1",
        orchestrator_api_key="gateway-only",
        subagent_base_url="https://workers.invalid/v1",
        subagent_api_key="worker-only",
        orchestrator_model=ModelSettings(model="trainable"),
        subagent_model=ModelSettings(model="worker"),
    )


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
        ModelSettings(model="small"),
        ModelSettings(model="large", reasoning={"effort": "low"}),
        ModelSettings(model="small"),
    ]
    with patch("examples.swe_smith.hierarchical.config.select_subagent_settings", side_effect=settings) as choose:
        result = Orchestrator(config, root, worker).run("Original issue")
    assert choose.call_count == 3
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
            reply("Cannot delegate"),
        ]
    )
    result = Orchestrator(config, client, Client([])).run("Issue")
    assert not (Path(smith.TESTBED) / "unexpected").exists()
    assert result.assignments == []
    assert "Available tools" in client.requests[1]["messages"][-1]["content"]
    assert "Invalid arguments" in client.requests[2]["messages"][-1]["content"]
    assert "Unknown agent" in client.requests[3]["messages"][-1]["content"]


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


def test_worker_retries_unparsed_tool_call(config):
    client = Client([reply("Attempting Bash", finish="tool_calls"), reply("Done")])
    worker = SubAgent(AgentSpec(name="fix_1", profile="fix", system_prompt="Fix"), client, config)
    result = worker.execute("Issue", TaskAssignment(agent_name="fix_1", prompt="Fix"), config.subagent_model)

    assert "Retry one tool call" in client.requests[1]["messages"][-1]["content"]
    assert result.summary == "Done"


@pytest.mark.parametrize("actor", ["orchestrator", "worker"])
def test_length_cutoff_never_executes_tools(config, actor):
    client = Client([reply(tool="bash", arguments={"command": "touch unexpected"}, finish="length")])
    if actor == "orchestrator":
        result = Orchestrator(config, client, Client([])).run("Issue").orchestrator
    else:
        worker = SubAgent(AgentSpec(name="fix_1", profile="fix", system_prompt="Fix"), client, config)
        result = worker.execute("Issue", TaskAssignment(agent_name="fix_1", prompt="Fix"), config.subagent_model)
    assert result.status == "failed"
    assert result.usage.total_tokens == 120
    assert not (Path(smith.TESTBED) / "unexpected").exists()


def test_api_failure_keeps_command_evidence_and_tokens(config):
    error = APIError("offline", httpx.Request("POST", "https://workers.invalid"), body=None)
    client = Client([reply(tool="bash", arguments={"command": "printf partial > partial.py && cat partial.py"}), error])
    worker = SubAgent(AgentSpec(name="fix_1", profile="fix", system_prompt="Fix"), client, config)
    result = worker.execute("Issue", TaskAssignment(agent_name="fix_1", prompt="Fix"), config.subagent_model)
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
    with patch("examples.swe_smith.hierarchical.config.select_subagent_settings", side_effect=ValueError("router bug")):
        with pytest.raises(ValueError, match="router bug"):
            Orchestrator(config, root, Client([])).run("Issue")


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
    monkeypatch.setenv("AGL_KEY", "secret-gateway-value")
    monkeypatch.setenv("JUDGE_API_KEY", "secret-judge-value")
    output, rc = smith._run("env", 2)
    assert rc == 0
    assert "secret-router-value" not in output
    assert "secret-gateway-value" not in output
    assert "secret-judge-value" not in output
    _, rc = smith._run("(sleep 0.3; touch orphan) & wait", 0.03)
    assert rc == 124
    time.sleep(0.4)
    assert not (Path(smith.TESTBED) / "orphan").exists()


@pytest.mark.parametrize("fatal", [True, False])
def test_reward_retries_are_finite(fatal):
    error = (
        urllib.error.HTTPError("https://events.invalid", 401, "Unauthorized", {}, None)
        if fatal
        else urllib.error.URLError("offline")
    )
    with (
        patch.dict("os.environ", {"AGL_EVENT_URL": "https://events.invalid"}),
        patch.object(smith.urllib.request, "urlopen", side_effect=error) as send,
        patch.object(smith.time, "sleep"),
    ):
        smith.post_event("reward", {"value": 0}, retry=True)
    assert send.call_count == (1 if fatal else 4)


@pytest.mark.parametrize(
    "router_enabled,feedback_mode,fixes_bug,judge_valid",
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
    config.oracle_feedback = feedback_mode
    config.judge_base_url = "https://judge.invalid/v1"
    config.judge_api_key = "judge-only"
    config.judge_model = ModelSettings(model="judge-model", temperature=0, max_tokens=512)
    semantic = router_enabled and feedback_mode == "semantic"
    if router_enabled:
        config.oracle_models = ["test/small", "test/large"]
        config.oracle_checkpoint = str(tmp_path / "router.pkl")
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
            reply(json.dumps({"score": score, "reason": "Evidence supports the assigned work"}) if judge_valid else "bad")
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
            assert replies["judge.invalid"] == []  # Scores exist before Lightning can start backward.
        if router_enabled:
            assert LinUCB.load(config.oracle_checkpoint).n_update == int(not semantic or judge_valid)

    monkeypatch.setenv("AGL_TASK_INPUT", "Fix addition")
    monkeypatch.setenv("AGL_EVAL_META", json.dumps(task))
    monkeypatch.setenv("HIER_OUTPUT_DIR", str(tmp_path / "results"))
    with (
        patch.object(run, "load_config", return_value=config),
        patch.object(run, "OpenAI", side_effect=make_client),
        patch(
            "examples.swe_smith.hierarchical.config.select_subagent_settings", wraps=select_subagent_settings
        ) as choose,
        patch.object(smith, "post_event", side_effect=publish) as send,
    ):
        assert run.main() == 0
    assert choose.call_count == 1
    assert choose.call_args.kwargs["problem"] == "Fix addition"
    saved_text = (tmp_path / "results/fixture-instance.json").read_text()
    saved = json.loads(saved_text)
    assert saved["reward"] == int(fixes_bug)
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
        assert isinstance(choose.call_args.kwargs["router"], ModelRouter)
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
        assert LinUCB.load(config.oracle_checkpoint).n_update == int(not semantic or judge_valid)
    else:
        assert choose.call_args.kwargs["router"] is None
        assert "oracle" not in saved


def test_unfixed_source_is_not_rewarded_and_pytest_cannot_be_shadowed(task):
    smith.checkout_bug_commit(task["instance_id"])
    smith.relocate_git()
    (Path(smith.TESTBED) / "pytest.py").write_text("print('PASSED test_calc.py::test_add')\n")
    reward, resolved, _, _ = smith.evaluate(task, 20)
    assert reward == 0
    assert not resolved


def test_env_config_keeps_worker_credentials_separate(monkeypatch):
    with patch.dict(
        "os.environ",
        {
            "AGL_OPENAI_BASE_URL": "https://gateway.invalid/v1",
            "AGL_KEY": "gateway",
            "OPENROUTER_API_KEY": "router",
            "SUBAGENT_REASONING": '{"effort":"low"}',
            "ORACLE_MODELS": '["test/small", "test/large"]',
            "ORACLE_LEARN": "false",
        },
        clear=True,
    ):
        config = load_config()
    assert config.orchestrator_api_key == "gateway"
    assert config.subagent_api_key == "router"
    assert config.subagent_model.reasoning == {"effort": "low"}
    assert config.orchestrator_model.reasoning is None
    assert config.oracle_models == ["test/small", "test/large"]
    assert config.oracle_learn is False


def test_oracle_routes_each_assignment_then_learns_from_delayed_score(config, tmp_path):
    config.oracle_models = ["test/small", "test/large"]
    config.oracle_checkpoint = str(tmp_path / "router.pkl")
    config.subagent_model.reasoning = {"effort": "low"}
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
    assert len(router.report()["decisions"]) == 2
    assert len({d["program_id"] for d in router.report()["decisions"]}) == 2
    assert [d["agent_name"] for d in router.report()["decisions"]] == ["fix_1", "fix_1"]
    assert [d["model"] for d in router.report()["decisions"]] == [r.model for r in swarm.assignments]
    assert all(r["extra_body"] == {"reasoning": {"effort": "low"}} for r in worker.requests)
    assert not np.array_equal(router.choices[0][0].features, router.choices[1][0].features)
    assert router.selector.n_update == 0
    assert LinUCB.load(config.oracle_checkpoint).n_update == 0

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
    for ctx, d in router.choices:
        expected_b[resumed.selector.index[d["model"]]] += d["reward"] * ctx.features
    np.testing.assert_allclose(resumed.selector.b, expected_b)


@pytest.mark.parametrize("validation_url", [False, True])
def test_oracle_frozen_evaluation_does_not_change_checkpoint(config, tmp_path, validation_url):
    config.oracle_models = ["test/small", "test/large"]
    config.oracle_checkpoint = str(tmp_path / "router.pkl")
    LinUCB(config.oracle_models).save(config.oracle_checkpoint)
    before = Path(config.oracle_checkpoint).read_bytes()
    if validation_url:
        config.orchestrator_base_url = "https://gateway.invalid/mode/val/rollout/1/v1"
    else:
        config.oracle_learn = False
    router = ModelRouter(config)
    spec = AgentSpec(name="fix_1", profile="fix", system_prompt="Fix")
    settings = router.select("Issue", spec, TaskAssignment(agent_name=spec.name, prompt="Fix"), config.subagent_model)
    router.feedback([1], [TaskResult(agent_name=spec.name, model=settings.model)])
    assert router.selector.n_update == 0
    assert router.report()["decisions"][0]["learned"] is False
    assert Path(config.oracle_checkpoint).read_bytes() == before
    assert not Path(config.oracle_checkpoint).with_suffix(".lock").exists()


def test_oracle_shared_checkpoint_keeps_updates_from_other_episodes(config, tmp_path):
    config.oracle_models = ["test/small", "test/large"]
    config.oracle_checkpoint = str(tmp_path / "router.pkl")
    routers = [ModelRouter(config) for _ in range(4)]

    def episode(router):
        spec = AgentSpec(name="fix_1", profile="fix", system_prompt="Fix")
        settings = router.select("Issue", spec, TaskAssignment(agent_name=spec.name, prompt="Fix"), config.subagent_model)
        result = TaskResult(agent_name=spec.name, model=settings.model, usage=TokenUsage(completion_tokens=100))
        router.feedback([1], [result])

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(episode, routers))
    learned = LinUCB.load(config.oracle_checkpoint)
    assert learned.n_select == 4
    assert learned.n_update == 4
    assert learned.pulls.sum() == 4
    x = routers[0].choices[0][0].features
    np.testing.assert_allclose(learned.A.sum(axis=0), 2 * np.eye(64) + 4 * np.outer(x, x))
    np.testing.assert_allclose(learned.b.sum(axis=0), 4 * (0.8 - 0.2 * 100 / 10000) * x)


def test_oracle_rejects_checkpoint_from_another_model_pool(config, tmp_path):
    config.oracle_checkpoint = str(tmp_path / "router.pkl")
    LinUCB(["old/model"]).save(config.oracle_checkpoint)
    config.oracle_models = ["test/small", "test/large"]
    with pytest.raises(ValueError, match="checkpoint pool"):
        ModelRouter(config)


@pytest.fixture
def assignment_result():
    return TaskResult(
        agent_name="worker_identity",
        model="worker_model_identity",
        assignment=TaskAssignment(agent_name="worker_identity", prompt="Find the arithmetic bug", context="Look at calc"),
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
    config.oracle_models = ["test/small"]
    config.oracle_checkpoint = str(tmp_path / "router.pkl")
    router = ModelRouter(config)
    spec = AgentSpec(name="fix_1", profile="fix", system_prompt="Fix")
    results = []
    for prompt in ["First", "Second", "Third"]:
        assignment = TaskAssignment(agent_name=spec.name, prompt=prompt)
        settings = router.select("Issue", spec, assignment, config.subagent_model)
        results.append(TaskResult(agent_name=spec.name, model=settings.model, assignment=assignment))
    before = Path(config.oracle_checkpoint).read_bytes()
    router.feedback([None, None, None], results)
    assert Path(config.oracle_checkpoint).read_bytes() == before
    assert router.selector.n_update == 0
    router.feedback([1, None, 0], results)
    assert router.selector.n_update == 2
    assert [d["learned"] for d in router.report()["decisions"]] == [True, False, True]
    assert [d["reward"] for d in router.report()["decisions"]] == [0.8, None, 0]
    assert router.report()["score_source"] == "semantic_judge"


def test_assignment_patches_are_captured_before_later_workers_change_source(task, config, tmp_path):
    config.oracle_models = ["test/small"]
    config.oracle_checkpoint = str(tmp_path / "router.pkl")
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
    assert "chat_template" not in local.actor_rollout_ref.rollout.engine_kwargs.vllm
    assert other.actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser == "hermes"


def test_training_smoke_keeps_lora_settings(tmp_path):
    pytest.importorskip("verl")
    from omegaconf import OmegaConf

    from examples.swe_smith.train_smith_agent import DEFAULT_CONFIG, build_config

    settings = OmegaConf.load(DEFAULT_CONFIG)
    settings.actor_rollout_ref.model.lora_rank = 16
    config_path = tmp_path / "experiment.yaml"
    OmegaConf.save(settings, config_path)
    config = build_config(config_path=config_path, smoke=True)

    assert config.actor_rollout_ref.model.lora_rank == 16
    assert config.actor_rollout_ref.model.lora.merge is True
    assert config.actor_rollout_ref.rollout.n == 4
    assert config.trainer.total_training_steps == 1
    assert config.data.train_files == str(tmp_path / "smoke_train.jsonl")
    assert config.agentlightning.trace_aggregator.trajectory_max_response_length == config.data.max_response_length
    assert "agents" not in config
