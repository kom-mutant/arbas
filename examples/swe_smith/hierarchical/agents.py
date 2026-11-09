import json
import time
from typing import TYPE_CHECKING, Any

from openai import APIError, OpenAI, RateLimitError
from pydantic import ValidationError

from . import config as configuration
from . import smith
from .config import Config
from .prompts import ORCHESTRATOR_PROMPT, PROFILE_PROMPTS, SUBAGENT_PROMPT
from .types import (
    AgentSpec,
    AssignTaskArgs,
    BashArgs,
    CreateSubagentArgs,
    ModelSettings,
    SwarmResult,
    TaskAssignment,
    TaskResult,
)

if TYPE_CHECKING:
    from .routing import ModelRouter

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a command in the shared task repository.",
        "parameters": BashArgs.model_json_schema(),
    },
}
ORCHESTRATOR_SCHEMAS = {
    "create_subagent": (CreateSubagentArgs, "Create a worker from the expert catalog."),
    "assign_task": (AssignTaskArgs, "Assign work and wait for the worker's report."),
}
ORCHESTRATOR_TOOLS = [
    {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": schema.model_json_schema()},
    }
    for name, (schema, description) in ORCHESTRATOR_SCHEMAS.items()
]


def _query(client: OpenAI, settings: ModelSettings, result: TaskResult, tools: list[dict], gateway_wait: float = 0):
    started = time.monotonic()
    while True:
        try:
            response = client.chat.completions.create(
                messages=result.messages,
                tools=tools,
                parallel_tool_calls=False,
                **settings.request_kwargs(),
            )
            break
        except RateLimitError as exc:
            remaining = gateway_wait - (time.monotonic() - started)
            if "gateway paused" not in exc.message.lower() or remaining <= 0:
                raise
            time.sleep(min(5, remaining))
    result.usage.add(response.usage)
    result.steps_taken += 1
    choice = response.choices[0]
    result.messages.append(
        choice.message.model_dump(
            mode="json",
            exclude_none=True,
            include={"role", "content", "tool_calls", "reasoning", "reasoning_content", "reasoning_details"},
        )
    )
    return choice


def _tool_reply(result: TaskResult, call: Any, output: str) -> None:
    result.messages.append({"role": "tool", "tool_call_id": call.id, "content": output})


class SubAgent:
    def __init__(self, spec: AgentSpec, client: OpenAI, config: Config) -> None:
        self.spec = spec
        self.client = client
        self.config = config

    def execute(self, problem: str, assignment: TaskAssignment, settings: ModelSettings) -> TaskResult:
        started = time.perf_counter()
        result = TaskResult(
            agent_name=self.spec.name,
            model=settings.model,
            assignment=assignment,
            messages=[
                {"role": "system", "content": SUBAGENT_PROMPT + "\n" + self.spec.system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "original_issue": problem,
                            "assignment": assignment.prompt,
                            "context": assignment.context,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        observation = "No commands executed."
        for _ in range(self.config.subagent_steps):
            try:
                choice = _query(self.client, settings, result, [BASH_TOOL])
            except APIError as exc:
                result.status, result.error = "failed", str(exc)
                break

            if choice.finish_reason not in ("stop", "tool_calls"):
                result.status, result.error = "failed", f"Incomplete response: {choice.finish_reason}"
                break

            message = choice.message
            if message.tool_calls:
                observation = self._execute_tool_calls(result, message.tool_calls) or observation
            elif (message.content or "").strip():
                result.status, result.summary = "completed", message.content
                result.duration_s = time.perf_counter() - started
                return result
            else:
                result.messages.append({"role": "user", "content": "Provide a nonempty final report or use Bash."})

        result.summary = f"Stopped: {result.error or 'step limit reached'}.\n{observation}"
        result.duration_s = time.perf_counter() - started
        return result

    def _execute_tool_calls(self, result: TaskResult, calls: list) -> str | None:
        if len(calls) != 1:
            for call in calls:
                _tool_reply(result, call, "Call one tool at a time.")
            return None

        call = calls[0]
        if call.function.name != "bash":
            _tool_reply(result, call, "Only Bash is available.")
            return None

        try:
            args = BashArgs.model_validate_json(call.function.arguments)
        except ValidationError as exc:
            _tool_reply(result, call, f"Invalid arguments: {exc}")
            return None

        reason = smith._forbidden_action(args.command)
        if reason:
            output, rc = reason, 126
        else:
            output, rc = smith._run(args.command, self.config.command_timeout)
        observation = f"Command: {args.command}\n" + smith.render_observation(rc, output, self.config.obs_char_cap)
        _tool_reply(result, call, observation)
        return observation


class Orchestrator:
    def __init__(
        self, config: Config, client: OpenAI, subagent_client: OpenAI, router: "ModelRouter | None" = None
    ) -> None:
        self.config = config
        self.client = client
        self.subagent_client = subagent_client
        self.router = router
        self.agents: dict[str, SubAgent] = {}
        self.assignments: list[TaskResult] = []

    def create_subagent(self, name: str) -> AgentSpec:
        spec = AgentSpec(name=f"{name}_{len(self.agents) + 1}", profile=name, system_prompt=PROFILE_PROMPTS[name])
        self.agents[spec.name] = SubAgent(spec, self.subagent_client, self.config)
        return spec

    def assign_task(self, agent: str, prompt: str, context: str = "") -> TaskResult:
        worker = self.agents[agent]
        assignment = TaskAssignment(agent_name=agent, prompt=prompt, context=context)
        settings = configuration.select_subagent_settings(
            worker.spec,
            assignment,
            self.config.subagent_model,
            problem=self.problem,
            router=self.router,
        )
        capture_evidence = self.router is not None and self.config.oracle_feedback == "semantic"
        patch_before = smith.capture_patch() if capture_evidence else None
        result = worker.execute(self.problem, assignment, settings)
        if capture_evidence:
            result.patch_before = patch_before
            result.patch_after = smith.capture_patch()
        self.assignments.append(result)
        return result

    def run(self, problem: str) -> SwarmResult:
        started = time.perf_counter()
        self.problem = problem
        result = TaskResult(
            agent_name="orchestrator",
            model=self.config.orchestrator_model.model,
            messages=[
                {"role": "system", "content": ORCHESTRATOR_PROMPT},
                {"role": "user", "content": problem},
            ],
        )
        for _ in range(self.config.orchestrator_steps):
            try:
                choice = _query(
                    self.client, self.config.orchestrator_model, result, ORCHESTRATOR_TOOLS, self.config.gateway_wait_s
                )
            except APIError as exc:
                result.status, result.error = "failed", str(exc)
                break

            if choice.finish_reason not in ("stop", "tool_calls"):
                result.status, result.error = "failed", f"Incomplete response: {choice.finish_reason}"
                break

            message = choice.message
            if message.tool_calls:
                self._execute_tool_calls(result, message.tool_calls)
            elif (message.content or "").strip():
                result.status, result.summary = "completed", message.content
                result.duration_s = time.perf_counter() - started
                return SwarmResult(orchestrator=result, assignments=self.assignments)
            else:
                result.messages.append({"role": "user", "content": "Continue delegating or provide a final answer."})

        result.summary = f"Stopped: {result.error or 'step limit reached'}."
        result.duration_s = time.perf_counter() - started
        return SwarmResult(orchestrator=result, assignments=self.assignments)

    def _execute_tool_calls(self, result: TaskResult, calls: list) -> None:
        if len(calls) != 1:
            for call in calls:
                _tool_reply(result, call, "Call one tool at a time and read its result.")
            return
        call = calls[0]
        if call.function.name not in ORCHESTRATOR_SCHEMAS:
            _tool_reply(result, call, "Available tools: create_subagent and assign_task.")
            return
        try:
            args = ORCHESTRATOR_SCHEMAS[call.function.name][0].model_validate_json(call.function.arguments)
        except ValidationError as exc:
            _tool_reply(result, call, f"Invalid arguments: {exc}")
            return
        if isinstance(args, CreateSubagentArgs):
            spec = self.create_subagent(args.name)
            output = {"agent_name": spec.name, "profile": spec.profile}
        elif args.agent not in self.agents:
            output = {"error": f"Unknown agent {args.agent!r}; create it first."}
        else:
            report = self.assign_task(args.agent, args.prompt, args.context)
            output = {
                "agent_name": report.agent_name,
                "status": report.status,
                "summary": report.summary,
                "error": report.error,
            }
        _tool_reply(result, call, json.dumps(output, ensure_ascii=False))
