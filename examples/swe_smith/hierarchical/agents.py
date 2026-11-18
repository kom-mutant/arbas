from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING, Any

from openai import APIError, OpenAI, RateLimitError
from pydantic import ValidationError

from . import smith
from .config import Config
from .prompts import ORCHESTRATOR_PROMPT, PROFILE_PROMPTS, SUBAGENT_PROMPT
from .types import (
    AgentRunResult,
    AgentSpec,
    AssignmentResult,
    AssignTaskArgs,
    BashArgs,
    CreateSubagentArgs,
    ModelSettings,
    Profile,
    SwarmResult,
    TaskAssignment,
)

if TYPE_CHECKING:
    from .oracle import ModelRouter

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
TOOL_RETRY = "Your tool call could not be read. Retry one tool call with its required arguments."


def _query(
    client: OpenAI,
    settings: ModelSettings,
    result: AgentRunResult,
    tools: list[dict[str, Any]],
    gateway_wait: float = 0,
) -> Any:
    started = time.monotonic()
    while True:
        try:
            response = client.chat.completions.create(
                messages=result.messages,
                **({"tools": tools, "parallel_tool_calls": False} if tools else {}),
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


def _tool_reply(result: AgentRunResult, call: Any, output: str) -> None:
    result.messages.append({"role": "tool", "tool_call_id": call.id, "content": output})


def _tool_error(result: AgentRunResult, kind: str) -> None:
    result.tool_errors[kind] = result.tool_errors.get(kind, 0) + 1


def _unparsed_call(choice: Any, tool_names: tuple[str, ...]) -> bool:
    if choice.finish_reason == "tool_calls":
        return True
    content = choice.message.content or ""
    if "<tool_call>" in content or "<function=" in content:
        return True
    visible = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    if "</think>" in visible:
        visible = visible.split("</think>")[-1]
    names = "|".join(re.escape(name) for name in tool_names)
    return bool(re.search(rf"(?m)(?:^|\s)(?:{names})\s*\(", visible))


class SubAgent:
    def __init__(self, spec: AgentSpec, client: OpenAI, config: Config) -> None:
        self.spec = spec
        self.client = client
        self.config = config

    def execute(self, problem: str, assignment: TaskAssignment, settings: ModelSettings) -> AssignmentResult:
        started = time.perf_counter()
        result = AssignmentResult(
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
        max_steps = self.config.subagent.max_steps
        for step in range(max_steps):
            reporting = step == max_steps - 1
            if reporting:
                result.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your action budget is exhausted. Return your final report now, without tool calls. "
                            "Summarize findings, file changes, checks actually run and unfinished work "
                            "from this conversation."
                        ),
                    }
                )
            try:
                choice = _query(self.client, settings, result, [] if reporting else [BASH_TOOL])
            except APIError as exc:
                result.status, result.error = "failed", str(exc)
                break

            if choice.finish_reason not in ("stop", "tool_calls"):
                result.status, result.error = "failed", f"Incomplete response: {choice.finish_reason}"
                break

            message = choice.message
            if reporting and message.tool_calls:
                _tool_error(result, "unexpected_call")
                result.error = "Tool call returned during final report"
                break
            if message.tool_calls:
                observation = self._execute_tool_calls(result, message.tool_calls) or observation
            elif _unparsed_call(choice, ("bash",)):
                _tool_error(result, "unrecognized_call")
                result.messages.append({"role": "user", "content": TOOL_RETRY})
            elif (message.content or "").strip():
                result.status, result.summary = ("step_limit" if reporting else "completed"), message.content
                result.duration_s = time.perf_counter() - started
                return result
            else:
                result.messages.append({"role": "user", "content": "Provide a nonempty final report or use Bash."})

        result.summary = f"Stopped: {result.error or 'step limit reached'}.\n{observation}"
        result.duration_s = time.perf_counter() - started
        return result

    def _execute_tool_calls(self, result: AssignmentResult, calls: list[Any]) -> str | None:
        if len(calls) != 1:
            _tool_error(result, "multiple_calls")
            for call in calls:
                _tool_reply(result, call, "Call one tool at a time.")
            return None

        call = calls[0]
        if call.function.name != "bash":
            _tool_error(result, "unknown_tool")
            _tool_reply(result, call, "Only Bash is available.")
            return None

        try:
            args = BashArgs.model_validate_json(call.function.arguments)
        except ValidationError as exc:
            _tool_error(result, "invalid_arguments")
            _tool_reply(result, call, f"Invalid arguments: {exc}")
            return None

        reason = smith._forbidden_action(args.command)
        if reason:
            output, rc = reason, 126
        else:
            output, rc = smith._run(args.command, self.config.sandbox.cmd_timeout)
        if rc != 0:
            _tool_error(result, "command_failed")
        observation = f"Command: {args.command}\n" + smith.render_observation(
            rc, output, self.config.sandbox.obs_char_cap
        )
        _tool_reply(result, call, observation)
        return observation


class Orchestrator:
    def __init__(
        self,
        config: Config,
        client: OpenAI,
        subagent_client: OpenAI,
        router: ModelRouter | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.subagent_client = subagent_client
        self.router = router
        self.agents: dict[str, SubAgent] = {}
        self.assignments: list[AssignmentResult] = []

    def create_subagent(self, name: Profile) -> AgentSpec:
        spec = AgentSpec(name=f"{name}_{len(self.agents) + 1}", profile=name, system_prompt=PROFILE_PROMPTS[name])
        self.agents[spec.name] = SubAgent(spec, self.subagent_client, self.config)
        return spec

    def assign_task(self, agent: str, prompt: str, context: str = "") -> AssignmentResult:
        worker = self.agents[agent]
        assignment = TaskAssignment(agent_name=agent, prompt=prompt, context=context)
        settings = (
            self.router.select(self.problem, worker.spec, assignment, self.config.subagent)
            if self.router is not None
            else self.config.subagent
        )
        capture_evidence = self.router is not None and self.config.oracle.feedback == "semantic"
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
        result = AgentRunResult(
            agent_name="orchestrator",
            model=self.config.orchestrator.model,
            messages=[
                {"role": "system", "content": ORCHESTRATOR_PROMPT},
                {"role": "user", "content": problem},
            ],
        )
        for _ in range(self.config.orchestrator.max_turns):
            try:
                choice = _query(
                    self.client,
                    self.config.orchestrator,
                    result,
                    ORCHESTRATOR_TOOLS,
                    self.config.sandbox.gateway_wait_s,
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
            elif _unparsed_call(choice, tuple(ORCHESTRATOR_SCHEMAS)):
                _tool_error(result, "unrecognized_call")
                result.messages.append({"role": "user", "content": TOOL_RETRY})
            elif (message.content or "").strip():
                result.status, result.summary = "completed", message.content
                result.duration_s = time.perf_counter() - started
                return SwarmResult(orchestrator=result, assignments=self.assignments)
            else:
                result.messages.append({"role": "user", "content": "Continue delegating or provide a final answer."})

        result.summary = f"Stopped: {result.error or 'step limit reached'}."
        result.duration_s = time.perf_counter() - started
        return SwarmResult(orchestrator=result, assignments=self.assignments)

    def _execute_tool_calls(self, result: AgentRunResult, calls: list[Any]) -> None:
        if len(calls) != 1:
            _tool_error(result, "multiple_calls")
            for call in calls:
                _tool_reply(result, call, "Call one tool at a time and read its result.")
            return
        call = calls[0]
        if call.function.name not in ORCHESTRATOR_SCHEMAS:
            _tool_error(result, "unknown_tool")
            _tool_reply(result, call, "Available tools: create_subagent and assign_task.")
            return
        try:
            args = ORCHESTRATOR_SCHEMAS[call.function.name][0].model_validate_json(call.function.arguments)
        except ValidationError as exc:
            _tool_error(result, "invalid_arguments")
            _tool_reply(result, call, f"Invalid arguments: {exc}")
            return
        if isinstance(args, CreateSubagentArgs):
            spec = self.create_subagent(args.name)
            output: dict[str, Any] = {"agent_name": spec.name, "profile": spec.profile}
        elif args.agent not in self.agents:
            _tool_error(result, "unknown_agent")
            available = ", ".join(self.agents) or "none yet"
            output = {"error": f"Unknown agent {args.agent!r}. Use the exact agent_name. Available: {available}."}
        else:
            report = self.assign_task(args.agent, args.prompt, args.context)
            output = {
                "agent_name": report.agent_name,
                "status": report.status,
                "summary": report.summary,
                "error": report.error,
            }
        _tool_reply(result, call, json.dumps(output, ensure_ascii=False))
