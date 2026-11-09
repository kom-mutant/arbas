from typing import Any, Literal

from pydantic import BaseModel, Field

Profile = Literal["research", "fix", "verify"]


class ModelSettings(BaseModel):
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    reasoning: dict[str, Any] | None = None

    def request_kwargs(self) -> dict[str, Any]:
        params = self.model_dump(exclude={"reasoning"})
        if self.reasoning is not None:
            params["extra_body"] = {"reasoning": self.reasoning}
        return params


class AgentSpec(BaseModel):
    name: str
    profile: Profile
    system_prompt: str


class TaskAssignment(BaseModel):
    agent_name: str
    prompt: str
    context: str = ""


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    max_prompt_tokens: int = 0

    def add(self, usage: Any) -> None:
        if usage is not None:
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
            self.total_tokens += usage.total_tokens
            self.max_prompt_tokens = max(self.max_prompt_tokens, usage.prompt_tokens)


class Judgment(BaseModel):
    model: str
    score: float | None = None
    reason: str = ""
    error: str | None = None
    response: str = ""
    latency_s: float | None = None


class TaskResult(BaseModel):
    agent_name: str
    model: str
    summary: str = ""
    status: Literal["completed", "failed", "step_limit"] = "step_limit"
    error: str | None = None
    steps_taken: int = 0
    duration_s: float = 0.0
    usage: TokenUsage = Field(default_factory=TokenUsage)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    assignment: TaskAssignment | None = None
    patch_before: str | None = None
    patch_after: str | None = None
    judgment: Judgment | None = None


class SwarmResult(BaseModel):
    orchestrator: TaskResult
    assignments: list[TaskResult]


class CreateSubagentArgs(BaseModel):
    name: Profile


class AssignTaskArgs(BaseModel):
    agent: str
    prompt: str
    context: str = ""


class BashArgs(BaseModel):
    command: str


class SemanticScore(BaseModel):
    score: float = Field(ge=0, le=1)
    reason: str