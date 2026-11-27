from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Profile = Literal["research", "fix", "verify"]


class ModelSettings(BaseModel):
    model: str
    temperature: float
    max_tokens: int
    reasoning: dict[str, Any] | None = None

    def request_kwargs(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
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
            prompt = usage.prompt_tokens or 0
            completion = usage.completion_tokens or 0
            total = usage.total_tokens or (prompt + completion)
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.total_tokens += total
            self.max_prompt_tokens = max(self.max_prompt_tokens, prompt)


class Judgment(BaseModel):
    usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str
    score: float | None = None
    reason: str = ""
    error: str | None = None
    response: str = ""
    latency_s: float | None = None


class AgentRunResult(BaseModel):
    agent_name: str
    model: str
    summary: str = ""
    status: Literal["completed", "failed", "step_limit"] = "step_limit"
    error: str | None = None
    steps_taken: int = 0
    duration_s: float = 0.0
    usage: TokenUsage = Field(default_factory=TokenUsage)
    tool_errors: dict[str, int] = Field(default_factory=dict)
    messages: list[dict[str, Any]] = Field(default_factory=list)


class AssignmentResult(AgentRunResult):
    assignment: TaskAssignment | None = None
    patch_before: str | None = None
    patch_after: str | None = None
    judgment: Judgment | None = None


class SwarmResult(BaseModel):
    orchestrator: AgentRunResult
    assignments: list[AssignmentResult]


class AssignTaskArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: Profile = Field(description="Worker role: research, fix, or verify.")
    prompt: str = Field(
        min_length=1,
        description="Required non-empty task instruction for the worker.",
    )
    context: str = Field(
        default="",
        description="Optional findings, file paths, or prior changes the worker needs.",
    )

    @field_validator("prompt")
    @classmethod
    def _require_non_blank_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must be a non-empty task instruction")
        return value


class BashArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str


class SemanticScore(BaseModel):
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    reason: str
