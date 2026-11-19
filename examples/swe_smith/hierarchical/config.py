import os
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from .types import ModelSettings


class OrchestratorSettings(ModelSettings):
    model_config = ConfigDict(extra="forbid")

    max_turns: int = Field(ge=1)


class SubagentSettings(ModelSettings):
    model_config = ConfigDict(extra="forbid")

    base_url: str
    max_steps: int = Field(ge=2)


class JudgeSettings(ModelSettings):
    model_config = ConfigDict(extra="forbid")

    base_url: str = ""


class SandboxSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cmd_timeout: int = Field(gt=0)
    eval_timeout: int = Field(gt=0)
    api_timeout: float = Field(gt=0)
    gateway_wait_s: float = Field(ge=0)
    obs_char_cap: int = Field(gt=0)
    f2p_only: bool


class RewardSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    len_pen_t0: int = Field(ge=0)
    len_pen_lambda: float = Field(ge=0.0, le=1.0)
    prompt_pen_soft_start: int = Field(ge=0)
    prompt_pen_hard_cap: int = Field(gt=0)
    prompt_pen_max: float = Field(ge=0.0, le=1.0)


class OracleSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[str]
    checkpoint: str
    cost_weight: float = Field(ge=0.0, le=1.0)
    cost_ref: float = Field(gt=0)
    learn: bool
    seed: int
    feedback: Literal["semantic", "episode"]


class AgentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    orchestrator: OrchestratorSettings
    subagent: SubagentSettings
    sandbox: SandboxSettings
    reward: RewardSettings
    oracle: OracleSettings
    judge: JudgeSettings


class Config(AgentSettings):
    orchestrator_base_url: str
    orchestrator_api_key: str = Field(repr=False)
    subagent_api_key: str = Field(repr=False)
    judge_api_key: str | None = Field(default=None, repr=False)

    @property
    def mode(self) -> Literal["train", "val", "inference"]:
        if "/mode/train/" in self.orchestrator_base_url:
            return "train"
        if "/mode/val/" in self.orchestrator_base_url:
            return "val"
        return "inference"


def load_config() -> Config:
    raw = os.environ.get("ARBAS_AGENT_CONFIG")
    if not raw:
        raise ValueError("ARBAS_AGENT_CONFIG environment variable is required")
    settings = AgentSettings.model_validate_json(raw)
    direct = os.environ.get("ORCHESTRATOR_BASE_URL")
    provider_key = {
        "openrouter.ai": "OPENROUTER_API_KEY",
        "polza.ai": "POLZA_API_KEY",
    }.get(urlsplit(settings.subagent.base_url).hostname)
    subagent_api_key = os.environ.get("SUBAGENT_API_KEY")
    if provider_key is not None:
        subagent_api_key = subagent_api_key or os.environ.get(provider_key)
        if not subagent_api_key:
            raise ValueError(f"Set {provider_key} or SUBAGENT_API_KEY for the subagent endpoint.")
    else:
        subagent_api_key = subagent_api_key or os.environ.get("POLZA_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    return Config(
        orchestrator_base_url=direct or os.environ["AGL_OPENAI_BASE_URL"],
        orchestrator_api_key=(
            os.environ["ORCHESTRATOR_API_KEY"]
            if direct
            else os.environ.get("AGL_KEY") or os.environ.get("OPENAI_API_KEY", "dummy")
        ),
        subagent_api_key=subagent_api_key or "",
        judge_api_key=os.environ.get("JUDGE_API_KEY"),
        **settings.model_dump(),
    )
