import os
import re
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .types import ModelSettings

_MEMORY_UNITS = {"b": 1, "k": 1024, "m": 1024**2, "g": 1024**3}


def parse_memory_bytes(spec: str) -> int:
    """Parse a Docker memory size like ``6g`` or ``512m`` (also accepts ``6Gi``) into bytes."""
    raw = spec.strip().lower()
    if raw.endswith(("ki", "mi", "gi")):
        raw = raw[:-1]
    match = re.fullmatch(r"([1-9]\d*)([bkmg])?", raw)
    if match is None:
        raise ValueError(f"invalid memory limit {spec!r}; expected e.g. '6g' or '512m'")
    return int(match.group(1)) * _MEMORY_UNITS[match.group(2) or "b"]


def normalize_memory_limit(spec: str) -> str:
    """Normalize a memory limit into canonical Docker CLI syntax (e.g. ``'6Gi' -> '6g'``)."""
    raw = spec.strip().lower()
    if raw.endswith(("ki", "mi", "gi")):
        raw = raw[:-1]
    parse_memory_bytes(raw)
    return raw


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

    max_workers: int = Field(ge=1)
    cpus: float = Field(gt=0)
    memory: str
    cmd_timeout: int = Field(gt=0)
    eval_timeout: int = Field(gt=0)
    api_timeout: float = Field(gt=0)
    gateway_wait_s: float = Field(ge=0)
    obs_char_cap: int = Field(gt=0)
    f2p_only: bool

    @field_validator("memory")
    @classmethod
    def _check_memory(cls, value: str) -> str:
        return normalize_memory_limit(value)


class RewardSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    partial_credit: bool = False
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
    mode: Literal["train", "val", "inference"] = "inference"
    orchestrator_base_url: str
    orchestrator_api_key: str = Field(repr=False)
    subagent_api_key: str = Field(repr=False)
    judge_api_key: str | None = Field(default=None, repr=False)


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
        mode=os.environ.get("AGL_ROLLOUT_MODE") or "inference",  # type: ignore[arg-type]
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
