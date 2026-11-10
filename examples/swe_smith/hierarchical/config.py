import json
import os
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from .types import AgentSpec, ModelSettings, TaskAssignment

if TYPE_CHECKING:
    from .routing import ModelRouter


class Config(BaseModel):
    orchestrator_base_url: str
    orchestrator_api_key: str = Field(repr=False)
    subagent_base_url: str
    subagent_api_key: str = Field(repr=False)
    orchestrator_model: ModelSettings
    subagent_model: ModelSettings
    orchestrator_steps: int = 40
    subagent_steps: int = 20
    command_timeout: int = 120
    obs_char_cap: int = 10000
    api_timeout: float = 120
    gateway_wait_s: float = 600
    oracle_models: list[str] = Field(default_factory=list)
    oracle_checkpoint: str = "/tmp/hierarchical-oracle/linucb.pkl"
    oracle_cost_weight: float = 0.2
    oracle_cost_ref: float = 10000
    oracle_learn: bool = True
    oracle_seed: int = 0
    oracle_feedback: Literal["semantic", "episode"] = "semantic"
    judge_base_url: str | None = None
    judge_api_key: str | None = Field(default=None, repr=False)
    judge_model: ModelSettings | None = None

    def judge_settings(self) -> ModelSettings:
        if self.judge_model:
            return self.judge_model
        fallback_model = (
            self.subagent_model.model if self.orchestrator_model.model == "auto" else self.orchestrator_model.model
        )
        return ModelSettings(model=fallback_model, temperature=0, max_tokens=1024)


def select_subagent_settings(
    agent: AgentSpec,
    assignment: TaskAssignment,
    defaults: ModelSettings,
    *,
    problem: str,
    router: "ModelRouter | None" = None,
) -> ModelSettings:
    """Choose settings for one assignment, using ORACLE when enabled."""
    if router is None:
        return defaults
    return router.select(problem, agent, assignment, defaults)


ENV_OVERRIDES = {
    "orchestrator_steps": "SMITH_MAX_TURNS",
    "subagent_steps": "SUBAGENT_MAX_STEPS",
    "command_timeout": "SMITH_CMD_TIMEOUT",
    "obs_char_cap": "SMITH_OBS_CHAR_CAP",
    "api_timeout": "HIER_API_TIMEOUT",
    "gateway_wait_s": "SMITH_GATEWAY_WAIT_S",
    "oracle_checkpoint": "ORACLE_CHECKPOINT",
    "oracle_cost_weight": "ORACLE_COST_WEIGHT",
    "oracle_cost_ref": "ORACLE_COST_REF",
    "oracle_learn": "ORACLE_LEARN",
    "oracle_seed": "ORACLE_SEED",
    "oracle_feedback": "ORACLE_FEEDBACK",
    "judge_base_url": "JUDGE_BASE_URL",
    "judge_api_key": "JUDGE_API_KEY",
}


def load_config() -> Config:
    overrides = {field: os.environ[env] for field, env in ENV_OVERRIDES.items() if env in os.environ}
    direct = os.environ.get("ORCHESTRATOR_BASE_URL")
    config = Config(
        orchestrator_base_url=direct or os.environ["AGL_OPENAI_BASE_URL"],
        orchestrator_api_key=(
            os.environ["ORCHESTRATOR_API_KEY"]
            if direct
            else os.environ.get("AGL_KEY") or os.environ.get("OPENAI_API_KEY", "dummy")
        ),
        subagent_base_url=os.environ.get("SUBAGENT_BASE_URL", "https://openrouter.ai/api/v1"),
        subagent_api_key=os.environ.get("SUBAGENT_API_KEY") or os.environ["OPENROUTER_API_KEY"],
        orchestrator_model=ModelSettings(
            model=os.environ.get("ORCHESTRATOR_MODEL", "auto"),
            temperature=os.environ.get("ORCHESTRATOR_TEMPERATURE", "1.0"),
            max_tokens=os.environ.get("AGL_MAX_TOKENS", "16384"),
            reasoning=json.loads(os.environ.get("ORCHESTRATOR_REASONING", "null")),
        ),
        subagent_model=ModelSettings(
            model=os.environ.get("SUBAGENT_MODEL", "qwen/qwen3.5-9b"),
            max_tokens=os.environ.get("SUBAGENT_MAX_TOKENS", "4096"),
            temperature=os.environ.get("SUBAGENT_TEMPERATURE", "0.7"),
            reasoning=json.loads(os.environ.get("SUBAGENT_REASONING", "null")),
        ),
        **overrides,
        oracle_models=json.loads(os.environ.get("ORACLE_MODELS", "[]")),
    )
    defaults = config.judge_settings()
    config.judge_model = ModelSettings(
        model=os.environ.get("JUDGE_MODEL") or defaults.model,
        temperature=defaults.temperature,
        max_tokens=int(os.environ.get("JUDGE_MAX_TOKENS", defaults.max_tokens)),
        reasoning=json.loads(os.environ.get("JUDGE_REASONING", "null")),
    )
    return config
