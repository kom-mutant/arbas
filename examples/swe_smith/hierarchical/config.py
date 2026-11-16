import json
import os
from typing import TYPE_CHECKING, Any, Literal, Mapping

from pydantic import BaseModel, Field

from .types import AgentSpec, ModelSettings, TaskAssignment

if TYPE_CHECKING:
    from .routing import ModelRouter


class AgentSettings(BaseModel):
    subagent_base_url: str
    orchestrator_model: ModelSettings
    subagent_model: ModelSettings
    orchestrator_steps: int
    subagent_steps: int = Field(ge=2)
    command_timeout: int
    obs_char_cap: int
    api_timeout: float
    gateway_wait_s: float
    eval_timeout: int
    f2p_only: bool
    length_penalty_start: int
    length_penalty_weight: float
    prompt_penalty_start: int
    prompt_penalty_cap: int
    prompt_penalty_max: float
    oracle_models: list[str]
    oracle_checkpoint: str
    oracle_cost_weight: float
    oracle_cost_ref: float
    oracle_learn: bool
    oracle_seed: int
    oracle_feedback: Literal["semantic", "episode"]
    judge_base_url: str
    judge_model: ModelSettings


class Config(AgentSettings):
    orchestrator_base_url: str
    orchestrator_api_key: str = Field(repr=False)
    subagent_api_key: str = Field(repr=False)
    judge_api_key: str | None = Field(default=None, repr=False)


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


AGENT_SETTINGS = {
    "orchestrator_steps": "SMITH_MAX_TURNS",
    "subagent_steps": "SUBAGENT_MAX_STEPS",
    "command_timeout": "SMITH_CMD_TIMEOUT",
    "obs_char_cap": "SMITH_OBS_CHAR_CAP",
    "api_timeout": "HIER_API_TIMEOUT",
    "gateway_wait_s": "SMITH_GATEWAY_WAIT_S",
    "eval_timeout": "SMITH_EVAL_TIMEOUT",
    "f2p_only": "SMITH_F2P_ONLY",
    "length_penalty_start": "SMITH_LEN_PEN_T0",
    "length_penalty_weight": "SMITH_LEN_PEN_LAMBDA",
    "prompt_penalty_start": "SMITH_PROMPT_PEN_SOFT_START",
    "prompt_penalty_cap": "SMITH_PROMPT_PEN_HARD_CAP",
    "prompt_penalty_max": "SMITH_PROMPT_PEN_MAX",
    "oracle_checkpoint": "ORACLE_CHECKPOINT",
    "oracle_cost_weight": "ORACLE_COST_WEIGHT",
    "oracle_cost_ref": "ORACLE_COST_REF",
    "oracle_learn": "ORACLE_LEARN",
    "oracle_seed": "ORACLE_SEED",
    "oracle_feedback": "ORACLE_FEEDBACK",
    "judge_base_url": "JUDGE_BASE_URL",
}

MODEL_SETTINGS = {
    "orchestrator_model": {
        "model": "ORCHESTRATOR_MODEL",
        "temperature": "ORCHESTRATOR_TEMPERATURE",
        "max_tokens": "AGL_MAX_TOKENS",
        "reasoning": "ORCHESTRATOR_REASONING",
    },
    "subagent_model": {
        "model": "SUBAGENT_MODEL",
        "temperature": "SUBAGENT_TEMPERATURE",
        "max_tokens": "SUBAGENT_MAX_TOKENS",
        "reasoning": "SUBAGENT_REASONING",
    },
    "judge_model": {
        "model": "JUDGE_MODEL",
        "temperature": "JUDGE_TEMPERATURE",
        "max_tokens": "JUDGE_MAX_TOKENS",
        "reasoning": "JUDGE_REASONING",
    },
}

def validate_agent_settings(settings: Mapping[str, Any]) -> None:
    required = {"SUBAGENT_BASE_URL", "ORACLE_MODELS", *AGENT_SETTINGS.values()}
    required.update(key for model in MODEL_SETTINGS.values() for key in model.values())
    missing = sorted(required - settings.keys())
    if missing:
        raise ValueError(f"Missing required settings: {', '.join(missing)}. Set them in agents in training.yaml.")


def parse_agent_settings(env: Mapping[str, Any]) -> AgentSettings:
    validate_agent_settings(env)
    settings = {field: env[name] for field, name in AGENT_SETTINGS.items()}
    models = {}
    for name, fields in MODEL_SETTINGS.items():
        values = {field: env[key] for field, key in fields.items()}
        if isinstance(values["reasoning"], str):
            values["reasoning"] = json.loads(values["reasoning"])
        models[name] = ModelSettings(**values)
    oracle_models = env["ORACLE_MODELS"]
    if isinstance(oracle_models, str):
        oracle_models = json.loads(oracle_models)
    return AgentSettings(subagent_base_url=env["SUBAGENT_BASE_URL"], oracle_models=oracle_models, **settings, **models)


def load_config() -> Config:
    settings = parse_agent_settings(os.environ)
    direct = os.environ.get("ORCHESTRATOR_BASE_URL")
    return Config(
        orchestrator_base_url=direct or os.environ["AGL_OPENAI_BASE_URL"],
        orchestrator_api_key=(
            os.environ["ORCHESTRATOR_API_KEY"]
            if direct
            else os.environ.get("AGL_KEY") or os.environ.get("OPENAI_API_KEY", "dummy")
        ),
        subagent_api_key=os.environ.get("SUBAGENT_API_KEY") or os.environ["OPENROUTER_API_KEY"],
        judge_api_key=os.environ.get("JUDGE_API_KEY"),
        **settings.model_dump(),
    )
