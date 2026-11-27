from __future__ import annotations

import argparse
import copy
import importlib.resources
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from .experiment import load_dataset, load_experiment_config, select_distinct_instances
from .hierarchical.config import AgentSettings

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXAMPLE_DIR / "training.yaml"
DATA_SOURCE = "swe_smith"
TRAIN_BACKEND = "fsdp"

INSTANCE_FIELDS = (
    "instance_id",
    "problem_statement",
    "image_name",
    "repo",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
)


def log(message: str) -> None:
    print(message, flush=True)


def resolve_experiment(config_source: Path | DictConfig, config_overrides: Sequence[str] = ()) -> DictConfig:
    experiment = (
        copy.deepcopy(config_source) if isinstance(config_source, DictConfig) else load_experiment_config(config_source)
    )
    if config_overrides:
        experiment = OmegaConf.merge(experiment, OmegaConf.from_dotlist(list(config_overrides)))
    return experiment


def build_agent_config(config_path: Path | DictConfig, config_overrides: Sequence[str] = ()) -> str:
    experiment = resolve_experiment(config_path, config_overrides)
    agents = OmegaConf.to_container(experiment.agents, resolve=True)
    if not isinstance(agents, dict):
        raise ValueError("agents section in config must be a mapping")
    return AgentSettings.model_validate(agents).model_dump_json()


def _project(instance: dict[str, Any], agent_config_json: str, run_name: str) -> dict[str, Any]:
    row = {field: instance.get(field) for field in INSTANCE_FIELDS}
    row["eval_meta"] = json.dumps(
        {
            "instance_id": instance.get("instance_id"),
            "repo": instance.get("repo"),
            "FAIL_TO_PASS": instance.get("FAIL_TO_PASS") or [],
            "PASS_TO_PASS": instance.get("PASS_TO_PASS") or [],
        },
        ensure_ascii=False,
    )
    row["arbas_agent_config"] = agent_config_json
    row["agl_run_name"] = run_name
    row["data_source"] = DATA_SOURCE
    row["data_id"] = str(instance.get("instance_id", ""))
    return row


def load_split_file(
    path: str,
    *,
    max_instances: int | None = None,
    agent_config_json: str,
    run_name: str,
) -> list[dict[str, Any]]:
    """Load a pre-split JSONL dataset and project to the VERL schema."""
    selected = [_project(row, agent_config_json, run_name) for row in load_dataset(Path(path))]
    if max_instances:
        selected = selected[:max_instances]
    return selected


def build_config(
    *,
    config_path: Path | DictConfig = DEFAULT_CONFIG,
    agl_base_url: str | None = None,
    agl_key: str | None = None,
    run_name: str | None = None,
    config_overrides: Sequence[str] = (),
) -> DictConfig:
    verl_pkg = importlib.resources.files("agentlightning.verl")
    with initialize_config_dir(config_dir=str(verl_pkg), version_base=None):
        base_cfg = compose(config_name="config")

    overrides = resolve_experiment(config_path, config_overrides)
    experiment_section = overrides.pop("experiment", None)
    overrides.pop("agents", None)
    experiment_name = (
        (experiment_section.get("name") if isinstance(experiment_section, Mapping) else None)
        or OmegaConf.select(overrides, "trainer.experiment_name")
        or base_cfg.trainer.experiment_name
    )

    OmegaConf.set_struct(base_cfg, False)
    base_cfg.agentlightning.local.agent_class = "examples.swe_smith.hierarchical.run:RolloutWorker"
    base_cfg.agentlightning.local.env_map = {
        "IMAGE_NAME": "input.image_name",
        "AGL_TASK_INPUT": "input.problem_statement",
        "AGL_EVAL_META": "input.eval_meta",
        "ARBAS_AGENT_CONFIG": "input.arbas_agent_config",
        "AGL_RUN_NAME": "input.agl_run_name",
        "DATA_ID": "input.data_id",
    }
    config = OmegaConf.merge(base_cfg, overrides)
    OmegaConf.set_struct(config, False)

    if agl_base_url:
        config.agentlightning.agl_base_url = agl_base_url
    if agl_key is not None:
        config.agentlightning.agl_key = agl_key

    model_path = str(config.actor_rollout_ref.model.path)
    model_config = Path(model_path) / "config.json"
    model_type = json.loads(model_config.read_text()).get("model_type") if model_config.is_file() else None
    uses_qwen35 = model_type in {"qwen3_5", "qwen3_5_moe"} if model_type else "qwen3.5" in model_path.lower()
    vllm_defaults = (
        {
            "tool_call_parser": "qwen3_coder",
            "reasoning_parser": "qwen3",
            "chat_template": str(EXAMPLE_DIR / "qwen35_chat_template.jinja"),
        }
        if uses_qwen35
        else {
            "tool_call_parser": "hermes",
            "chat_template": str(EXAMPLE_DIR / "swe_smith_chat_template.jinja"),
        }
    )
    existing_vllm = OmegaConf.select(config, "actor_rollout_ref.rollout.engine_kwargs.vllm") or OmegaConf.create({})
    OmegaConf.update(
        config,
        "actor_rollout_ref.rollout.engine_kwargs.vllm",
        OmegaConf.merge(vllm_defaults, existing_vllm),
        merge=False,
    )

    rollout_mode = config.actor_rollout_ref.rollout.mode
    exp_full_name = f"{experiment_name}_{rollout_mode}_{Path(model_path).name}_{TRAIN_BACKEND}"
    if run_name:
        exp_full_name += f"_{run_name}"
    config.trainer.experiment_name = exp_full_name
    if os.environ.get("AGL_RUN_DIR"):
        config.trainer.default_local_dir = str(Path(os.environ["AGL_RUN_DIR"]) / "checkpoints")

    rollout = config.actor_rollout_ref.rollout
    if rollout.name != "vllm":
        raise ValueError("Training requires rollout.name=vllm.")
    if uses_qwen35 and OmegaConf.select(rollout, "engine_kwargs.vllm.gdn_prefill_backend") != "triton":
        raise ValueError("This vLLM profile requires gdn_prefill_backend=triton.")
    if OmegaConf.select(config, "ray_kwargs.ray_init.address") != "local":
        raise ValueError("Training requires its own local Ray instance.")
    env_vars = config.ray_kwargs.ray_init.runtime_env.env_vars
    if str(env_vars.MAX_JOBS) not in {"1", "2"}:
        raise ValueError("MAX_JOBS must be 1 or 2 for training on the shared GPU server.")
    env_vars.MAX_JOBS = str(env_vars.MAX_JOBS)
    return config


def prepare_ray(config: DictConfig) -> None:
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

    env_path = "ray_kwargs.ray_init.runtime_env.env_vars"
    defaults = get_ppo_ray_runtime_env()["env_vars"]
    # Agent Lightning replaces the whole env_vars map if the config supplies one.
    env_vars = OmegaConf.merge(defaults, OmegaConf.select(config, env_path))
    if ipc_dir := os.environ.get("VERL_WEIGHT_TRANSFER_IPC_DIR"):
        env_vars.VERL_WEIGHT_TRANSFER_IPC_DIR = ipc_dir
    if cpath := os.environ.get("CPATH"):
        env_vars.CPATH = cpath
    OmegaConf.update(config, env_path, env_vars)


def train(config: DictConfig, *, agent_config_json: str, run_name: str) -> None:
    runtime_env = config.ray_kwargs.ray_init.runtime_env.env_vars
    os.environ["MAX_JOBS"] = str(runtime_env.MAX_JOBS)

    import ray

    from agentlightning.verl.entrypoint import run_ppo

    if not config.agentlightning.agl_key:
        raise RuntimeError("AGL_KEY is required")
    if not os.environ.get("AGL_RUN_DIR"):
        raise RuntimeError("Start training with 'bash examples/swe_smith/run.sh train'.")

    train_dataset_path = config.data.train_files
    val_dataset_path = config.data.val_files
    train_dataset = load_split_file(train_dataset_path, agent_config_json=agent_config_json, run_name=run_name)
    count = config.data.get("max_train_instances")
    if count is not None:
        train_dataset = select_distinct_instances(train_dataset, count)
    val_dataset = load_split_file(val_dataset_path, agent_config_json=agent_config_json, run_name=run_name)
    instances = train_dataset + val_dataset
    distinct_repos = sorted({row["repo"] for row in instances})

    log("=== Preflight ===")
    log(f"  Agent Lightning: {config.agentlightning.agl_base_url}")
    log(f"  model:        {config.actor_rollout_ref.model.path}")
    log(f"  LoRA rank:    {config.actor_rollout_ref.model.lora_rank}")
    log(f"  train file:   {train_dataset_path}")
    log(f"  val file:     {val_dataset_path}")
    log(f"  instances:    {len(instances)}  (train {len(train_dataset)} / val {len(val_dataset)})")
    log(f"  distinct repos (images to prepare): {len(distinct_repos)}")
    gdn_prefill = OmegaConf.select(config, "actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend")
    log(f"  GDN prefill:  {gdn_prefill}")
    log(f"  MAX_JOBS:     {os.environ.get('MAX_JOBS', 'unset')}")

    log("\n=== Start VERL training ===")
    ray_was_initialized = ray.is_initialized()
    if not ray_was_initialized:
        prepare_ray(config)
    started = time.perf_counter()
    completed = False
    try:
        run_ppo(config=config, train_dataset=train_dataset, val_dataset=val_dataset)
        completed = True
    finally:
        log(f"VERL finished: completed={completed} trainer_wall_s={time.perf_counter() - started:.2f}")
        if not ray_was_initialized:
            ray.shutdown(wait_for_processes=True)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train a SWE-smith agent with VERL/GRPO via Agent Lightning")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--agl-base-url", default="http://localhost:8080")
    parser.add_argument("--agl-key", default=os.environ.get("AGL_KEY", ""))
    parser.add_argument("--run-name", default=os.environ.get("AGL_RUN_NAME"))
    return parser.parse_known_args()


def main() -> None:
    args, config_overrides = parse_args()
    run_name = args.run_name or "run"
    experiment = resolve_experiment(args.config, config_overrides)
    agent_config_json = build_agent_config(experiment)
    config = build_config(
        config_path=experiment,
        agl_base_url=args.agl_base_url,
        agl_key=args.agl_key,
        run_name=run_name,
    )
    train(config, agent_config_json=agent_config_json, run_name=run_name)


if __name__ == "__main__":
    main()
