from __future__ import annotations

import argparse
import importlib.resources
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

if __package__:
    from .hierarchical.config import AgentSettings
    from .pull_images import select_distinct_instances
else:
    from hierarchical.config import AgentSettings
    from pull_images import select_distinct_instances

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


def load_experiment_config(config_path: Path) -> DictConfig:
    resolved = config_path.expanduser().resolve()
    config = OmegaConf.load(resolved)
    if not isinstance(config, DictConfig):
        raise ValueError(f"Expected mapping in {resolved}")
    base_file = config.pop("base", None)
    if base_file:
        candidate = (resolved.parent / Path(base_file).expanduser()).resolve()
        base_path = (
            candidate if candidate.is_file() else (EXAMPLE_DIR / Path(base_file).expanduser()).resolve()
        )
        if not base_path.is_file():
            raise FileNotFoundError(f"Base config not found: {candidate} (or {base_path})")
        base_cfg = OmegaConf.load(base_path)
        if "base" in base_cfg:
            raise ValueError(f"Base config {base_path} cannot include another base config.")
        config = OmegaConf.merge(base_cfg, config)
    data = config.get("data")
    if data is not None:
        for field in ("train_files", "val_files"):
            if data.get(field):
                data[field] = str(resolved.parent / Path(str(data[field])).expanduser())
    return config


def build_agent_config(
    config_path: Path,
    run_name: str,
    config_overrides: Sequence[str] = (),
) -> str:
    experiment = load_experiment_config(config_path)
    if config_overrides:
        experiment = OmegaConf.merge(experiment, OmegaConf.from_dotlist(list(config_overrides)))
    agents = OmegaConf.to_container(experiment.agents, resolve=True)
    if not isinstance(agents, dict):
        raise ValueError("agents section in config must be a mapping")
    oracle = agents.get("oracle")
    if isinstance(oracle, dict) and oracle.get("checkpoint") == "auto":
        oracle["checkpoint"] = f"/agl-logs/{run_name}/oracle/linucb.pkl"
    settings = AgentSettings.model_validate(agents)
    return settings.model_dump_json()


def _project(instance: dict[str, Any], agent_config_json: str, run_name: str) -> dict[str, Any]:
    row = {field: instance.get(field) for field in INSTANCE_FIELDS}
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
    with Path(path).open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    selected = [_project(row, agent_config_json, run_name) for row in rows]
    if max_instances:
        selected = selected[:max_instances]
    if not selected:
        raise ValueError(f"No instances loaded from {path}")
    return selected


def build_config(
    *,
    config_path: Path = DEFAULT_CONFIG,
    agl_base_url: str | None = None,
    agl_key: str | None = None,
    run_name: str | None = None,
    config_overrides: Sequence[str] = (),
) -> DictConfig:
    verl_pkg = importlib.resources.files("agentlightning.verl")
    with initialize_config_dir(config_dir=str(verl_pkg), version_base=None):
        base_cfg = compose(config_name="config")

    overrides = load_experiment_config(config_path)
    experiment_name = overrides.pop("experiment", {}).get("name", base_cfg.trainer.experiment_name)
    overrides.pop("agents", None)
    base_cfg.agentlightning.k8s.job_template_path = str(EXAMPLE_DIR / "job-template-openai.yaml")
    base_cfg.agentlightning.local.agent_class = "examples.swe_smith.hierarchical.docker_runner:DockerRunner"
    base_cfg.agentlightning.local.env_map = {
        "IMAGE_NAME": "input.image_name",
        "AGL_TASK_INPUT": "input.problem_statement",
        "ARBAS_AGENT_CONFIG": "input.arbas_agent_config",
        "AGL_RUN_NAME": "input.agl_run_name",
        "DATA_ID": "input.data_id",
    }

    if agl_base_url:
        OmegaConf.update(overrides, "agentlightning.agl_base_url", agl_base_url, merge=True)
    if agl_key is not None:
        OmegaConf.update(overrides, "agentlightning.agl_key", agl_key, merge=True)

    cli_override_conf = OmegaConf.from_dotlist(list(config_overrides))
    cli_override_conf.pop("experiment", None)
    cli_override_conf.pop("agents", None)

    rollout_mode = OmegaConf.select(
        cli_override_conf,
        "actor_rollout_ref.rollout.mode",
        default=OmegaConf.select(
            overrides, "actor_rollout_ref.rollout.mode", default=base_cfg.actor_rollout_ref.rollout.mode
        ),
    )
    model_path = OmegaConf.select(
        cli_override_conf,
        "actor_rollout_ref.model.path",
        default=OmegaConf.select(
            overrides, "actor_rollout_ref.model.path", default=base_cfg.actor_rollout_ref.model.path
        ),
    )
    model_config = Path(model_path) / "config.json"
    model_type = json.loads(model_config.read_text()).get("model_type") if model_config.is_file() else None
    uses_qwen35 = model_type in {"qwen3_5", "qwen3_5_moe"} if model_type else "qwen3.5" in model_path.lower()
    if uses_qwen35:
        defaults = {
            "tool_call_parser": "qwen3_coder",
            "reasoning_parser": "qwen3",
            "chat_template": str(EXAMPLE_DIR / "qwen35_chat_template.jinja"),
        }
    else:
        defaults = {"tool_call_parser": "hermes", "chat_template": str(EXAMPLE_DIR / "swe_smith_chat_template.jinja")}
    existing_vllm = OmegaConf.select(overrides, "actor_rollout_ref.rollout.engine_kwargs.vllm") or OmegaConf.create({})
    OmegaConf.update(
        overrides, "actor_rollout_ref.rollout.engine_kwargs.vllm", OmegaConf.merge(defaults, existing_vllm), merge=True
    )

    experiment = OmegaConf.select(overrides, "trainer.experiment_name") or experiment_name
    exp_full_name = f"{experiment}_{rollout_mode}_{Path(model_path).name}_{TRAIN_BACKEND}"
    if run_name:
        exp_full_name += f"_{run_name}"
    OmegaConf.update(overrides, "trainer.experiment_name", exp_full_name, merge=True)

    OmegaConf.set_struct(base_cfg, False)
    config = OmegaConf.merge(base_cfg, overrides, cli_override_conf)
    OmegaConf.set_struct(config, False)
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
    from ray._private.utils import validate_socket_filepath
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

    ray_tmpdir = os.environ.get("RAY_TMPDIR")
    if ray_tmpdir:
        session_dir = f"session_2000-01-01_00-00-00_000000_{os.getpid()}"
        socket_path = Path(ray_tmpdir).absolute() / session_dir / "sockets" / "plasma_store"
        try:
            validate_socket_filepath(str(socket_path))
        except OSError as exc:
            raise RuntimeError("Ray socket path is too long. Use a shorter AGL_TMPDIR or RAY_TMPDIR.") from exc

    env_path = "ray_kwargs.ray_init.runtime_env.env_vars"
    defaults = get_ppo_ray_runtime_env()["env_vars"]
    # Agent Lightning replaces the whole env_vars map if the config supplies one.
    env_vars = OmegaConf.merge(defaults, OmegaConf.select(config, env_path))
    if ipc_dir := os.environ.get("VERL_WEIGHT_TRANSFER_IPC_DIR"):
        env_vars.VERL_WEIGHT_TRANSFER_IPC_DIR = ipc_dir
    OmegaConf.update(config, env_path, env_vars)


def train(config: DictConfig, *, agent_config_json: str, run_name: str) -> None:
    runtime_env = config.ray_kwargs.ray_init.runtime_env.env_vars
    for name in ("MAX_JOBS", "VLLM_USE_FLASHINFER_MOE_FP16"):
        os.environ[name] = str(runtime_env[name])

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
    agent_config_json = build_agent_config(args.config, run_name, config_overrides)
    config = build_config(
        config_path=args.config,
        agl_base_url=args.agl_base_url,
        agl_key=args.agl_key,
        run_name=run_name,
        config_overrides=config_overrides,
    )
    train(config, agent_config_json=agent_config_json, run_name=run_name)


if __name__ == "__main__":
    main()
