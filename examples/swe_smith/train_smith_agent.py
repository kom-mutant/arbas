
from __future__ import annotations

import argparse
import importlib.resources
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

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


def _project(instance: dict[str, Any]) -> dict[str, Any]:
    row = {field: instance.get(field) for field in INSTANCE_FIELDS}
    row["data_source"] = DATA_SOURCE
    row["data_id"] = str(instance.get("instance_id", ""))
    return row


def load_split_file(
    path: str,
    *,
    max_instances: int | None = None,
) -> list[dict[str, Any]]:
    """Load a pre-split, pre-curated JSONL dataset and project to the VERL schema."""
    with Path(path).open() as file:
        rows = [json.loads(line) for line in file if line.strip()]
    selected = [_project(row) for row in rows]
    if max_instances:
        selected = selected[:max_instances]
    if not selected:
        raise ValueError(f"No instances loaded from {path}")
    return selected


def build_config(
    *,
    config_path: Path = DEFAULT_CONFIG,
    smoke: bool = False,
    model: str | None = None,
    agl_base_url: str | None = None,
    agl_key: str | None = None,
    run_name: str | None = None,
    config_overrides: Sequence[str] = (),
) -> DictConfig:
    verl_pkg = importlib.resources.files("agentlightning.verl")
    with initialize_config_dir(config_dir=str(verl_pkg), version_base=None):
        base_cfg = compose(config_name="config")

    overrides = OmegaConf.load(config_path)
    smoke_overrides = overrides.pop("smoke", {})
    if smoke:
        overrides = OmegaConf.merge(overrides, smoke_overrides)
    overrides.pop("agents", None)
    for key in ("train_files", "val_files"):
        overrides.data[key] = str(config_path.resolve().parent / Path(overrides.data[key]).expanduser())
    base_cfg.agentlightning.k8s.job_template_path = str(EXAMPLE_DIR / "job-template-openai.yaml")

    if model:
        overrides.actor_rollout_ref.model.path = model
    if agl_base_url:
        overrides.agentlightning.agl_base_url = agl_base_url
    if agl_key is not None:
        overrides.agentlightning.agl_key = agl_key

    cli_override_conf = OmegaConf.from_dotlist(list(config_overrides))
    rollout_mode = OmegaConf.select(
        cli_override_conf, "actor_rollout_ref.rollout.mode", default=overrides.actor_rollout_ref.rollout.mode
    )
    model_path = OmegaConf.select(
        cli_override_conf, "actor_rollout_ref.model.path", default=overrides.actor_rollout_ref.model.path
    )
    model_config = Path(model_path) / "config.json"
    model_type = json.loads(model_config.read_text()).get("model_type") if model_config.is_file() else None
    uses_qwen35 = model_type in {"qwen3_5", "qwen3_5_moe"} if model_type else "qwen3.5" in model_path.lower()
    engine = overrides.actor_rollout_ref.rollout.engine_kwargs
    if uses_qwen35:
        defaults = {"tool_call_parser": "qwen3_coder", "reasoning_parser": "qwen3"}
    else:
        defaults = {"tool_call_parser": "hermes", "chat_template": str(EXAMPLE_DIR / "swe_smith_chat_template.jinja")}
    engine.vllm = OmegaConf.merge(defaults, engine.vllm)

    experiment = overrides.trainer.experiment_name
    overrides.trainer.experiment_name = f"{experiment}_{rollout_mode}_{Path(model_path).name}_{TRAIN_BACKEND}"
    if run_name:
        overrides.trainer.experiment_name += f"_{run_name}"

    OmegaConf.set_struct(base_cfg, False)
    config = OmegaConf.merge(base_cfg, overrides, cli_override_conf)
    OmegaConf.set_struct(config, False)
    return config


def train(
    config: DictConfig,
    *,
    train_dataset_path: str | None = None,
    val_dataset_path: str | None = None,
    max_val_instances: int | None = None,
) -> None:
    from agentlightning.verl.entrypoint import run_ppo

    if not config.agentlightning.agl_key:
        raise RuntimeError("AGL_KEY is required")

    train_dataset_path = train_dataset_path or config.data.train_files
    val_dataset_path = val_dataset_path or config.data.val_files
    train_dataset = load_split_file(train_dataset_path)
    val_dataset = load_split_file(val_dataset_path, max_instances=max_val_instances)
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

    log("\n=== Start VERL training ===")
    run_ppo(config=config, train_dataset=train_dataset, val_dataset=val_dataset)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train a SWE-smith agent with VERL/GRPO via Agent Lightning")
    parser.add_argument("--config", type=Path, default=os.environ.get("AGL_TRAIN_CONFIG", DEFAULT_CONFIG))
    parser.add_argument("--smoke", action="store_true", help="Apply the smoke settings from the training config.")
    parser.add_argument(
        "--train-dataset-path",
        default=os.environ.get("AGL_TRAIN_DATASET_PATH"),
        help="Pre-split training JSONL, used as-is.",
    )
    parser.add_argument(
        "--val-dataset-path",
        default=os.environ.get("AGL_VAL_DATASET_PATH"),
        help="Pre-split validation JSONL, used as-is. Pairs with --train-dataset-path.",
    )
    parser.add_argument(
        "--max-val-instances",
        type=int,
        default=None,
        help="Optional cap on validation instances (default: all). Each validation eval "
        "runs ALL val instances at the test_freq cadence, so capping bounds eval time.",
    )

    parser.add_argument("--model", default=os.environ.get("AGL_MODEL_NAME"))
    parser.add_argument("--agl-base-url", default="http://localhost:8080")
    parser.add_argument("--agl-key", default=os.environ.get("AGL_KEY", ""))
    parser.add_argument("--run-name", default=os.environ.get("AGL_RUN_NAME"))
    args, config_overrides = parser.parse_known_args()
    return args, config_overrides


def main() -> None:
    args, config_overrides = parse_args()
    config = build_config(
        config_path=args.config,
        smoke=args.smoke,
        model=args.model,
        agl_base_url=args.agl_base_url,
        agl_key=args.agl_key,
        run_name=args.run_name or ("smoke" if args.smoke else None),
        config_overrides=config_overrides,
    )
    train(
        config,
        train_dataset_path=args.train_dataset_path,
        val_dataset_path=args.val_dataset_path,
        max_val_instances=args.max_val_instances,
    )


if __name__ == "__main__":
    main()
