"""Load SWE-smith experiment settings and select their datasets and images."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

EXAMPLE_DIR = Path(__file__).resolve().parent


def load_experiment_config(config_path: Path) -> DictConfig:
    resolved = config_path.expanduser().resolve()
    config = OmegaConf.load(resolved)
    if not isinstance(config, DictConfig):
        raise ValueError(f"Expected mapping in {resolved}")
    raw_override_data = config.get("data")
    if raw_override_data is not None and not isinstance(raw_override_data, Mapping):
        raise ValueError(f"Expected mapping in data section of {resolved}")
    override_data = raw_override_data or {}
    base_file = config.pop("base", None)
    base_path = resolved
    if base_file:
        candidate = (resolved.parent / Path(base_file).expanduser()).resolve()
        base_path = candidate if candidate.is_file() else (EXAMPLE_DIR / Path(base_file).expanduser()).resolve()
        if not base_path.is_file():
            raise FileNotFoundError(f"Base config not found: {candidate} (or {base_path})")
        base_cfg = OmegaConf.load(base_path)
        if not isinstance(base_cfg, DictConfig):
            raise ValueError(f"Expected mapping in {base_path}")
        if "base" in base_cfg:
            raise ValueError(f"Base config {base_path} cannot include another base config.")
        config = OmegaConf.merge(base_cfg, config)
    data = config.get("data")
    if data is not None:
        if not isinstance(data, Mapping):
            raise ValueError(f"Expected mapping in data section of {resolved}")
        for field in ("train_files", "val_files"):
            if data.get(field):
                source_dir = resolved.parent if field in override_data else base_path.parent
                data[field] = str(source_dir / Path(str(data[field])).expanduser())
    return config


def select_distinct_instances(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("max_train_instances must be positive")
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        selected.setdefault(row["instance_id"], row)
        if len(selected) == count:
            return list(selected.values())
    raise ValueError(f"Need {count} distinct training tasks, found {len(selected)}")


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        items = [json.loads(line) for line in file if line.strip()]
    if not items:
        raise ValueError(f"Dataset is empty: {path}")
    return items


def distinct_images(dataset: list[dict[str, Any]], limit: int | None) -> list[str]:
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        dataset = select_distinct_instances(dataset, limit)
    images: list[str] = []
    for item in dataset:
        image = item.get("image_name")
        if not image:
            raise ValueError(f"Instance {item.get('instance_id')} missing 'image_name'")
        if image not in images:
            images.append(image)
    return images


def experiment_images(config: Mapping[str, Any]) -> list[str]:
    data = config["data"]
    images = distinct_images(load_dataset(Path(data["train_files"])), data.get("max_train_instances"))
    trainer = config.get("trainer") or {}
    if trainer.get("val_before_train", True) or (trainer.get("test_freq") or -1) > 0:
        images = list(dict.fromkeys(images + distinct_images(load_dataset(Path(data["val_files"])), None)))
    return images
