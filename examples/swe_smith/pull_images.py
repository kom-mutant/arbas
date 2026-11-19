#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.

"""Prepare SWE-smith images for Kubernetes rollouts as ``<image_name>:arbas``."""

from __future__ import annotations

import argparse
import contextlib
import json
import shlex
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASETS = (
    EXAMPLE_DIR / "train.jsonl",
    EXAMPLE_DIR / "val.jsonl",
)
ROLLOUT_IMAGE_TAG = "arbas"
OPENAI_PACKAGE = "openai"


def select_distinct_instances(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("max_train_instances must be positive")
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        selected.setdefault(row["instance_id"], row)
        if len(selected) == count:
            return list(selected.values())
    raise ValueError(f"Need {count} distinct training tasks, found {len(selected)}")


@dataclass(frozen=True)
class ImagePreparationResult:
    source_image: str
    target_image: str
    status: str
    error: str | None = None


def load_dataset(path: Path) -> list[dict[str, Any]]:
    items = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
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


def openai_image_name(image_name: str) -> str:
    if not image_name:
        raise ValueError("image_name is required")
    if "@" in image_name:
        raise ValueError("digest-based image names are not supported")
    if image_name.rfind(":") > image_name.rfind("/"):
        raise ValueError(
            f"SWE-smith image_name must be untagged because job-template-openai.yaml "
            f"uses '<image_name>:{ROLLOUT_IMAGE_TAG}': {image_name}"
        )
    return f"{image_name}:{ROLLOUT_IMAGE_TAG}"


def image_exists(client: Any, image: str) -> bool:
    try:
        return client.images.get(image) is not None
    except Exception:
        return False


def openai_dockerfile(source_image: str) -> str:
    install_cmd = (
        f"python -m pip install --no-cache-dir {shlex.quote(OPENAI_PACKAGE)} 'pydantic>=2.10,<3' 'numpy>=1.23'"
    )
    return (
        f"FROM {source_image}\n"
        'LABEL agentlightning.swe-smith.openai-layer="1"\n'
        f"RUN bash -lc {shlex.quote(install_cmd)}\n"
    )


def build_openai_image(client: Any, source_image: str, target_image: str) -> None:
    dockerfile = openai_dockerfile(source_image)
    with TemporaryDirectory(prefix="swe-smith-openai-") as context_dir:
        (Path(context_dir) / "Dockerfile").write_text(dockerfile, encoding="utf-8")
        client.images.build(path=context_dir, tag=target_image, rm=True, forcerm=True, pull=False)


def prepare_one(
    client: Any,
    source_image: str,
    *,
    pull_missing: bool,
    force_build: bool,
) -> ImagePreparationResult:
    try:
        target_image = openai_image_name(source_image)
        if not force_build and image_exists(client, target_image):
            return ImagePreparationResult(source_image, target_image, "exists")

        if image_exists(client, source_image):
            source_status = "base-local"
        elif pull_missing:
            client.images.pull(source_image)
            source_status = "base-pulled"
        else:
            return ImagePreparationResult(
                source_image,
                target_image,
                "failed",
                "base image is not loaded in the active Docker daemon; run with "
                "minikube docker-env active or pass --pull-missing",
            )

        status = "rebuilt" if force_build else "built"
        build_openai_image(client, source_image, target_image)
        return ImagePreparationResult(source_image, target_image, f"{source_status},{status}")
    except Exception as exc:
        target = f"{source_image}:{ROLLOUT_IMAGE_TAG}"
        with contextlib.suppress(Exception):
            target = openai_image_name(source_image)
        return ImagePreparationResult(source_image, target, "failed", str(exc))


def _load_config_for_images(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    cfg = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected mapping in {resolved}")
    data = cfg.get("data") or {}
    for field in ("train_files", "val_files"):
        if data.get(field):
            data[field] = str(resolved.parent / Path(data[field]).expanduser())
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SWE-smith per-repo Docker images")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--dataset", action="append", default=None, help="SWE-smith JSONL path (repeatable).")
    selection.add_argument("--image", action="append", help="Prepare an image directly (repeatable).")
    selection.add_argument("--config", type=Path, help="Select images using an experiment YAML.")
    parser.add_argument("--list", action="store_true", help="Print selected images as JSON without preparing them.")
    parser.add_argument("--limit", type=int, default=None, help="Only consider the first N distinct instances.")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--pull-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pull a base image only if missing locally (default: true).",
    )
    parser.add_argument(
        "--force-build",
        action="store_true",
        help=f"Rebuild :{ROLLOUT_IMAGE_TAG} images even when the target tag already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        images = experiment_images(_load_config_for_images(args.config))
    elif args.image:
        images = list(dict.fromkeys(args.image))
    else:
        dataset_paths = [Path(path) for path in (args.dataset or [str(p) for p in DEFAULT_DATASETS])]
        rows = [item for path in dataset_paths for item in load_dataset(path)]
        images = distinct_images(rows, args.limit)
    if args.list:
        print(json.dumps(images))
        return

    try:
        import docker
    except ImportError as exc:
        raise SystemExit("pull_images.py requires the Docker SDK for Python: pip install docker") from exc

    print(f"Preparing local :{ROLLOUT_IMAGE_TAG} images for job-template-openai.yaml")
    for image in images:
        print(f"  {image} -> {openai_image_name(image)}")

    client = docker.from_env()
    failed: list[ImagePreparationResult] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(prepare_one, client, img, pull_missing=args.pull_missing, force_build=args.force_build): img
            for img in images
        }
        for future in as_completed(futures):
            result = future.result()
            if result.error:
                print(f"  FAILED {result.source_image} -> {result.target_image}: {result.error}")
                failed.append(result)
            else:
                print(f"  {result.status.upper():18} {result.source_image} -> {result.target_image}")

    if failed:
        names = ", ".join(result.source_image for result in failed)
        raise SystemExit(f"Failed to prepare SWE-smith images: {names}")
    print(f"Prepared {len(images)} SWE-smith :{ROLLOUT_IMAGE_TAG} images.")


if __name__ == "__main__":
    main()
