#!/usr/bin/env python3

"""Pull the SWE-smith repository images that rollout sandboxes start from."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .experiment import distinct_images, experiment_images, load_dataset, load_experiment_config

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASETS = (
    EXAMPLE_DIR / "train.jsonl",
    EXAMPLE_DIR / "val.jsonl",
)
_DOCKER_SCRUBBED_ENV = {"OPENROUTER_API_KEY", "POLZA_API_KEY", "AGL_KEY"}


@dataclass(frozen=True)
class ImageResult:
    image: str
    status: str
    error: str | None = None


def _docker_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key not in _DOCKER_SCRUBBED_ENV}


def image_exists(image: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", image], capture_output=True, env=_docker_env()).returncode == 0


def prepare_one(image: str, *, pull_missing: bool) -> ImageResult:
    if image_exists(image):
        return ImageResult(image, "exists")
    if not pull_missing:
        return ImageResult(image, "failed", "image is not loaded in the active Docker daemon; pass --pull-missing")
    pulled = subprocess.run(["docker", "pull", image], capture_output=True, text=True, env=_docker_env())
    if pulled.returncode != 0:
        return ImageResult(image, "failed", (pulled.stderr or pulled.stdout).strip())
    return ImageResult(image, "pulled")


def prepare_images(images: Sequence[str], *, pull_missing: bool = True, max_workers: int = 4) -> None:
    unique = list(dict.fromkeys(str(image) for image in images))
    if not unique:
        return
    print("Preparing images for the current experiment", flush=True)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(lambda image: prepare_one(image, pull_missing=pull_missing), unique))
    for result in results:
        print(f"  {result.status.upper():7} {result.image}" + (f": {result.error}" if result.error else ""))
    failed = [result.image for result in results if result.error]
    if failed:
        raise SystemExit(f"Failed to prepare SWE-smith images: {', '.join(failed)}")
    print(f"Prepared {len(unique)} SWE-smith images.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull SWE-smith per-repo Docker images")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--dataset", action="append", default=None, help="SWE-smith JSONL path (repeatable).")
    selection.add_argument("--image", action="append", help="Pull an image directly (repeatable).")
    selection.add_argument("--config", type=Path, help="Select images using an experiment YAML.")
    parser.add_argument("--list", action="store_true", help="Print selected images as JSON without pulling them.")
    parser.add_argument("--limit", type=int, default=None, help="Only consider the first N distinct instances.")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--pull-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pull an image only if missing locally (default: true).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        images = experiment_images(load_experiment_config(args.config))
    elif args.image:
        images = list(dict.fromkeys(args.image))
    else:
        dataset_paths = [Path(path) for path in (args.dataset or [str(p) for p in DEFAULT_DATASETS])]
        rows = [item for path in dataset_paths for item in load_dataset(path)]
        images = distinct_images(rows, args.limit)
    if args.list:
        print(json.dumps(images))
        return
    prepare_images(images, pull_missing=args.pull_missing, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
