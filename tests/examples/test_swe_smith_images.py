# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from examples.swe_smith import pull_images


@dataclass
class _FakeImage:
    name: str

    def tag(self, repository: str, tag: str) -> bool:
        return True


class _FakeImages:
    def __init__(self, existing: set[str] | None = None) -> None:
        self.existing = set(existing or set())
        self.pulled: list[str] = []
        self.built: list[tuple[str, str, bool]] = []

    def get(self, image: str) -> _FakeImage:
        if image not in self.existing:
            raise KeyError(image)
        fake = _FakeImage(image)

        def tag(repository: str, tag: str) -> bool:
            self.existing.add(f"{repository}:{tag}")
            return True

        fake.tag = tag  # type: ignore[method-assign]
        return fake

    def pull(self, image: str) -> _FakeImage:
        self.pulled.append(image)
        self.existing.add(image)
        return _FakeImage(image)

    def build(
        self,
        *,
        path: str,
        tag: str,
        rm: bool,
        forcerm: bool,
        pull: bool,
    ) -> tuple[_FakeImage, list[Any]]:
        dockerfile = Path(path, "Dockerfile").read_text()
        self.built.append((tag, dockerfile, pull))
        self.existing.add(tag)
        return _FakeImage(tag), []


class _FakeClient:
    def __init__(self, existing: set[str] | None = None) -> None:
        self.images = _FakeImages(existing)


def test_openai_image_name_matches_job_template_convention() -> None:
    assert pull_images.openai_image_name("jyangballin/swesmith.foo") == ("jyangballin/swesmith.foo:arbas")


def test_image_limit_selects_the_same_distinct_instances_as_training() -> None:
    rows = [{"instance_id": "a", "image_name": "image-a"},
            {"instance_id": "a", "image_name": "duplicate"},
            {"instance_id": "b", "image_name": "image-b"}]
    assert pull_images.distinct_images(rows, 2) == ["image-a", "image-b"]
    with pytest.raises(ValueError, match="Need 3 distinct"):
        pull_images.distinct_images(rows, 3)


@pytest.mark.parametrize("trainer, expected", [
    ("val_before_train: false", ["repo/a"]),
    ("val_before_train: false\n  test_freq: 1", ["repo/a", "repo/val"]),
    ("test_freq: -1", ["repo/a", "repo/val"]),
])
def test_list_images_from_experiment(tmp_path, monkeypatch, capsys, trainer, expected) -> None:
    base_dir = tmp_path / "shared"
    base_dir.mkdir()
    (base_dir / "train.jsonl").write_text('\n'.join(json.dumps(row) for row in [
        {"instance_id": "a", "image_name": "repo/a"},
        {"instance_id": "b", "image_name": "repo/b"},
    ]))
    (base_dir / "base.yaml").write_text("data:\n  train_files: train.jsonl\n  max_train_instances: 2\n")
    (tmp_path / "val.jsonl").write_text(json.dumps({"instance_id": "v", "image_name": "repo/val"}))
    config = tmp_path / "training.yaml"
    config.write_text(
        "base: shared/base.yaml\ndata:\n  val_files: val.jsonl\n  max_train_instances: 1\n"
        f"trainer:\n  {trainer}\n"
    )
    monkeypatch.setattr(sys, "argv", ["pull_images.py", "--config", str(config), "--list"])
    pull_images.main()
    assert json.loads(capsys.readouterr().out) == expected


def test_openai_image_name_rejects_already_tagged_sources() -> None:
    with pytest.raises(ValueError, match="must be untagged"):
        pull_images.openai_image_name("jyangballin/swesmith.foo:latest")


def test_prepare_one_skips_when_openai_image_already_loaded() -> None:
    client = _FakeClient({"repo/image:arbas"})

    result = pull_images.prepare_one(
        client,
        "repo/image",
        pull_missing=True,
        force_build=False,
    )

    assert result.status == "exists"
    assert client.images.pulled == []
    assert client.images.built == []


def test_prepare_one_builds_openai_layer_from_local_base_without_pulling() -> None:
    client = _FakeClient({"repo/image"})

    result = pull_images.prepare_one(
        client,
        "repo/image",
        pull_missing=True,
        force_build=False,
    )

    assert result.status == "base-local,built"
    assert client.images.pulled == []
    assert len(client.images.built) == 1
    tag, dockerfile, pull = client.images.built[0]
    assert tag == "repo/image:arbas"
    assert "FROM repo/image" in dockerfile
    assert "pip install --no-cache-dir openai" in dockerfile
    assert "numpy>=1.23" in dockerfile
    assert "COPY ORACLE" not in dockerfile
    assert pull is False


def test_prepare_one_does_not_reuse_images_without_oracle_dependencies() -> None:
    client = _FakeClient({"repo/image", "repo/image:openai", "repo/image:agl-openai"})

    result = pull_images.prepare_one(
        client,
        "repo/image",
        pull_missing=False,
        force_build=False,
    )

    assert result.status == "base-local,built"
    assert "repo/image:arbas" in client.images.existing
    assert client.images.pulled == []
    assert len(client.images.built) == 1


def test_prepare_one_pulls_missing_base_only_when_allowed() -> None:
    client = _FakeClient()

    result = pull_images.prepare_one(
        client,
        "repo/image",
        pull_missing=True,
        force_build=False,
    )

    assert result.status == "base-pulled,built"
    assert client.images.pulled == ["repo/image"]
    assert client.images.built[0][0] == "repo/image:arbas"


def test_prepare_one_can_require_preloaded_base_image() -> None:
    client = _FakeClient()

    result = pull_images.prepare_one(
        client,
        "repo/image",
        pull_missing=False,
        force_build=False,
    )

    assert result.status == "failed"
    assert "base image is not loaded" in (result.error or "")
    assert client.images.pulled == []
    assert client.images.built == []
