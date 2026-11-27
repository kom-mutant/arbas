# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from examples.swe_smith import pull_images


class _FakeDocker:
    def __init__(self, existing: set[str] | None = None, *, pull_error: str | None = None) -> None:
        self.existing = set(existing or set())
        self.pull_error = pull_error
        self.calls: list[list[str]] = []
        self.envs: list[dict[str, str] | None] = []

    def __call__(self, argv: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        self.envs.append(kwargs.get("env"))
        if argv[:3] == ["docker", "image", "inspect"]:
            rc = 0 if argv[3] in self.existing else 1
            return subprocess.CompletedProcess(argv, rc, "", "")
        if argv[:2] == ["docker", "pull"]:
            if self.pull_error is not None:
                return subprocess.CompletedProcess(argv, 1, "", self.pull_error)
            self.existing.add(argv[2])
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise AssertionError(f"Unexpected command: {argv}")


def test_image_limit_selects_the_same_distinct_instances_as_training() -> None:
    rows = [
        {"instance_id": "a", "image_name": "image-a"},
        {"instance_id": "a", "image_name": "duplicate"},
        {"instance_id": "b", "image_name": "image-b"},
    ]
    assert pull_images.distinct_images(rows, 2) == ["image-a", "image-b"]
    with pytest.raises(ValueError, match="Need 3 distinct"):
        pull_images.distinct_images(rows, 3)


@pytest.mark.parametrize(
    ("trainer", "expected"),
    [
        ("val_before_train: false", ["repo/a"]),
        ("val_before_train: false\n  test_freq: 1", ["repo/a", "repo/val"]),
        ("test_freq: -1", ["repo/a", "repo/val"]),
    ],
)
def test_list_images_from_experiment(tmp_path, monkeypatch, capsys, trainer, expected) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"instance_id": "a", "image_name": "repo/a"},
                {"instance_id": "b", "image_name": "repo/b"},
            ]
        )
    )
    (tmp_path / "val.jsonl").write_text(json.dumps({"instance_id": "v", "image_name": "repo/val"}))
    config = tmp_path / "training.yaml"
    config.write_text(
        "data:\n"
        "  train_files: data/train.jsonl\n"
        "  val_files: val.jsonl\n"
        "  max_train_instances: 1\n"
        f"trainer:\n  {trainer}\n"
    )
    override_dir = tmp_path / "profiles"
    override_dir.mkdir()
    override = override_dir / "smoke.yaml"
    override.write_text("base: ../training.yaml\ndata:\n  max_train_instances: 1\n")
    monkeypatch.setattr(sys, "argv", ["pull_images.py", "--config", str(override), "--list"])
    pull_images.main()
    assert json.loads(capsys.readouterr().out) == expected


def test_docker_calls_scrub_api_keys(monkeypatch) -> None:
    fake = _FakeDocker()
    monkeypatch.setattr(pull_images.subprocess, "run", fake)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")
    monkeypatch.setenv("POLZA_API_KEY", "polza-secret")
    monkeypatch.setenv("AGL_KEY", "agl-secret")
    monkeypatch.setenv("DOCKER_HOST", "unix:///tmp/custom.sock")

    assert pull_images.prepare_one("repo/image", pull_missing=True) == pull_images.ImageResult("repo/image", "pulled")
    assert len(fake.envs) == 2
    for env in fake.envs:
        assert env is not None
        assert env["DOCKER_HOST"] == "unix:///tmp/custom.sock"
        assert "OPENROUTER_API_KEY" not in env
        assert "POLZA_API_KEY" not in env
        assert "AGL_KEY" not in env


def test_image_exists_uses_docker_image_inspect(monkeypatch) -> None:
    fake = _FakeDocker({"repo/present"})
    monkeypatch.setattr(pull_images.subprocess, "run", fake)

    assert pull_images.image_exists("repo/present") is True
    assert pull_images.image_exists("repo/missing") is False
    assert fake.calls == [
        ["docker", "image", "inspect", "repo/present"],
        ["docker", "image", "inspect", "repo/missing"],
    ]


def test_prepare_one_skips_pull_when_base_image_already_exists(monkeypatch) -> None:
    fake = _FakeDocker({"repo/image"})
    monkeypatch.setattr(pull_images.subprocess, "run", fake)

    result = pull_images.prepare_one("repo/image", pull_missing=True)

    assert result == pull_images.ImageResult("repo/image", "exists")
    assert fake.calls == [["docker", "image", "inspect", "repo/image"]]


def test_prepare_one_pulls_missing_base_image_when_allowed(monkeypatch) -> None:
    fake = _FakeDocker()
    monkeypatch.setattr(pull_images.subprocess, "run", fake)

    result = pull_images.prepare_one("repo/image", pull_missing=True)

    assert result == pull_images.ImageResult("repo/image", "pulled")
    assert fake.calls == [
        ["docker", "image", "inspect", "repo/image"],
        ["docker", "pull", "repo/image"],
    ]


def test_prepare_one_can_require_preloaded_base_image(monkeypatch) -> None:
    fake = _FakeDocker()
    monkeypatch.setattr(pull_images.subprocess, "run", fake)

    result = pull_images.prepare_one("repo/image", pull_missing=False)

    assert result.status == "failed"
    assert "image is not loaded" in (result.error or "")
    assert fake.calls == [["docker", "image", "inspect", "repo/image"]]


def test_prepare_one_reports_docker_pull_error(monkeypatch) -> None:
    fake = _FakeDocker(pull_error="pull access denied")
    monkeypatch.setattr(pull_images.subprocess, "run", fake)

    result = pull_images.prepare_one("repo/image", pull_missing=True)

    assert result == pull_images.ImageResult("repo/image", "failed", "pull access denied")


def test_prepare_images_and_main_pull_unique_images_and_fail_on_error(monkeypatch, capsys) -> None:
    fake = _FakeDocker({"repo/a"})
    monkeypatch.setattr(pull_images.subprocess, "run", fake)

    pull_images.prepare_images(["repo/a", "repo/b", "repo/a"])
    out = capsys.readouterr().out
    assert "EXISTS  repo/a" in out
    assert "PULLED  repo/b" in out
    assert "Prepared 2 SWE-smith images." in out
    assert [c for c in fake.calls if c[1] == "pull"] == [["docker", "pull", "repo/b"]]

    monkeypatch.setattr(pull_images.subprocess, "run", _FakeDocker(pull_error="network down"))
    monkeypatch.setattr(sys, "argv", ["pull_images.py", "--image", "repo/missing"])
    with pytest.raises(SystemExit, match="Failed to prepare SWE-smith images: repo/missing"):
        pull_images.main()
