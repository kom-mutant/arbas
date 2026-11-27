from __future__ import annotations

import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

from examples.swe_smith.hierarchical.sandbox import _CMD_ENV, ContainerSandbox

_SCRUBBED_SECRETS = {"AGL_KEY", "OPENROUTER_API_KEY", "POLZA_API_KEY", "JUDGE_API_KEY"}


class LocalTestSandbox(ContainerSandbox):
    """Test-only ``ContainerSandbox`` subclass running git/pytest/bash in a local ``tmp_path`` directory."""

    python = sys.executable
    shell = ("bash", "-c")

    def __init__(self, root: Path, *, git_dir: Path | None = None) -> None:
        self.root = str(root.resolve())
        self.git_dir = str((git_dir or root.parent / f".{root.name}-base.git").resolve())
        self._tests = b""
        self._configs = b""
        self._env = {
            **{key: value for key, value in os.environ.items() if key not in _SCRUBBED_SECRETS},
            **_CMD_ENV,
        }

    def __enter__(self) -> LocalTestSandbox:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def _exec(
        self, argv: list[str], *, stdin: bytes | None = None, timeout: float
    ) -> subprocess.CompletedProcess[bytes]:
        proc = subprocess.Popen(
            argv,
            cwd=self.root,
            env=self._env,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(input=stdin, timeout=timeout)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
            raise subprocess.TimeoutExpired(argv, timeout, output=stdout, stderr=stderr) from None
        finally:
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
        return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)
