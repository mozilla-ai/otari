"""The shell steps of the release workflow, run with bash as a GitHub runner runs them."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

_RELEASE_WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "otari-release.yml"


def _run_step(name: str, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    steps = yaml.safe_load(_RELEASE_WORKFLOW.read_text())["jobs"]["open-release-pr"]["steps"]
    script = next(step["run"] for step in steps if step.get("name") == name)
    return subprocess.run(
        ["bash", "-e", "-c", script],
        cwd=cwd,
        env={"PATH": os.environ["PATH"], "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", **env},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize("version", ["0.7.0", "0.7.0-rc.1"])
def test_a_semver_version_is_accepted(tmp_path: Path, version: str) -> None:
    result = _run_step("Validate version format", tmp_path, {"VERSION": version})

    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize("version", ["v0.7.0", "0.7", "0.7.0\n", "0.7.0\nextra", "a[$(touch pwned)]\n0.7.0"])
def test_a_version_that_is_not_exactly_semver_is_refused(tmp_path: Path, version: str) -> None:
    result = _run_step("Validate version format", tmp_path, {"VERSION": version})

    assert result.returncode == 1
    assert not (tmp_path / "pwned").exists()
