"""The shell steps of the release workflow, run with bash as a GitHub runner runs them."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

_RELEASE_WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "otari-release.yml"

_BREAKING_NOTES = "- **BREAKING:** **ports:** Add a required port method\n"
_PLAIN_NOTES = "- **ports:** Add an optional port method\n"


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


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=repo,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"},
        capture_output=True,
        check=True,
    )


@pytest.mark.parametrize("version", ["0.7.0", "0.7.0-rc.1", "0.7.0-rc-1"])
def test_a_semver_version_is_accepted(tmp_path: Path, version: str) -> None:
    result = _run_step("Validate version format", tmp_path, {"VERSION": version})

    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize(
    "version",
    [
        "v0.7.0",
        "0.7",
        "0.7.0-",
        "0.7.0-.",
        "0.7.0-rc.",
        "0.7.0-rc..1",
        "0.7.0\n",
        "0.7.0\nextra",
        "a[$(touch pwned)]\n0.7.0",
    ],
)
def test_a_version_that_is_not_exactly_semver_is_refused(tmp_path: Path, version: str) -> None:
    result = _run_step("Validate version format", tmp_path, {"VERSION": version})

    assert result.returncode == 1
    assert not (tmp_path / "pwned").exists()


def _check_version(tmp_path: Path, *, tags: list[str], version: str, notes: str) -> str | None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    for tag in tags:
        _git(repo, "commit", "--quiet", "--allow-empty", "--message", tag)
        _git(repo, "tag", tag)
    _git(repo, "commit", "--quiet", "--allow-empty", "--message", "an unreleased change")
    notes_path = tmp_path / "notes.md"
    notes_path.write_text(notes)
    output_path = tmp_path / "output"
    output_path.touch()

    result = _run_step(
        "Check the version against breaking changes",
        repo,
        {"VERSION": version, "NOTES": str(notes_path), "GITHUB_OUTPUT": str(output_path)},
    )

    assert result.returncode == 0, result.stderr
    outputs = dict(line.split("=", 1) for line in output_path.read_text().splitlines())
    return outputs.get("warning")


@pytest.mark.parametrize(
    ("tags", "version"),
    [
        (["v0.6.3"], "0.7.0"),
        (["v0.6.3"], "1.0.0"),
        (["v1.2.3"], "2.0.0"),
        (["v0.6.3", "v0.7.0-rc.1"], "0.7.0"),
    ],
)
def test_a_breaking_release_that_raises_the_version_has_no_warning(
    tmp_path: Path, tags: list[str], version: str
) -> None:
    assert _check_version(tmp_path, tags=tags, version=version, notes=_BREAKING_NOTES) is None


@pytest.mark.parametrize(
    ("tags", "version", "part"),
    [
        (["v0.6.3"], "0.6.4", "minor"),
        (["v1.2.3"], "1.3.0", "major"),
        (["v0.6.3", "v0.7.0-rc.1"], "0.6.4", "minor"),
    ],
)
def test_a_breaking_release_that_does_not_raise_the_version_has_a_warning(
    tmp_path: Path, tags: list[str], version: str, part: str
) -> None:
    warning = _check_version(tmp_path, tags=tags, version=version, notes=_BREAKING_NOTES)

    assert warning is not None
    assert f"raises the {part} version, but v{version} does not raise it over {tags[0]}" in warning


def test_a_release_without_a_breaking_change_has_no_warning(tmp_path: Path) -> None:
    assert _check_version(tmp_path, tags=["v0.6.3"], version="0.6.4", notes=_PLAIN_NOTES) is None


def test_the_version_is_not_evaluated_as_an_expression(tmp_path: Path) -> None:
    _check_version(tmp_path, tags=["v0.6.3"], version="a[$(touch pwned)]", notes=_BREAKING_NOTES)

    assert not (tmp_path / "repo" / "pwned").exists()
