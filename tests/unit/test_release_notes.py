"""The release notes that git-cliff renders from cliff.toml.

A deployment reads the notes to decide whether an upgrade needs work on its side, so a breaking change must stand out.
These tests render real commits with the git-cliff version that the release workflow pins.
uvx fetches that version from PyPI, so a cold cache needs network access.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLIFF_CONFIG = _REPO_ROOT / "cliff.toml"
_RELEASE_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "otari-release.yml"

_UNRELEASED_COMMITS = [
    "feat(ports)!: add a required port method",
    "fix(catalog): read hosted providers\n\nBREAKING CHANGE: adapters must implement a new port method",
    "refactor(ports)!: rename a port method",
    "fix: correct a plain bug",
    "refactor: tidy a plain module",
    "Update a readme without a type",
]


def _pinned_git_cliff_version() -> str:
    versions = {match.group(1) for match in re.finditer(r"git-cliff@(\S+)", _RELEASE_WORKFLOW.read_text())}
    assert len(versions) == 1, f"the release workflow pins more than one git-cliff version: {versions}"
    return versions.pop()


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=repo,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"},
        capture_output=True,
        check=True,
    )


@pytest.fixture(scope="module")
def notes(tmp_path_factory: pytest.TempPathFactory) -> str:
    if shutil.which("uvx") is None:
        pytest.skip("uvx is not on PATH")

    repo = tmp_path_factory.mktemp("repo")
    _git(repo, "init", "--quiet")
    _git(repo, "commit", "--quiet", "--allow-empty", "--message", "feat: start the project")
    _git(repo, "tag", "v0.1.0")
    for message in _UNRELEASED_COMMITS:
        _git(repo, "commit", "--quiet", "--allow-empty", "--message", message)

    rendered = subprocess.run(
        [
            "uvx",
            f"git-cliff@{_pinned_git_cliff_version()}",
            "--config",
            str(_CLIFF_CONFIG),
            "--offline",
            "--unreleased",
            "--strip",
            "header",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert rendered.returncode == 0, rendered.stderr
    return rendered.stdout


@pytest.fixture(scope="module")
def entries_by_group(notes: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    entries: list[str] | None = None
    for line in notes.splitlines():
        if line.startswith("### "):
            entries = groups.setdefault(line.removeprefix("### "), [])
        elif line.startswith("- ") and entries is not None:
            entries.append(line)
    return groups


@pytest.mark.parametrize(
    ("group", "entry"),
    [
        ("Features", "- **BREAKING:** **ports:** Add a required port method"),
        ("Bug Fixes", "- **BREAKING:** **catalog:** Read hosted providers"),
        ("Maintenance", "- **BREAKING:** **ports:** Rename a port method"),
    ],
)
def test_a_breaking_commit_is_marked(entries_by_group: dict[str, list[str]], group: str, entry: str) -> None:
    assert any(line.startswith(entry) for line in entries_by_group[group]), entries_by_group


def test_a_plain_commit_is_not_marked(entries_by_group: dict[str, list[str]]) -> None:
    assert any(line.startswith("- Correct a plain bug") for line in entries_by_group["Bug Fixes"]), entries_by_group


def test_a_plain_commit_of_a_hidden_type_stays_hidden(notes: str) -> None:
    assert "Tidy a plain module" not in notes


def test_a_commit_without_a_type_still_renders(entries_by_group: dict[str, list[str]]) -> None:
    assert any(line.startswith("- Update a readme without a type") for line in entries_by_group["Other"]), (
        entries_by_group
    )


def test_no_entry_renders_under_a_raw_type_name(entries_by_group: dict[str, list[str]]) -> None:
    assert set(entries_by_group) == {"Features", "Bug Fixes", "Maintenance", "Other"}
