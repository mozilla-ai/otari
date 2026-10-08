"""This repository's own committed guardrail: it composes, and the git and GitHub gates fire."""

import re
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

import otari_agent.hook as hook_cli

pytestmark = pytest.mark.usefixtures("isolated_home", "no_otari_env")

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _in_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)


def _validate(*args: str) -> Result:
    return CliRunner().invoke(hook_cli.guardrails, ["validate", "--repo-only", *args])


def test_the_committed_guardrail_has_no_errors() -> None:
    result = _validate()
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    ("command", "gate_id"),
    [
        ("git push origin main", "no-push-to-main"),
        ("git push origin HEAD:main", "no-push-to-main"),
        ("git push --set-upstream origin main", "no-push-to-main"),
        ("git push --force origin main", "no-push-to-main"),
        ("git push --force-with-lease origin main", "no-push-to-main"),
        ("git push origin +main", "no-push-to-main"),
        ("git merge origin/main", "no-merging-main-into-a-branch"),
        ("git merge --no-ff origin/main", "no-merging-main-into-a-branch"),
        ("gh pr update-branch 12", "no-merging-main-into-a-branch"),
        ("gh issue create --title x", "no-agent-filed-issues"),
    ],
)
def test_a_required_gate_refuses_the_command(command: str, gate_id: str) -> None:
    result = _validate("--command", command)
    assert re.search(rf"fires\s+{gate_id} \(required\)", result.output), result.output


@pytest.mark.parametrize(
    "command",
    [
        "git push origin feat/some-change",
        "git push origin main-fix",
        "git pull --rebase origin main",
        "git merge --ff-only origin/main",
    ],
)
def test_an_ordinary_branch_command_passes_the_git_gates(command: str) -> None:
    result = _validate("--command", command)
    for gate_id in ("no-push-to-main", "no-merging-main-into-a-branch"):
        assert re.search(rf"quiet\s+{gate_id} \(required\)", result.output), result.output
