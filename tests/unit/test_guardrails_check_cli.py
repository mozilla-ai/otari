"""Unit tests for `otari guardrails check`, which runs a repo's gates against a change.

These tests run against real repositories, because the paths Git reports for a range are what is under test.
The model call is the one thing stubbed.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

import otari_agent.hook as hook_cli

pytestmark = [
    pytest.mark.skipif(shutil.which("git") is None, reason="needs a real git binary"),
    pytest.mark.usefixtures("isolated_home", "no_otari_env"),
]

_HEADER = 'schema_version: "1.0"\npolicy:\n  id: demo/guardrails\ngates:\n'

_CHANGELOG_GATE = (
    "  - id: no-hand-edited-changelog\n"
    "    type: path\n"
    "    runs: [pre_tool_use.edit_target, stop.working_tree]\n"
    "    enforcement: required\n"
    '    forbidden: ["CHANGELOG.md"]\n'
    "    message: CHANGELOG.md is generated at release time.\n"
)

_COMMAND_GATE = (
    "  - id: no-force-push\n"
    "    type: command\n"
    "    runs: [pre_tool_use.command]\n"
    "    enforcement: required\n"
    '    forbidden: ["git push --force"]\n'
)

_VERIFIER_GATE = (
    "  - id: tree-is-clean\n"
    "    type: verifier\n"
    "    runs: [stop.verifier]\n"
    "    enforcement: required\n"
    "    verifier: .otari/verifiers/check.sh\n"
    "    message: The verifier refused this tree.\n"
)

_JUDGE_GATE = (
    "  - id: no-narrative-comments\n"
    "    type: judge\n"
    "    runs: [stop.session]\n"
    "    enforcement: advisory\n"
    "    rubric: Does this diff add a comment that restates the code?\n"
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repo: Path) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "change")
    return _git(repo, "rev-parse", "HEAD")


def _write_guardrail(root: Path, gates: str) -> None:
    path = root / hook_cli.GUARDRAIL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HEADER + gates, encoding="utf-8")


def _write_verifier(root: Path, body: str) -> None:
    script = root / ".otari/verifiers/check.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(f"#!/usr/bin/env bash\n{body}\n", encoding="utf-8")
    script.chmod(0o755)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repository on `main` with one commit holding a guardrail, run from its root."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    (root / "README.md").write_text("readme\n", encoding="utf-8")
    _write_guardrail(root, _CHANGELOG_GATE + _COMMAND_GATE)
    _commit(root)
    monkeypatch.chdir(root)
    return root


def _invoke(*args: str) -> Result:
    return CliRunner().invoke(hook_cli.guardrails, ["check", *args])


def test_a_required_gate_failing_on_the_working_tree_exits_one(repo: Path) -> None:
    (repo / "CHANGELOG.md").write_text("## 1.2.0\n", encoding="utf-8")

    result = _invoke()

    assert result.exit_code == 1, result.output
    assert "fail        no-hand-edited-changelog (path, required)" in result.output
    assert "CHANGELOG.md is generated at release time." in result.output


def test_a_change_touching_no_forbidden_path_exits_zero(repo: Path) -> None:
    (repo / "README.md").write_text("edited\n", encoding="utf-8")

    result = _invoke()

    assert result.exit_code == 0, result.output
    assert "pass        no-hand-edited-changelog" in result.output


def test_commits_since_the_base_are_part_of_the_change(repo: Path) -> None:
    """A branch's own commits are what a pull request carries, not only what is uncommitted."""
    _git(repo, "switch", "-q", "-c", "feature")
    (repo / "CHANGELOG.md").write_text("## 1.2.0\n", encoding="utf-8")
    _commit(repo)

    result = _invoke("--base", "main")

    assert result.exit_code == 1, result.output
    assert "no-hand-edited-changelog" in result.output


def test_a_commit_that_reached_the_base_later_is_not_part_of_the_change(repo: Path) -> None:
    """The change starts at the merge base, so another branch's merged work is not checked here."""
    _git(repo, "switch", "-q", "-c", "feature")
    (repo / "feature.txt").write_text("feature\n", encoding="utf-8")
    _commit(repo)
    _git(repo, "switch", "-q", "main")
    (repo / "CHANGELOG.md").write_text("## 1.2.0\n", encoding="utf-8")
    _commit(repo)
    _git(repo, "switch", "-q", "feature")

    result = _invoke("--base", "main")

    assert result.exit_code == 0, result.output
    assert "1 changed path(s)" in result.output


def test_head_reads_the_commits_and_not_the_working_tree(repo: Path) -> None:
    _git(repo, "switch", "-q", "-c", "feature")
    (repo / "feature.txt").write_text("feature\n", encoding="utf-8")
    head = _commit(repo)
    (repo / "CHANGELOG.md").write_text("uncommitted\n", encoding="utf-8")

    result = _invoke("--base", "main", "--head", head)

    assert result.exit_code == 0, result.output
    assert "1 changed path(s)" in result.output


def test_a_deleted_path_is_part_of_the_change(repo: Path) -> None:
    (repo / "CHANGELOG.md").write_text("## 1.1.0\n", encoding="utf-8")
    _commit(repo)
    (repo / "CHANGELOG.md").unlink()

    result = _invoke()

    assert result.exit_code == 1, result.output
    assert "no-hand-edited-changelog" in result.output


def test_a_command_gate_is_reported_as_unable_to_run(repo: Path) -> None:
    (repo / "README.md").write_text("edited\n", encoding="utf-8")

    result = _invoke()

    assert result.exit_code == 0, result.output
    assert "cannot run against a change, which carries no commands: no-force-push." in result.output


def test_a_failing_required_verifier_exits_one_and_runs_in_the_checked_repo(repo: Path) -> None:
    _write_guardrail(repo, _VERIFIER_GATE)
    _write_verifier(repo, "pwd\nexit 1")

    result = _invoke()

    assert result.exit_code == 1, result.output
    assert "fail        tree-is-clean (verifier, required)" in result.output
    assert str(repo.resolve()) in result.output


def test_a_verifier_cannot_run_against_a_commit(repo: Path) -> None:
    _write_guardrail(repo, _VERIFIER_GATE)
    _write_verifier(repo, "exit 1")
    head = _commit(repo)

    result = _invoke("--base", f"{head}~1", "--head", head)

    assert result.exit_code == 0, result.output
    assert "cannot run  tree-is-clean (verifier, required)" in result.output


def test_guardrails_from_another_checkout_cannot_be_changed_by_the_change(repo: Path, tmp_path: Path) -> None:
    """The case a CI job exists for: a change that rewrites its own gate and verifier still meets the base's."""
    trusted = tmp_path / "trusted"
    _write_guardrail(trusted, _CHANGELOG_GATE + _VERIFIER_GATE)
    _write_verifier(trusted, "exit 1")
    _write_guardrail(repo, _VERIFIER_GATE)
    _write_verifier(repo, "exit 0")
    (repo / "CHANGELOG.md").write_text("## 1.2.0\n", encoding="utf-8")

    result = _invoke("--guardrails-from", str(trusted))

    assert result.exit_code == 1, result.output
    assert f"from {trusted}" in result.output
    assert "fail        no-hand-edited-changelog" in result.output
    assert "fail        tree-is-clean" in result.output


def test_type_runs_only_the_named_gate_types(repo: Path) -> None:
    _write_guardrail(repo, _CHANGELOG_GATE + _VERIFIER_GATE)
    _write_verifier(repo, "exit 1")
    (repo / "CHANGELOG.md").write_text("## 1.2.0\n", encoding="utf-8")

    result = _invoke("--type", "verifier")

    assert result.exit_code == 1, result.output
    assert "tree-is-clean" in result.output
    assert "no-hand-edited-changelog" not in result.output


def test_a_narrowed_run_does_not_list_the_command_gates(repo: Path) -> None:
    (repo / "README.md").write_text("edited\n", encoding="utf-8")

    result = _invoke("--type", "path")

    assert result.exit_code == 0, result.output
    assert "no-force-push" not in result.output


def test_a_judge_finding_is_reported_and_never_fails_the_check(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_guardrail(repo, _JUDGE_GATE)
    _git(repo, "switch", "-q", "-c", "feature")
    (repo / "added.py").write_text("x = 1  # set x to 1\n", encoding="utf-8")
    head = _commit(repo)
    seen: list[str] = []

    def fake_run_judge(rubric: str, diff: str, transcript: str, **kwargs: object) -> tuple[str, str]:
        seen.append(diff)
        return "fail", "The comment restates the assignment."

    monkeypatch.setattr(hook_cli, "_hook_run_judge", fake_run_judge)

    result = _invoke("--base", "main", "--head", head, "--type", "judge")

    assert result.exit_code == 0, result.output
    assert "fail        no-narrative-comments (judge, advisory)" in result.output
    assert "The comment restates the assignment." in result.output
    assert seen and "+x = 1  # set x to 1" in seen[0]
    # NOTE: The judges workflow finds a failing judge with this pattern before it creates its PR comment.
    assert re.search(r"^  fail ", result.output, re.MULTILINE)


def test_the_judge_options_reach_the_judge_call(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_guardrail(repo, _JUDGE_GATE)
    head = _commit(repo)
    seen: list[dict[str, object]] = []

    def fake_run_judge(rubric: str, diff: str, transcript: str, **kwargs: object) -> tuple[str, str]:
        seen.append(kwargs)
        return "pass", "fine"

    monkeypatch.setattr(hook_cli, "_hook_run_judge", fake_run_judge)

    result = _invoke(
        "--base", f"{head}~1", "--head", head, "--type", "judge", "--judge-cli", "codex", "--judge-model", "m"
    )

    assert result.exit_code == 0, result.output
    assert [(call["judge_cli"], call["model"]) for call in seen] == [(("codex",), "m")]


def test_max_judges_caps_the_judge_calls(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    judges = "".join(_JUDGE_GATE.replace("no-narrative-comments", f"judge-{index}") for index in range(3))
    _write_guardrail(repo, judges)
    head = _commit(repo)
    seen: list[str] = []

    def fake_run_judge(rubric: str, diff: str, transcript: str, **kwargs: object) -> tuple[str, str]:
        seen.append(rubric)
        return "pass", "fine"

    monkeypatch.setattr(hook_cli, "_hook_run_judge", fake_run_judge)

    result = _invoke("--base", f"{head}~1", "--head", head, "--type", "judge", "--max-judges", "2")

    assert result.exit_code == 0, result.output
    assert len(seen) == 2
    assert re.search(r"^  not_run +judge-2 ", result.output, re.MULTILINE)


def test_asking_for_verifiers_against_a_commit_is_a_usage_error(repo: Path) -> None:
    head = _git(repo, "rev-parse", "HEAD")

    result = _invoke("--head", head, "--type", "verifier", "--type", "path")

    assert result.exit_code == 2
    assert "cannot run with --head" in result.output


def test_a_judge_runs_claude_with_no_tools_outside_the_repo(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CI judge reads a pull request author's diff next to a model credential, so the model gets no tool."""
    _write_guardrail(repo, _JUDGE_GATE)
    head = _commit(repo)
    real_run = subprocess.run
    calls: list[tuple[list[str], object]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[0] != "/fake/bin/claude":
            return real_run(cmd, **kwargs)  # type: ignore[call-overload, no-any-return]
        calls.append((cmd, kwargs.get("cwd")))
        return subprocess.CompletedProcess(cmd, 0, '{"outcome": "pass", "reasoning": "fine"}', "")

    monkeypatch.setattr(shutil, "which", lambda name: f"/fake/bin/{name}")
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = _invoke("--base", f"{head}~1", "--head", head, "--type", "judge")

    assert result.exit_code == 0, result.output
    [(argv, cwd)] = calls
    assert argv[argv.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in argv
    assert cwd == hook_cli._hook_judge_workdir()


def test_the_user_guardrail_is_not_part_of_a_check(repo: Path, isolated_home: Path) -> None:
    """A check reads the same on every machine, so one person's files in ~/.otari/ stay out of it."""
    _write_guardrail(isolated_home, _CHANGELOG_GATE.replace("no-hand-edited-changelog", "mine"))
    _write_guardrail(repo, _COMMAND_GATE)
    (repo / "CHANGELOG.md").write_text("## 1.2.0\n", encoding="utf-8")

    result = _invoke()

    assert result.exit_code == 0, result.output
    assert "mine" not in result.output


def test_a_base_naming_no_commit_is_refused(repo: Path) -> None:
    result = _invoke("--base", "no-such-branch")

    assert result.exit_code != 0
    assert "--base 'no-such-branch' names no commit" in result.output


def test_a_revision_spelled_as_an_option_is_refused(repo: Path) -> None:
    result = _invoke("--base", "--output=/tmp/x")

    assert result.exit_code == 2
    assert "must name a commit, not an option" in result.output


def test_fails_outside_a_git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = _invoke()

    assert result.exit_code != 0
    assert "Not inside a Git repository." in result.output


def test_git_failing_to_run_is_reported_as_such(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing Git must not read as a revision that names no commit."""

    def no_git(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(2, "No such file or directory", "git")

    monkeypatch.setattr(subprocess, "run", no_git)

    result = _invoke()

    assert result.exit_code != 0
    assert "`git rev-parse` could not run" in result.output
    assert "names no commit" not in result.output


@pytest.mark.parametrize("value", ["0", str(hook_cli._HOOK_JUDGE_MAX_GATES_CEILING + 1), "many"])
def test_a_judge_gate_cap_outside_its_range_is_a_usage_error(repo: Path, value: str) -> None:
    result = _invoke("--max-judges", value)
    assert result.exit_code == 2
    assert "--max-judges" in result.output
