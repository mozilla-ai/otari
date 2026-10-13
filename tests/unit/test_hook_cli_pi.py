"""Unit tests for `otari hook --harness pi` and `otari hook setup --harness pi`.

pi's payload carries pi-native tool names (`bash`, `read`, `edit`, `write`) and
`tool_input` (`{command}` / `{path}`), and, on Stop, the session's commands as
`bash_commands` instead of a transcript the hook would parse. The pi-otari
extension is what registers the hook with pi; see docs/agent-guardrails.md.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

import otari_agent.hook as hook_cli
from otari_agent.domain.check import PolicyCheckResult

pytestmark = pytest.mark.usefixtures("isolated_home", "no_otari_env")

_GATES_YAML = (
    'schema_version: "1.0"\npolicy:\n  id: test\ngates:\n'
    "  - id: no-changelog\n    type: path\n"
    "    runs: [pre_tool_use.edit_target, stop.working_tree]\n    enforcement: required\n"
    '    forbidden: ["CHANGELOG.md"]\n    message: changelog is protected\n'
    "  - id: no-force-push\n    type: command\n"
    "    runs: [pre_tool_use.command]\n    enforcement: required\n"
    '    forbidden: ["git push --force"]\n    message: no force push\n'
    "  - id: no-env-read\n    type: path\n"
    "    runs: [pre_tool_use.read_target]\n    enforcement: required\n"
    '    forbidden: [".env"]\n    message: env is protected\n'
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    guardrail = tmp_path / hook_cli.GUARDRAIL_FILE
    guardrail.parent.mkdir(parents=True, exist_ok=True)
    guardrail.write_text(_GATES_YAML, encoding="utf-8")
    return tmp_path


def _invoke(payload: dict[str, Any] | str) -> Any:
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    return CliRunner().invoke(hook_cli.hook, ["--harness", "pi"], input=raw)


def _pre(repo: Path, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    return {"hook_event_name": "PreToolUse", "cwd": str(repo), "tool_name": tool_name, "tool_input": tool_input}


def test_blocks_a_forbidden_bash_command(repo: Path) -> None:
    result = _invoke(_pre(repo, "bash", {"command": "git push --force origin main"}))
    assert result.exit_code == 2, result.output
    assert "no force push" in result.output


def test_allows_a_harmless_bash_command(repo: Path) -> None:
    assert _invoke(_pre(repo, "bash", {"command": "git status"})).exit_code == 0


@pytest.mark.parametrize("tool", ["edit", "write"])
def test_blocks_a_protected_edit_path(repo: Path, tool: str) -> None:
    result = _invoke(_pre(repo, tool, {"path": str(repo / "CHANGELOG.md")}))
    assert result.exit_code == 2, result.output
    assert "changelog is protected" in result.output


def test_relative_path_is_resolved_against_cwd(repo: Path) -> None:
    result = _invoke(_pre(repo, "edit", {"path": "CHANGELOG.md"}))
    assert result.exit_code == 2, result.output


def test_home_relative_path_is_expanded(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    # pi expands `~/...` itself, so a gate must see the real target.
    monkeypatch.setenv("HOME", str(repo.parent))
    result = _invoke(_pre(repo, "edit", {"path": f"~/{repo.name}/CHANGELOG.md"}))
    assert result.exit_code == 2, result.output
    result = _invoke(_pre(repo, "read", {"path": f"~/{repo.name}/.env"}))
    assert result.exit_code == 2, result.output


def test_read_maps_to_read_target(repo: Path) -> None:
    result = _invoke(_pre(repo, "read", {"path": ".env"}))
    assert result.exit_code == 2, result.output
    assert "env is protected" in result.output
    # An edit gate scoped to edit_target does not see a read, and vice versa.
    assert _invoke(_pre(repo, "read", {"path": "CHANGELOG.md"})).exit_code == 0
    assert _invoke(_pre(repo, "write", {"path": ".env"})).exit_code == 0


def test_evidence_surfaces(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    captured: list[dict[str, Any]] = []

    def fake_check(spec: Any, **kwargs: Any) -> PolicyCheckResult:
        captured.append(kwargs)
        return PolicyCheckResult(policy_id="t", schema_version="1.0", results=(), blocked=False)

    monkeypatch.setattr(hook_cli, "check_policy", fake_check)
    _invoke(_pre(repo, "edit", {"path": "docs/a.md"}))
    _invoke(_pre(repo, "read", {"path": "docs/b.md"}))
    _invoke(_pre(repo, "bash", {"command": "ls"}))
    assert [(c["path_source"], c["paths"], c["commands"]) for c in captured] == [
        ("pre_tool_use.edit_target", ["docs/a.md"], []),
        ("pre_tool_use.read_target", ["docs/b.md"], []),
        ("pre_tool_use.command", [], ["ls"]),
    ]


@pytest.mark.parametrize("tool", ["Bash", "grep", "Edit", "glob"])
def test_unknown_or_foreign_tool_names_are_allowed(repo: Path, tool: str) -> None:
    payload = _pre(repo, tool, {"command": "git push --force", "file_path": "CHANGELOG.md", "path": "CHANGELOG.md"})
    assert _invoke(payload).exit_code == 0


def test_stop_uses_bash_commands_and_ignores_transcript(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_check(spec: Any, **kwargs: Any) -> PolicyCheckResult:
        captured.update(kwargs)
        return PolicyCheckResult(policy_id="t", schema_version="1.0", results=(), blocked=False)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("pi's transcript must never be parsed")

    monkeypatch.setattr(hook_cli, "check_policy", fake_check)
    monkeypatch.setattr(hook_cli, "_hook_collect_changed_paths", lambda root: ["src/x.py"])
    monkeypatch.setattr(hook_cli, "_hook_collect_transcript_commands", boom)
    monkeypatch.setattr(hook_cli, "_hook_collect_codex_transcript_commands", boom)
    transcript = repo / "session.jsonl"
    transcript.write_text("not claude format\n", encoding="utf-8")
    payload = {
        "hook_event_name": "Stop",
        "cwd": str(repo),
        "stop_hook_active": False,
        "bash_commands": ["pytest -q", "git status"],
        "transcript_path": str(transcript),
    }
    assert _invoke(payload).exit_code == 0
    assert captured["commands"] == ["pytest -q", "git status"]
    assert captured["command_scope"] == "session"
    assert captured["paths"] == ["src/x.py"]

    captured.clear()
    del payload["bash_commands"]
    assert _invoke(payload).exit_code == 0
    assert captured["commands"] is None


def test_stop_blocks_on_a_protected_changed_path(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    monkeypatch.setattr(hook_cli, "_hook_collect_changed_paths", lambda root: ["CHANGELOG.md"])
    payload = {"hook_event_name": "Stop", "cwd": str(repo), "bash_commands": []}
    result = _invoke(payload)
    assert result.exit_code == 2, result.output
    assert "blocked (pi, Stop)" in result.output


@pytest.mark.parametrize("raw", ["", "not json", "[]", "null"])
def test_bad_input_fails_open(repo: Path, raw: str) -> None:
    assert _invoke(raw).exit_code == 0


def test_missing_fields_fail_open(repo: Path) -> None:
    assert _invoke({"hook_event_name": "PreToolUse", "cwd": str(repo), "tool_name": "edit"}).exit_code == 0
    assert _invoke(_pre(repo, "bash", {})).exit_code == 0
    assert _invoke({"hook_event_name": "Other", "cwd": str(repo)}).exit_code == 0
    assert _invoke({"hook_event_name": "PreToolUse", "cwd": str(repo / "nope")}).exit_code == 0


def test_stop_outside_a_repo_exits_zero_without_evaluating(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """What stops a pi judge call from looping: it runs in a non-repo workdir, where the hook bails."""

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("nothing may be evaluated outside a repo")

    monkeypatch.setattr(hook_cli, "check_policy", boom)
    monkeypatch.setattr(subprocess, "run", boom)
    outside = tmp_path / "judge-workdir"
    outside.mkdir()
    result = _invoke({"hook_event_name": "Stop", "cwd": str(outside), "bash_commands": ["ls"]})
    assert result.exit_code == 0, result.output


# `otari hook setup --harness pi`

_PI = "/usr/local/bin/pi"
_PKG = "@mozilla-ai/pi-otari"


def _fake_pi(
    monkeypatch: pytest.MonkeyPatch, *, listing: str = "", install_rc: int = 0, install_out: str = "installed"
) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        assert argv[0] == _PI, "only pi may be run"
        if argv[1:] == ["list"]:
            return subprocess.CompletedProcess(argv, 0, stdout=listing, stderr="")
        return subprocess.CompletedProcess(argv, install_rc, stdout=install_out, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: _PI if name == "pi" else None)
    return calls


def _setup(input: str | None = None) -> Any:  # noqa: A002
    return CliRunner().invoke(hook_cli.hook, ["setup", "--harness", "pi"], input=input)


def _assert_no_harness_files(root: Path) -> None:
    assert not (root / ".claude").exists()
    assert not (root / ".codex").exists()


def test_setup_pi_missing_prints_install_instructions(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(repo)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("must not run anything"))
    result = _setup()
    assert result.exit_code == 0, result.output
    assert f"pi install npm:{_PKG}" in result.output
    _assert_no_harness_files(repo)


def test_setup_pi_already_installed_runs_no_install(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(repo)
    calls = _fake_pi(monkeypatch, listing=f"User packages:\n  npm:{_PKG}\n")
    result = _setup(input="n\n")  # declines the starter guardrail
    assert result.exit_code == 0, result.output
    assert "already installed" in result.output
    assert calls == [[_PI, "list"]]
    _assert_no_harness_files(repo)


def test_setup_pi_installs_on_confirm(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(repo)
    calls = _fake_pi(monkeypatch, listing="No packages installed.\n", install_out="added pi-otari")
    result = _setup(input="y\nn\n")  # install, then decline the starter
    assert result.exit_code == 0, result.output
    assert calls == [[_PI, "list"], [_PI, "install", f"npm:{_PKG}"]]
    assert "added pi-otari" in result.output
    _assert_no_harness_files(repo)


def test_setup_pi_declined_does_not_install(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(repo)
    calls = _fake_pi(monkeypatch)
    result = _setup(input="n\nn\n")
    assert result.exit_code == 0, result.output
    assert calls == [[_PI, "list"]]
    assert "Skipping" in result.output


def test_setup_pi_install_failure_is_a_click_exception(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(repo)
    _fake_pi(monkeypatch, install_rc=1, install_out="npm ERR!")
    result = _setup(input="y\n")
    assert result.exit_code != 0
    assert "npm ERR!" in result.output
    assert "exited 1" in result.output


def test_setup_pi_offers_the_starter_guardrail_in_a_repo(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(repo)
    _fake_pi(monkeypatch, listing=_PKG)
    (repo / hook_cli.GUARDRAIL_FILE).unlink()
    result = _setup(input="y\n")
    assert result.exit_code == 0, result.output
    assert (repo / hook_cli.GUARDRAIL_FILE).is_file()
    _assert_no_harness_files(repo)


def test_setup_pi_outside_a_repo_installs_the_extension_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    calls = _fake_pi(monkeypatch)
    result = _setup(input="y\n")
    assert result.exit_code == 0, result.output
    assert calls[-1] == [_PI, "install", f"npm:{_PKG}"]
    assert list(tmp_path.iterdir()) == []
