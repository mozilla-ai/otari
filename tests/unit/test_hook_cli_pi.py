"""Unit tests for `otari hook --harness pi` and `otari hook setup --harness pi`.

pi's payload carries pi-native tool names (`bash`, `read`, `edit`, `write`) and
`tool_input` (`{command}` / `{path}`), and, on Stop, the session's commands as
`bash_commands` instead of a transcript the hook would parse. The pi-otari
extension is what registers the hook with pi; see docs/agent-guardrails.md.
"""

import json
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


def test_setup_writes_nothing_for_pi(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(repo)
    result = CliRunner().invoke(hook_cli.hook, ["setup", "--harness", "pi"])
    assert result.exit_code == 0, result.output
    assert "pi-otari" in result.output
    assert not (repo / ".claude").exists()
    assert not (repo / ".codex").exists()
