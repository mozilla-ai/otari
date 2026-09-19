"""Unit tests for `otari hook setup`, the Claude Code settings installer.

`hook` became a Click group (invoke_without_command=True) so this subcommand
could live alongside the existing callback; tests/unit/test_hook_cli.py
already pins that the group-with-no-subcommand shape still behaves exactly
like the old plain command.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

import gateway.cli as gateway_cli
from gateway.core.config import GatewayConfig

_FAKE_OTARI_PATH = "/opt/otari/.venv/bin/otari"

_CHANGED_PATH_ONLY_GATES = (
    'schema_version: "1.0"\npolicy:\n  id: x\ngates:\n'
    "  - id: g\n    type: changed_path\n    enforcement: required\n"
    '    forbidden: ["CHANGELOG.md"]\n    message: m\n'
)

_COMMAND_MATCH_GATES = (
    'schema_version: "1.0"\npolicy:\n  id: x\ngates:\n'
    "  - id: g\n    type: command_match\n    enforcement: required\n"
    '    forbidden: ["npm"]\n    message: m\n'
)


@pytest.fixture(autouse=True)
def _fixed_otari_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway_cli, "_otari_binary_path", lambda: _FAKE_OTARI_PATH)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _invoke(*args: str, input: str | None = None) -> Any:  # noqa: A002 - matches CliRunner's own kwarg name
    return CliRunner().invoke(gateway_cli.hook, ["setup", *args], input=input)


def _read_settings(repo: Path) -> dict[str, Any]:
    parsed = json.loads((repo / ".claude" / "settings.local.json").read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


def test_fails_outside_a_git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = _invoke()
    assert result.exit_code != 0
    assert "Not inside a Git repository" in result.output


def test_declining_the_starter_policy_still_registers_the_hook(repo: Path) -> None:
    result = _invoke("--api-key", "k", input="n\n")
    assert result.exit_code == 0, result.output
    assert not (repo / ".otari-gates.yml").exists()
    settings = _read_settings(repo)
    entry = settings["hooks"]["PreToolUse"][0]
    assert entry["matcher"] == "Edit|Write|NotebookEdit"
    assert entry["hooks"][0]["command"] == f"{_FAKE_OTARI_PATH} hook --harness claude-code --api-key k"


def test_accepting_the_starter_policy_writes_one_and_its_command_match_gate_adds_bash(repo: Path) -> None:
    result = _invoke("--api-key", "k", input="y\n")
    assert result.exit_code == 0, result.output
    gates_file = repo / ".otari-gates.yml"
    assert gates_file.is_file()
    assert "command_match" in gates_file.read_text(encoding="utf-8")
    settings = _read_settings(repo)
    assert settings["hooks"]["PreToolUse"][0]["matcher"] == "Edit|Write|NotebookEdit|Bash"


def test_an_existing_changed_path_only_policy_keeps_bash_out_of_the_matcher(repo: Path) -> None:
    (repo / ".otari-gates.yml").write_text(_CHANGED_PATH_ONLY_GATES, encoding="utf-8")
    result = _invoke("--api-key", "k")
    assert result.exit_code == 0, result.output
    settings = _read_settings(repo)
    assert settings["hooks"]["PreToolUse"][0]["matcher"] == "Edit|Write|NotebookEdit"


def test_an_existing_command_match_policy_adds_bash_to_the_matcher(repo: Path) -> None:
    (repo / ".otari-gates.yml").write_text(_COMMAND_MATCH_GATES, encoding="utf-8")
    result = _invoke("--api-key", "k")
    assert result.exit_code == 0, result.output
    settings = _read_settings(repo)
    assert settings["hooks"]["PreToolUse"][0]["matcher"] == "Edit|Write|NotebookEdit|Bash"


def test_an_unparseable_policy_defaults_to_the_narrower_matcher(repo: Path) -> None:
    (repo / ".otari-gates.yml").write_text("not: valid: yaml: at: all:\n  - [", encoding="utf-8")
    result = _invoke("--api-key", "k")
    assert result.exit_code == 0, result.output
    settings = _read_settings(repo)
    assert settings["hooks"]["PreToolUse"][0]["matcher"] == "Edit|Write|NotebookEdit"


def test_explicit_api_key_flag_skips_resolution_and_prompting(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (repo / ".otari-gates.yml").write_text(_CHANGED_PATH_ONLY_GATES, encoding="utf-8")

    def fail_if_called(config_path: str | None = None) -> GatewayConfig:
        raise AssertionError("load_config should not be called when --api-key is given")

    monkeypatch.setattr(gateway_cli, "load_config", fail_if_called)
    result = _invoke("--api-key", "explicit-key")
    assert result.exit_code == 0, result.output
    settings = _read_settings(repo)
    command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert command == f"{_FAKE_OTARI_PATH} hook --harness claude-code --api-key explicit-key"


def test_no_api_key_means_no_credential_resolution_or_prompt(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no `--api-key`, the generated command carries no credential at all:

    `otari hook` evaluates the local policy in process by default and needs
    neither one, so `setup` must not resolve a `master_key` from config (even
    when one is configured) or prompt for anything to get the same result.
    """
    (repo / ".otari-gates.yml").write_text(_CHANGED_PATH_ONLY_GATES, encoding="utf-8")

    def fail_if_called(config_path: str | None = None) -> GatewayConfig:
        raise AssertionError("load_config should not be called when --api-key is not given either")

    monkeypatch.setattr(gateway_cli, "load_config", fail_if_called)
    result = _invoke()  # no --api-key, and no prompt input provided: must not be asked for one
    assert result.exit_code == 0, result.output
    settings = _read_settings(repo)
    command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert command == f"{_FAKE_OTARI_PATH} hook --harness claude-code"


def test_rerunning_updates_the_existing_entry_instead_of_duplicating_it(repo: Path) -> None:
    (repo / ".otari-gates.yml").write_text(_CHANGED_PATH_ONLY_GATES, encoding="utf-8")
    first = _invoke("--api-key", "first-key")
    assert first.exit_code == 0, first.output
    assert "Added" in first.output

    second = _invoke("--api-key", "second-key")
    assert second.exit_code == 0, second.output
    assert "Updated" in second.output

    settings = _read_settings(repo)
    assert len(settings["hooks"]["PreToolUse"]) == 1
    command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert command == f"{_FAKE_OTARI_PATH} hook --harness claude-code --api-key second-key"


def test_rerunning_preserves_unrelated_hooks_and_permissions(repo: Path) -> None:
    (repo / ".otari-gates.yml").write_text(_CHANGED_PATH_ONLY_GATES, encoding="utf-8")
    settings_path = repo / ".claude" / "settings.local.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "permissions": {"allow": ["Bash(git status)"]},
                "hooks": {
                    "PreToolUse": [
                        {"matcher": "Read", "hooks": [{"type": "command", "command": "/usr/bin/echo unrelated"}]}
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    result = _invoke("--api-key", "k")
    assert result.exit_code == 0, result.output

    settings = _read_settings(repo)
    assert settings["permissions"] == {"allow": ["Bash(git status)"]}
    matchers = {entry["matcher"] for entry in settings["hooks"]["PreToolUse"]}
    assert matchers == {"Read", "Edit|Write|NotebookEdit"}
    read_entry = next(e for e in settings["hooks"]["PreToolUse"] if e["matcher"] == "Read")
    assert read_entry["hooks"][0]["command"] == "/usr/bin/echo unrelated"


def test_registers_both_pretooluse_and_stop_hooks(repo: Path) -> None:
    result = _invoke("--api-key", "k", input="n\n")
    assert result.exit_code == 0, result.output
    settings = _read_settings(repo)

    assert "PreToolUse" in settings["hooks"]
    stop_entries = settings["hooks"]["Stop"]
    assert len(stop_entries) == 1
    stop_entry = stop_entries[0]
    # Stop is not tool-scoped: no matcher key at all, not a null one.
    assert "matcher" not in stop_entry
    assert stop_entry["hooks"][0]["command"] == settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]


def test_rerunning_updates_the_stop_entry_instead_of_duplicating_it(repo: Path) -> None:
    (repo / ".otari-gates.yml").write_text(_CHANGED_PATH_ONLY_GATES, encoding="utf-8")
    first = _invoke("--api-key", "first-key")
    assert first.exit_code == 0, first.output
    assert "Added the Stop hook" in first.output

    second = _invoke("--api-key", "second-key")
    assert second.exit_code == 0, second.output
    assert "Updated the Stop hook" in second.output

    settings = _read_settings(repo)
    assert len(settings["hooks"]["Stop"]) == 1
    command = settings["hooks"]["Stop"][0]["hooks"][0]["command"]
    assert command == f"{_FAKE_OTARI_PATH} hook --harness claude-code --api-key second-key"


def test_stop_hook_registration_preserves_an_unrelated_stop_entry(repo: Path) -> None:
    (repo / ".otari-gates.yml").write_text(_CHANGED_PATH_ONLY_GATES, encoding="utf-8")
    settings_path = repo / ".claude" / "settings.local.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps({"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "/usr/bin/echo unrelated"}]}]}}),
        encoding="utf-8",
    )

    result = _invoke("--api-key", "k")
    assert result.exit_code == 0, result.output

    settings = _read_settings(repo)
    commands = {entry["hooks"][0]["command"] for entry in settings["hooks"]["Stop"]}
    assert "/usr/bin/echo unrelated" in commands
    assert f"{_FAKE_OTARI_PATH} hook --harness claude-code --api-key k" in commands


def test_rejects_an_existing_settings_file_that_is_not_valid_json(repo: Path) -> None:
    (repo / ".otari-gates.yml").write_text(_CHANGED_PATH_ONLY_GATES, encoding="utf-8")
    settings_path = repo / ".claude" / "settings.local.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text("{not valid json", encoding="utf-8")

    result = _invoke("--api-key", "k")
    assert result.exit_code != 0
    assert "not valid JSON" in result.output
    # Untouched, not clobbered with a fresh empty structure.
    assert settings_path.read_text(encoding="utf-8") == "{not valid json"


def test_rejects_a_non_object_hooks_section(repo: Path) -> None:
    (repo / ".otari-gates.yml").write_text(_CHANGED_PATH_ONLY_GATES, encoding="utf-8")
    settings_path = repo / ".claude" / "settings.local.json"
    settings_path.parent.mkdir(parents=True)
    original = json.dumps({"hooks": ["not", "an", "object"]})
    settings_path.write_text(original, encoding="utf-8")

    result = _invoke("--api-key", "k")
    assert result.exit_code != 0
    assert '"hooks" must be a JSON object' in result.output
    assert settings_path.read_text(encoding="utf-8") == original


def test_rejects_a_non_array_pretooluse_list(repo: Path) -> None:
    (repo / ".otari-gates.yml").write_text(_CHANGED_PATH_ONLY_GATES, encoding="utf-8")
    settings_path = repo / ".claude" / "settings.local.json"
    settings_path.parent.mkdir(parents=True)
    original = json.dumps({"hooks": {"PreToolUse": "not-a-list"}})
    settings_path.write_text(original, encoding="utf-8")

    result = _invoke("--api-key", "k")
    assert result.exit_code != 0
    assert '"hooks.PreToolUse" must be a JSON array' in result.output
    assert settings_path.read_text(encoding="utf-8") == original


def test_rejects_a_non_array_stop_list(repo: Path) -> None:
    """_merge_hook_entry's own validation, generalized to whichever event it

    is called for: registering the Stop hook must reject a malformed
    hooks.Stop the same way registering PreToolUse already does, not just
    the one event this check happened to be written against first.

    Unlike the PreToolUse-malformed case, the file is not left byte-for-byte
    untouched: hook_setup registers PreToolUse first, which succeeds and
    writes the file, before it attempts Stop and fails. What must hold is
    narrower: the malformed hooks.Stop value itself is never touched, and
    PreToolUse is registered correctly despite the later failure.
    """
    (repo / ".otari-gates.yml").write_text(_CHANGED_PATH_ONLY_GATES, encoding="utf-8")
    settings_path = repo / ".claude" / "settings.local.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(json.dumps({"hooks": {"Stop": "not-a-list"}}), encoding="utf-8")

    result = _invoke("--api-key", "k")
    assert result.exit_code != 0
    assert '"hooks.Stop" must be a JSON array' in result.output

    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert settings["hooks"]["Stop"] == "not-a-list"
    assert (
        settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
        == f"{_FAKE_OTARI_PATH} hook --harness claude-code --api-key k"
    )
