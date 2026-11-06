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

import otari_agent.hook as hook_cli

pytestmark = pytest.mark.usefixtures("isolated_home", "no_otari_env")


def _guardrail_path(root: Path) -> Path:
    """`.otari/guardrails.yml` under `root`, with its parent directory created."""
    path = root / hook_cli.GUARDRAIL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


_FAKE_OTARI_PATH = "/opt/otari/.venv/bin/otari"

_PATH_ONLY_GATES = (
    'schema_version: "1.0"\npolicy:\n  id: x\ngates:\n'
    "  - id: g\n    type: path\n"
    "    runs: [pre_tool_use.edit_target, stop.working_tree]\n    enforcement: required\n"
    '    forbidden: ["CHANGELOG.md"]\n    message: m\n'
)

_READ_GATES = (
    'schema_version: "1.0"\npolicy:\n  id: x\ngates:\n'
    "  - id: g\n    type: path\n"
    "    runs: [pre_tool_use.read_target]\n    enforcement: required\n"
    '    forbidden: ["**/.env"]\n    message: m\n'
)

_COMMAND_GATES = (
    'schema_version: "1.0"\npolicy:\n  id: x\ngates:\n'
    "  - id: g\n    type: command\n    runs: [pre_tool_use.command]\n    enforcement: required\n"
    '    forbidden: ["npm"]\n    message: m\n'
)


@pytest.fixture(autouse=True)
def _fixed_otari_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hook_cli, "_otari_binary_path", lambda: _FAKE_OTARI_PATH)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _invoke(*args: str, input: str | None = None) -> Any:  # noqa: A002 - matches CliRunner's own kwarg name
    return CliRunner().invoke(hook_cli.hook, ["setup", *args], input=input)


def _read_settings(repo: Path) -> dict[str, Any]:
    parsed = json.loads((repo / ".claude" / "settings.local.json").read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


def test_fails_outside_a_git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = _invoke()
    assert result.exit_code != 0
    assert "Not inside a Git repository" in result.output


def test_declining_the_starter_policy_says_the_user_level_gates_still_run(repo: Path, isolated_home: Path) -> None:
    _guardrail_path(isolated_home).write_text(_COMMAND_GATES, encoding="utf-8")
    result = _invoke(input="n\n")
    assert result.exit_code == 0, result.output
    assert "every gate check passes" not in result.output
    assert "your own gates in ~/.otari/" in result.output


def test_declining_the_starter_policy_still_registers_the_hook(repo: Path) -> None:
    result = _invoke(input="n\n")
    assert result.exit_code == 0, result.output
    assert not _guardrail_path(repo).exists()
    settings = _read_settings(repo)
    entry = settings["hooks"]["PreToolUse"][0]
    assert entry["matcher"] == "Edit|Write|NotebookEdit"
    assert entry["hooks"][0]["command"] == f"{_FAKE_OTARI_PATH} hook --harness claude-code"


def test_accepting_the_starter_policy_writes_one_and_its_command_gate_adds_bash(repo: Path) -> None:
    """The single file, the smaller of the two shapes to start with."""
    result = _invoke(input="y\n")
    assert result.exit_code == 0, result.output
    gates_file = repo / hook_cli.GUARDRAIL_FILE
    assert gates_file.is_file()
    assert not (repo / hook_cli.GUARDRAIL_DIR).is_dir()
    assert "command" in gates_file.read_text(encoding="utf-8")
    settings = _read_settings(repo)
    assert settings["hooks"]["PreToolUse"][0]["matcher"] == "Edit|Write|NotebookEdit|Bash"


def test_an_existing_path_only_policy_keeps_bash_out_of_the_matcher(repo: Path) -> None:
    _guardrail_path(repo).write_text(_PATH_ONLY_GATES, encoding="utf-8")
    result = _invoke()
    assert result.exit_code == 0, result.output
    settings = _read_settings(repo)
    assert settings["hooks"]["PreToolUse"][0]["matcher"] == "Edit|Write|NotebookEdit"


def test_an_existing_command_policy_adds_bash_to_the_matcher(repo: Path) -> None:
    _guardrail_path(repo).write_text(_COMMAND_GATES, encoding="utf-8")
    result = _invoke()
    assert result.exit_code == 0, result.output
    settings = _read_settings(repo)
    assert settings["hooks"]["PreToolUse"][0]["matcher"] == "Edit|Write|NotebookEdit|Bash"


def test_an_existing_read_policy_adds_read_to_the_matcher(repo: Path) -> None:
    """A read gate is the one `runs` value that earns its own matcher group.

    Nothing else distinguishes it: a read gate and an edit gate are the same
    `path` type, so the matcher has to be picked from `runs` rather than from
    the gate type the way Bash's is.
    """
    _guardrail_path(repo).write_text(_READ_GATES, encoding="utf-8")
    result = _invoke()
    assert result.exit_code == 0, result.output
    settings = _read_settings(repo)
    assert settings["hooks"]["PreToolUse"][0]["matcher"] == "Edit|Write|NotebookEdit|Read"


def test_a_user_level_read_gate_adds_read_to_the_matcher(repo: Path, isolated_home: Path) -> None:
    """The hook composes the user's own gates in this repo, so the matcher has to reach them too."""
    _guardrail_path(repo).write_text(_PATH_ONLY_GATES, encoding="utf-8")
    _guardrail_path(isolated_home).write_text(_READ_GATES.replace("id: g\n", "id: personal\n"), encoding="utf-8")
    result = _invoke()
    assert result.exit_code == 0, result.output
    assert _read_settings(repo)["hooks"]["PreToolUse"][0]["matcher"] == "Edit|Write|NotebookEdit|Read"


def test_repo_only_writes_the_flag_into_the_registration(repo: Path) -> None:
    """The flag has to reach the registered command, or a later `setup` run silently drops it.

    `_merge_hook_entry` matches an existing entry on the binary and "hook" alone and then replaces
    the whole command string, so a flag added to the file by hand does not survive a re-run.
    """
    _guardrail_path(repo).write_text(_PATH_ONLY_GATES, encoding="utf-8")
    result = _invoke("--repo-only")
    assert result.exit_code == 0, result.output
    settings = _read_settings(repo)
    for event in ("PreToolUse", "Stop"):
        command = settings["hooks"][event][0]["hooks"][0]["command"]
        assert command == f"{_FAKE_OTARI_PATH} hook --harness claude-code --repo-only"


def test_without_repo_only_the_registration_carries_no_such_flag(repo: Path) -> None:
    _guardrail_path(repo).write_text(_PATH_ONLY_GATES, encoding="utf-8")
    result = _invoke()
    assert result.exit_code == 0, result.output
    command = _read_settings(repo)["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert command == f"{_FAKE_OTARI_PATH} hook --harness claude-code"


def test_repo_only_keeps_a_user_level_read_gate_out_of_the_matcher(repo: Path, isolated_home: Path) -> None:
    """The matcher is picked from the gates the hook will run, so `--repo-only` must narrow it too.

    Without this, a personal read gate widens the matcher for a repo that never checks it, and the
    hook is invoked on every Read for nothing.
    """
    _guardrail_path(repo).write_text(_PATH_ONLY_GATES, encoding="utf-8")
    _guardrail_path(isolated_home).write_text(_READ_GATES.replace("id: g\n", "id: personal\n"), encoding="utf-8")
    result = _invoke("--repo-only")
    assert result.exit_code == 0, result.output
    assert _read_settings(repo)["hooks"]["PreToolUse"][0]["matcher"] == "Edit|Write|NotebookEdit"


def _write_global_registration(isolated_home: Path, *, repo_only: bool = False, command: str | None = None) -> Path:
    """A claude-code hook entry in `~/.claude/settings.json`, as if registered for every repository."""
    path = isolated_home / ".claude" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if command is None:
        command = f"{_FAKE_OTARI_PATH} hook --harness claude-code" + (" --repo-only" if repo_only else "")
    path.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {"matcher": "Edit|Write|NotebookEdit", "hooks": [{"type": "command", "command": command}]}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_repo_only_warns_of_a_global_registration_without_it(repo: Path, isolated_home: Path) -> None:
    """A global registration that still omits the flag would still enforce personal gates here."""
    global_settings = _write_global_registration(isolated_home)
    _guardrail_path(repo).write_text(_PATH_ONLY_GATES, encoding="utf-8")
    result = _invoke("--repo-only")
    assert result.exit_code == 0, result.output
    assert str(global_settings) in result.output
    assert "--repo-only" in result.output


def test_repo_only_does_not_warn_when_the_global_registration_already_has_it(repo: Path, isolated_home: Path) -> None:
    global_settings = _write_global_registration(isolated_home, repo_only=True)
    _guardrail_path(repo).write_text(_PATH_ONLY_GATES, encoding="utf-8")
    result = _invoke("--repo-only")
    assert result.exit_code == 0, result.output
    assert str(global_settings) not in result.output


def test_without_repo_only_a_global_registration_draws_no_warning(repo: Path, isolated_home: Path) -> None:
    """Both registrations already enforce the same gates here, so there is nothing to warn about."""
    global_settings = _write_global_registration(isolated_home)
    _guardrail_path(repo).write_text(_PATH_ONLY_GATES, encoding="utf-8")
    result = _invoke()
    assert result.exit_code == 0, result.output
    assert str(global_settings) not in result.output


def test_repo_only_warns_of_a_global_registration_that_names_otari_on_the_path(repo: Path, isolated_home: Path) -> None:
    """The entry runs the same hook whether or not it spells otari as an absolute path."""
    global_settings = _write_global_registration(isolated_home, command="otari hook --harness claude-code")
    _guardrail_path(repo).write_text(_PATH_ONLY_GATES, encoding="utf-8")
    result = _invoke("--repo-only")
    assert result.exit_code == 0, result.output
    assert str(global_settings) in result.output


def test_repo_only_ignores_a_global_entry_that_does_not_run_otari_hook(repo: Path, isolated_home: Path) -> None:
    global_settings = _write_global_registration(isolated_home, command="otari guardrails check")
    _guardrail_path(repo).write_text(_PATH_ONLY_GATES, encoding="utf-8")
    result = _invoke("--repo-only")
    assert result.exit_code == 0, result.output
    assert str(global_settings) not in result.output


def test_the_global_registration_warning_names_the_cost_and_the_alternative(repo: Path, isolated_home: Path) -> None:
    """Flagging the global entry turns the person's gates off everywhere, so the warning says so."""
    _write_global_registration(isolated_home)
    _guardrail_path(repo).write_text(_PATH_ONLY_GATES, encoding="utf-8")
    result = _invoke("--repo-only")
    assert result.exit_code == 0, result.output
    assert "every repository" in result.output
    assert "per repository" in result.output


def test_a_broken_repo_file_still_leaves_the_user_level_gates_in_the_matcher(repo: Path, isolated_home: Path) -> None:
    """The hook still enforces the user's gates when the repo's files fail, so it must still be called for them."""
    _guardrail_path(repo).write_text("gates: [", encoding="utf-8")
    _guardrail_path(isolated_home).write_text(_COMMAND_GATES, encoding="utf-8")
    result = _invoke()
    assert result.exit_code == 0, result.output
    assert _read_settings(repo)["hooks"]["PreToolUse"][0]["matcher"] == "Edit|Write|NotebookEdit|Bash"


def test_a_policy_with_no_read_gate_keeps_read_out_of_the_matcher(repo: Path) -> None:
    _guardrail_path(repo).write_text(_PATH_ONLY_GATES, encoding="utf-8")
    result = _invoke()
    assert result.exit_code == 0, result.output
    assert "Read" not in _read_settings(repo)["hooks"]["PreToolUse"][0]["matcher"]
    assert "pre_tool_use.read_target to also cover Read" in result.output


def test_a_read_and_command_policy_earns_both_extra_groups(repo: Path) -> None:
    combined = _READ_GATES + (
        "  - id: c\n    type: command\n    runs: [pre_tool_use.command]\n"
        '    enforcement: required\n    forbidden: ["npm install"]\n    message: m\n'
    )
    _guardrail_path(repo).write_text(combined, encoding="utf-8")
    result = _invoke()
    assert result.exit_code == 0, result.output
    settings = _read_settings(repo)
    assert settings["hooks"]["PreToolUse"][0]["matcher"] == "Edit|Write|NotebookEdit|Read|Bash"


def test_codex_has_no_read_group_to_add(repo: Path) -> None:
    """Codex has no read tool: its reads go through the shell, where they are command evidence."""
    _guardrail_path(repo).write_text(_READ_GATES, encoding="utf-8")
    result = _invoke("--harness", "codex")
    assert result.exit_code == 0, result.output
    settings = json.loads((repo / ".codex" / "hooks.json").read_text(encoding="utf-8"))
    assert settings["hooks"]["PreToolUse"][0]["matcher"] == "apply_patch"
    assert "Read" not in result.output


def test_an_unparseable_policy_defaults_to_the_narrower_matcher(repo: Path) -> None:
    _guardrail_path(repo).write_text("not: valid: yaml: at: all:\n  - [", encoding="utf-8")
    result = _invoke()
    assert result.exit_code == 0, result.output
    settings = _read_settings(repo)
    assert settings["hooks"]["PreToolUse"][0]["matcher"] == "Edit|Write|NotebookEdit"


def test_the_registered_command_carries_no_credential(repo: Path) -> None:
    """`otari hook` evaluates in process, so `setup` registers no credential.

    Nothing is resolved from config and nothing is prompted for: no input is
    provided here, so a prompt would hang the run rather than pass it.
    """
    _guardrail_path(repo).write_text(_PATH_ONLY_GATES, encoding="utf-8")
    result = _invoke()
    assert result.exit_code == 0, result.output
    settings = _read_settings(repo)
    command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert command == f"{_FAKE_OTARI_PATH} hook --harness claude-code"


def test_rerunning_updates_the_existing_entry_instead_of_duplicating_it(repo: Path) -> None:
    _guardrail_path(repo).write_text(_PATH_ONLY_GATES, encoding="utf-8")
    first = _invoke()
    assert first.exit_code == 0, first.output
    assert "Added" in first.output

    second = _invoke()
    assert second.exit_code == 0, second.output
    assert "Updated" in second.output

    settings = _read_settings(repo)
    assert len(settings["hooks"]["PreToolUse"]) == 1
    command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert command == f"{_FAKE_OTARI_PATH} hook --harness claude-code"


def test_a_narrowed_matcher_leaves_a_shared_entrys_sibling_command_alone(repo: Path) -> None:
    """A matcher belongs to its entry, so narrowing ours must not narrow a sibling command's."""
    settings = repo / ".claude" / "settings.local.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Edit|Write|NotebookEdit|Read",
                            "hooks": [
                                {"type": "command", "command": f"{_FAKE_OTARI_PATH} hook --harness claude-code"},
                                {"type": "command", "command": "sibling-hook"},
                            ],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    _guardrail_path(repo).write_text(_PATH_ONLY_GATES, encoding="utf-8")

    result = _invoke("--repo-only")
    assert result.exit_code == 0, result.output

    entries = _read_settings(repo)["hooks"]["PreToolUse"]
    shared = next(entry for entry in entries if any(h["command"] == "sibling-hook" for h in entry["hooks"]))
    assert shared["matcher"] == "Edit|Write|NotebookEdit|Read"
    assert [h["command"] for h in shared["hooks"]] == ["sibling-hook"]

    ours = next(entry for entry in entries if any("otari" in h["command"] for h in entry["hooks"]))
    assert ours is not shared
    assert ours["matcher"] == "Edit|Write|NotebookEdit"
    assert ours["hooks"] == [
        {"type": "command", "command": f"{_FAKE_OTARI_PATH} hook --harness claude-code --repo-only"}
    ]


def test_splitting_a_shared_entry_keeps_the_hook_items_other_fields(repo: Path) -> None:
    """Only the command is setup's to rewrite, so a field a person added to the item survives the move."""
    settings = repo / ".claude" / "settings.local.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Edit|Write|NotebookEdit|Read",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": f"{_FAKE_OTARI_PATH} hook --harness claude-code",
                                    "timeout": 30,
                                },
                                {"type": "command", "command": "sibling-hook"},
                            ],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    _guardrail_path(repo).write_text(_PATH_ONLY_GATES, encoding="utf-8")

    assert _invoke("--repo-only").exit_code == 0

    entries = _read_settings(repo)["hooks"]["PreToolUse"]
    ours = next(entry for entry in entries if any("otari" in h["command"] for h in entry["hooks"]))
    assert ours["hooks"] == [
        {
            "type": "command",
            "command": f"{_FAKE_OTARI_PATH} hook --harness claude-code --repo-only",
            "timeout": 30,
        }
    ]


def test_rerunning_still_updates_a_solo_entry_in_place(repo: Path) -> None:
    _guardrail_path(repo).write_text(_PATH_ONLY_GATES, encoding="utf-8")
    assert _invoke().exit_code == 0
    assert _invoke("--repo-only").exit_code == 0
    entries = _read_settings(repo)["hooks"]["PreToolUse"]
    assert len(entries) == 1
    assert entries[0]["hooks"][0]["command"].endswith("--repo-only")


def test_rerunning_preserves_unrelated_hooks_and_permissions(repo: Path) -> None:
    _guardrail_path(repo).write_text(_PATH_ONLY_GATES, encoding="utf-8")
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

    result = _invoke()
    assert result.exit_code == 0, result.output

    settings = _read_settings(repo)
    assert settings["permissions"] == {"allow": ["Bash(git status)"]}
    matchers = {entry["matcher"] for entry in settings["hooks"]["PreToolUse"]}
    assert matchers == {"Read", "Edit|Write|NotebookEdit"}
    read_entry = next(e for e in settings["hooks"]["PreToolUse"] if e["matcher"] == "Read")
    assert read_entry["hooks"][0]["command"] == "/usr/bin/echo unrelated"


def test_registers_both_pretooluse_and_stop_hooks(repo: Path) -> None:
    result = _invoke(input="n\n")
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
    _guardrail_path(repo).write_text(_PATH_ONLY_GATES, encoding="utf-8")
    first = _invoke()
    assert first.exit_code == 0, first.output
    assert "Added the Stop hook" in first.output

    second = _invoke()
    assert second.exit_code == 0, second.output
    assert "Updated the Stop hook" in second.output

    settings = _read_settings(repo)
    assert len(settings["hooks"]["Stop"]) == 1
    command = settings["hooks"]["Stop"][0]["hooks"][0]["command"]
    assert command == f"{_FAKE_OTARI_PATH} hook --harness claude-code"


def test_stop_hook_registration_preserves_an_unrelated_stop_entry(repo: Path) -> None:
    _guardrail_path(repo).write_text(_PATH_ONLY_GATES, encoding="utf-8")
    settings_path = repo / ".claude" / "settings.local.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps({"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "/usr/bin/echo unrelated"}]}]}}),
        encoding="utf-8",
    )

    result = _invoke()
    assert result.exit_code == 0, result.output

    settings = _read_settings(repo)
    commands = {entry["hooks"][0]["command"] for entry in settings["hooks"]["Stop"]}
    assert "/usr/bin/echo unrelated" in commands
    assert f"{_FAKE_OTARI_PATH} hook --harness claude-code" in commands


def test_rejects_an_existing_settings_file_that_is_not_valid_json(repo: Path) -> None:
    _guardrail_path(repo).write_text(_PATH_ONLY_GATES, encoding="utf-8")
    settings_path = repo / ".claude" / "settings.local.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text("{not valid json", encoding="utf-8")

    result = _invoke()
    assert result.exit_code != 0
    assert "not valid JSON" in result.output
    # Untouched, not clobbered with a fresh empty structure.
    assert settings_path.read_text(encoding="utf-8") == "{not valid json"


def test_rejects_a_non_object_hooks_section(repo: Path) -> None:
    _guardrail_path(repo).write_text(_PATH_ONLY_GATES, encoding="utf-8")
    settings_path = repo / ".claude" / "settings.local.json"
    settings_path.parent.mkdir(parents=True)
    original = json.dumps({"hooks": ["not", "an", "object"]})
    settings_path.write_text(original, encoding="utf-8")

    result = _invoke()
    assert result.exit_code != 0
    assert '"hooks" must be a JSON object' in result.output
    assert settings_path.read_text(encoding="utf-8") == original


def test_rejects_a_non_array_pretooluse_list(repo: Path) -> None:
    _guardrail_path(repo).write_text(_PATH_ONLY_GATES, encoding="utf-8")
    settings_path = repo / ".claude" / "settings.local.json"
    settings_path.parent.mkdir(parents=True)
    original = json.dumps({"hooks": {"PreToolUse": "not-a-list"}})
    settings_path.write_text(original, encoding="utf-8")

    result = _invoke()
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
    _guardrail_path(repo).write_text(_PATH_ONLY_GATES, encoding="utf-8")
    settings_path = repo / ".claude" / "settings.local.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(json.dumps({"hooks": {"Stop": "not-a-list"}}), encoding="utf-8")

    result = _invoke()
    assert result.exit_code != 0
    assert '"hooks.Stop" must be a JSON array' in result.output

    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert settings["hooks"]["Stop"] == "not-a-list"
    assert settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == f"{_FAKE_OTARI_PATH} hook --harness claude-code"


def test_setup_leaves_an_existing_directory_guardrail_alone(repo: Path) -> None:
    existing = repo / hook_cli.GUARDRAIL_DIR / "mine.yml"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text(_PATH_ONLY_GATES, encoding="utf-8")
    result = _invoke()
    assert result.exit_code == 0, result.output
    assert not (repo / hook_cli.GUARDRAIL_DIR / "starter.yml").exists()
    assert existing.read_text(encoding="utf-8") == _PATH_ONLY_GATES


def test_a_command_gate_in_a_composed_directory_still_widens_the_matcher(repo: Path) -> None:
    command_gate = (
        'schema_version: "1.0"\npolicy:\n  id: demo/shell\ngates:\n'
        "  - id: no-force-push\n"
        "    type: command\n"
        "    runs: [pre_tool_use.command]\n"
        "    enforcement: advisory\n"
        '    forbidden: ["git push --force"]\n'
        "    message: m\n"
    )
    for relative, body in ((hook_cli.GUARDRAIL_FILE, _PATH_ONLY_GATES), (".otari/guardrails/shell.yml", command_gate)):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")

    result = _invoke()
    assert result.exit_code == 0, result.output
    assert _read_settings(repo)["hooks"]["PreToolUse"][0]["matcher"] == "Edit|Write|NotebookEdit|Bash"
