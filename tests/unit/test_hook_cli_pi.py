"""Unit tests for `otari hook --harness pi` and `otari hook setup --harness pi`.

pi's payload carries pi-native tool names (`bash`, `read`, `edit`, `write`) and
`tool_input` (`{command}` / `{path}`), and, on Stop, the session's commands as
`bash_commands` instead of a transcript the hook would parse. The pi-otari
extension is what registers the hook with pi; see cli/docs/agent-guardrails.md.
"""

import json
import shutil
import subprocess
import sys
import time
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


def test_tilde_name_path_is_not_expanded(repo: Path) -> None:
    # `~name` is not pi's `~/` form; expanduser would treat it as another user's home and raise for an unknown one.
    (repo / "CHANGELOG.md").write_text("", encoding="utf-8")
    result = _invoke(_pre(repo, "edit", {"path": "~nosuchuser-xyz/CHANGELOG.md"}))
    assert result.exit_code == 0, result.output
    assert hook_cli._hook_pi_normalize_path("~nosuchuser-xyz/x", "pi") == "~nosuchuser-xyz/x"


@pytest.mark.parametrize(
    "alias",
    [".env", "@.env", "@./.env", "file://{repo}/.env", "file://localhost{repo}/.env", "file://LOCALHOST{repo}/.env"],
)
def test_pi_path_aliases_reach_the_read_gate(repo: Path, alias: str) -> None:
    """pi strips a leading `@` and opens a `file://` URL's path, so each of these reads `.env`."""
    result = _invoke(_pre(repo, "read", {"path": alias.format(repo=repo)}))
    assert result.exit_code == 2, result.output


def test_pi_path_aliases_reach_the_edit_gate(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    monkeypatch.setenv("HOME", str(repo.parent))
    for alias in ("@CHANGELOG.md", f"@~/{repo.name}/CHANGELOG.md", f"@{repo}/CHANGELOG.md"):
        result = _invoke(_pre(repo, "write", {"path": alias}))
        assert result.exit_code == 2, (alias, result.output)


def test_a_remote_file_url_is_left_alone() -> None:
    assert hook_cli._hook_pi_normalize_path("file://example.com/x/.env", "pi") == "file://example.com/x/.env"


def test_windows_home_form_is_expanded_only_on_windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(sys, "platform", "win32")
    assert hook_cli._hook_pi_normalize_path("~\\project\\.env", "pi") == str(tmp_path / "project\\.env")
    assert hook_cli._hook_pi_normalize_path("~name\\x", "pi") == "~name\\x"
    assert hook_cli._hook_pi_normalize_path("/c/Users/a/.env", "pi") == "C:\\Users\\a\\.env"
    assert hook_cli._hook_pi_normalize_path("@/mnt/d/x", "pi") == "D:\\x"
    assert hook_cli._hook_pi_normalize_path("/cygdrive/e", "pi") == "E:\\"
    assert hook_cli._hook_pi_normalize_path("/usr/local", "pi") == "/usr/local"
    monkeypatch.setattr(sys, "platform", "linux")
    assert hook_cli._hook_pi_normalize_path("~\\project", "pi") == "~\\project"
    assert hook_cli._hook_pi_normalize_path("/c/Users", "pi") == "/c/Users"


def test_pi_normalizes_unicode_spaces_like_pi() -> None:
    assert hook_cli._hook_pi_normalize_path("my\u00a0notes.md", "pi") == "my notes.md"
    assert hook_cli._hook_pi_normalize_path("@@x", "pi") == "@x"
    assert hook_cli._hook_pi_normalize_path("@.env", "claude-code") == "@.env"


def test_nested_judge_call_exits_before_evaluating(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    # A pi judge call runs the pi-otari Stop hook again; the marker ends that even where the repo root is found.
    monkeypatch.setenv(hook_cli._HOOK_NESTED_ENV, "1")
    result = _invoke(_pre(repo, "edit", {"path": "CHANGELOG.md"}))
    assert result.exit_code == 0, result.output
    assert result.output == ""


def test_judge_and_generate_calls_carry_the_nested_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append(kwargs["env"])
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    hook_cli._hook_run_judge_subprocess(["pi"], "p", deadline=time.monotonic() + 60, label="pi -p")
    hook_cli._gates_generate_run_cli(["pi"], "p", label="pi -p")
    assert [env[hook_cli._HOOK_NESTED_ENV] for env in seen] == ["1", "1"]


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


_LINT_GATE_YAML = (
    'schema_version: "1.0"\npolicy:\n  id: test\ngates:\n'
    "  - id: lint-after-python\n    type: command_if_changed\n"
    "    runs: [stop.session]\n    enforcement: required\n"
    '    when_changed: ["**/*.py"]\n    require: ["make lint"]\n    message: run make lint\n'
)


@pytest.mark.parametrize(
    ("bash_commands", "exit_code", "reason"),
    [(["pytest -q", "make lint"], 0, ""), ([], 2, "run make lint"), (None, 2, "evidence was not submitted")],
    ids=["ran", "empty-session", "omitted"],
)
def test_stop_enforces_a_required_command_gate(
    monkeypatch: pytest.MonkeyPatch, repo: Path, bash_commands: list[str] | None, exit_code: int, reason: str
) -> None:
    """Neither a session that ran nothing nor one whose commands were not sent satisfies a required gate."""
    (repo / hook_cli.GUARDRAIL_FILE).write_text(_LINT_GATE_YAML, encoding="utf-8")
    monkeypatch.setattr(hook_cli, "_hook_collect_changed_paths", lambda root: ["src/x.py"])
    payload: dict[str, Any] = {"hook_event_name": "Stop", "cwd": str(repo), "stop_hook_active": False}
    if bash_commands is not None:
        payload["bash_commands"] = bash_commands
    result = _invoke(payload)
    assert result.exit_code == exit_code, result.output
    assert "could not load" not in result.output
    assert reason in result.output


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
            assert kwargs.get("cwd") == hook_cli._hook_judge_workdir(), "listed from the judge workdir"
            return subprocess.CompletedProcess(argv, 0, stdout=listing, stderr="")
        return subprocess.CompletedProcess(argv, install_rc, stdout=install_out, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: _PI if name == "pi" else None)
    return calls


def _setup(input: str | None = None, *extra: str) -> Any:  # noqa: A002
    return CliRunner().invoke(hook_cli.hook, ["setup", "--harness", "pi", *extra], input=input)


def _user_listing(tmp_path: Path, version: str | None, source: str = f"npm:{_PKG}") -> str:
    install = tmp_path / "pi-npm" / _PKG
    install.mkdir(parents=True, exist_ok=True)
    if version is not None:
        (install / "package.json").write_text(json.dumps({"name": _PKG, "version": version}), encoding="utf-8")
    return f"User packages:\n  {source}\n    {install}\n"


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
    calls = _fake_pi(monkeypatch, listing=_user_listing(repo.parent, "0.7.2"))
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
    _fake_pi(monkeypatch, listing=_user_listing(repo.parent, "0.7.2"))
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


def test_setup_pi_a_project_only_install_is_not_enough(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The judge workdir cannot load a project's packages, so only a user-wide install counts."""
    monkeypatch.chdir(repo)
    listing = f"Project packages:\n  npm:{_PKG}\n    {repo}/.pi/npm/node_modules/{_PKG}\n"
    calls = _fake_pi(monkeypatch, listing=listing)
    result = _setup(input="y\n")
    assert result.exit_code == 0, result.output
    assert calls[-1] == [_PI, "install", f"npm:{_PKG}"]


def test_setup_pi_offers_to_update_a_version_without_guardrails(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(repo)
    calls = _fake_pi(monkeypatch, listing=_user_listing(repo.parent, "0.7.0"), install_out="updated")
    result = _setup(input="y\n")
    assert result.exit_code == 0, result.output
    assert "0.7.0" in result.output
    assert calls == [[_PI, "list"], [_PI, "update", f"npm:{_PKG}"]]
    assert "already installed" not in result.output


def test_setup_pi_a_pinned_old_version_is_an_error(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(repo)
    calls = _fake_pi(monkeypatch, listing=_user_listing(repo.parent, "0.5.0", source=f"npm:{_PKG}@0.5.0"))
    result = _setup()
    assert result.exit_code != 0
    assert f"pi remove npm:{_PKG}@0.5.0" in result.output
    assert calls == [[_PI, "list"]]


def test_setup_pi_refuses_repo_only(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(repo)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("must not run anything"))
    result = _setup(None, "--repo-only")
    assert result.exit_code != 0
    assert "--repo-only is not supported with --harness pi" in result.output
