"""Unit tests for `otari hook --harness codex` and `otari hook setup --harness codex`.

Codex's own payload/transcript shapes differ from Claude Code's (see
`_HOOK_COMMAND_TOOL_FIELDS_BY_HARNESS`, `_CODEX_PATCH_TOOL_NAME`,
`_hook_collect_codex_transcript_commands`, `_hook_extract_codex_judge_transcript`
in `gateway.cli`); this file covers those, the same way
`tests/unit/test_hook_cli.py` and `tests/unit/test_hook_setup_cli.py` cover
the Claude Code harness. Mocks the network boundary (httpx.post) and the Git
boundary (subprocess.run), same as test_hook_cli.py.
"""

import json
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

import otari_agent.hook as hook_cli

_GATES_YAML = "schema_version: '1.0'\npolicy:\n  id: test\ngates: []\n"

_FAKE_OTARI_PATH = "/opt/otari/.venv/bin/otari"


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._payload


@pytest.fixture(autouse=True)
def _judge_log_in_tmp_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(hook_cli, "_hook_judge_log_path", lambda: tmp_path / "judge-calls.log")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".otari-guardrails.yml").write_text(_GATES_YAML, encoding="utf-8")
    return tmp_path


def _invoke(payload: dict[str, Any], **extra_args: str) -> Any:
    args = ["--api-key", "test-key", "--harness", "codex"]
    for key, value in extra_args.items():
        args += [f"--{key.replace('_', '-')}", value]
    return CliRunner().invoke(hook_cli.hook, args, input=json.dumps(payload))


# --- PreToolUse: apply_patch (Codex's own edit tool) ------------------------


def test_pretooluse_extracts_a_single_path_from_an_apply_patch_envelope(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse(
            {
                "blocked": True,
                "results": [{"gate_id": "g", "enforcement": "required", "outcome": "fail", "message": "no"}],
            }
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    patch_text = "*** Begin Patch\n*** Update File: CHANGELOG.md\n@@\n-old\n+new\n*** End Patch"
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "apply_patch",
        "tool_input": {"command": patch_text},
    }
    result = _invoke(payload)
    assert result.exit_code == 2, result.output
    assert captured["json"]["changed_paths"] == ["CHANGELOG.md"]
    assert captured["json"]["commands"] == []


def test_pretooluse_apply_patch_covers_every_file_it_touches(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    patch_text = (
        "*** Begin Patch\n"
        "*** Add File: src/new_module.py\n"
        "+content\n"
        "*** Update File: README.md\n"
        "@@\n-old\n+new\n"
        "*** Delete File: old_file.py\n"
        "*** End Patch"
    )
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "apply_patch",
        "tool_input": {"command": patch_text},
    }
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    assert captured["json"]["changed_paths"] == ["src/new_module.py", "README.md", "old_file.py"]


def test_pretooluse_apply_patch_rename_reports_both_old_and_new_path(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    """A rename is "*** Update File: <old>" immediately followed by "*** Move to: <new>",
    neither line alone naming where the file ends up; a gate scoped to either path should see it.
    """
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    patch_text = (
        "*** Begin Patch\n*** Update File: old_name.py\n*** Move to: new_name.py\n@@\n-old\n+new\n*** End Patch"
    )
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "apply_patch",
        "tool_input": {"command": patch_text},
    }
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    assert captured["json"]["changed_paths"] == ["old_name.py", "new_name.py"]


def test_pretooluse_ignores_an_apply_patch_with_no_command(repo: Path) -> None:
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "apply_patch",
        "tool_input": {},
    }
    result = _invoke(payload)
    assert result.exit_code == 0, result.output


def test_pretooluse_apply_patch_with_no_recognizable_header_is_a_no_op(repo: Path) -> None:
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "apply_patch",
        "tool_input": {"command": "not a real patch envelope"},
    }
    result = _invoke(payload)
    assert result.exit_code == 0, result.output


def test_pretooluse_apply_patch_path_outside_the_repo_is_skipped(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    patch_text = "*** Begin Patch\n*** Update File: /etc/passwd\n*** Update File: CHANGELOG.md\n*** End Patch"
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "apply_patch",
        "tool_input": {"command": patch_text},
    }
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    assert captured["json"]["changed_paths"] == ["CHANGELOG.md"]


# --- PreToolUse: shell / Code Mode exec -------------------------------------


def test_pretooluse_submits_a_bash_command_for_codex(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse(
            {
                "blocked": True,
                "results": [
                    {"gate_id": "no-force-push", "enforcement": "required", "outcome": "fail", "message": "no"}
                ],
            }
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "Bash",
        "tool_input": {"command": "git push --force"},
    }
    result = _invoke(payload)
    assert result.exit_code == 2, result.output
    assert captured["json"]["commands"] == ["git push --force"]


def test_pretooluse_submits_a_code_mode_exec_snippet_whole(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    """Code Mode wraps any number of tools.exec_command()/tools.apply_patch() calls in one
    JS snippet rather than naming a single command; the whole snippet is submitted as "the
    command" so a forbidden phrase still matches wherever it appears, with nothing parsed
    out of it.
    """
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    snippet = 'text(await tools.exec_command({cmd:"git status --short"}));'
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "code_mode_exec",
        "tool_input": {"command": snippet},
    }
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    assert captured["json"]["commands"] == [snippet]


def test_pretooluse_ignores_unhandled_codex_tools(repo: Path) -> None:
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "wait",
        "tool_input": {"cell_id": "1"},
    }
    result = _invoke(payload)
    assert result.exit_code == 0, result.output


# --- Stop: transcript command collection ------------------------------------


def _response_item(payload: dict[str, Any]) -> str:
    return json.dumps({"type": "response_item", "payload": payload})


def test_stop_event_collects_classic_function_call_commands(
    monkeypatch: pytest.MonkeyPatch, repo: Path, tmp_path: Path
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text(
        "\n".join(
            [
                _response_item(
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]}
                ),
                _response_item(
                    {"type": "function_call", "name": "Bash", "arguments": json.dumps({"command": "git status"})}
                ),
                _response_item(
                    {
                        "type": "function_call",
                        "name": "shell",
                        "arguments": json.dumps({"command": ["ls", "-la"]}),
                    }
                ),
                _response_item({"type": "function_call", "name": "wait", "arguments": json.dumps({"cell_id": "1"})}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    payload = {"hook_event_name": "Stop", "cwd": str(repo), "transcript_path": str(transcript)}
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    assert captured["json"]["commands"] == ["git status", "ls -la"]


def test_stop_event_skips_a_function_call_whose_arguments_is_not_a_json_string(
    monkeypatch: pytest.MonkeyPatch, repo: Path, tmp_path: Path
) -> None:
    """`arguments` is documented as a JSON-encoded string; a malformed record carrying it
    pre-parsed (a dict, here) must be skipped, not crash json.loads with an uncaught TypeError.
    """

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text(
        "\n".join(
            [
                _response_item({"type": "function_call", "name": "Bash", "arguments": {"command": "pwd"}}),
                _response_item(
                    {"type": "function_call", "name": "Bash", "arguments": json.dumps({"command": "git status"})}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    payload = {"hook_event_name": "Stop", "cwd": str(repo), "transcript_path": str(transcript)}
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    assert captured["json"]["commands"] == ["git status"]


def test_stop_event_collects_code_mode_exec_snippets_from_the_transcript(
    monkeypatch: pytest.MonkeyPatch, repo: Path, tmp_path: Path
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    transcript = tmp_path / "rollout.jsonl"
    snippet = 'text(await tools.exec_command({cmd:"npm install"}));'
    transcript.write_text(
        _response_item({"type": "custom_tool_call", "name": "exec", "input": snippet}) + "\n",
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    payload = {"hook_event_name": "Stop", "cwd": str(repo), "transcript_path": str(transcript)}
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    assert captured["json"]["commands"] == [snippet]


def test_stop_event_skips_a_malformed_codex_transcript_line(
    monkeypatch: pytest.MonkeyPatch, repo: Path, tmp_path: Path
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text(
        "not json\n"
        + _response_item({"type": "function_call", "name": "Bash", "arguments": '{"command": "pwd"}'})
        + "\n",
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    payload = {"hook_event_name": "Stop", "cwd": str(repo), "transcript_path": str(transcript)}
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    assert captured["json"]["commands"] == ["pwd"]


def test_stop_event_codex_transcript_missing_submits_no_command_evidence(
    monkeypatch: pytest.MonkeyPatch, repo: Path, tmp_path: Path
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    payload = {
        "hook_event_name": "Stop",
        "cwd": str(repo),
        "transcript_path": str(tmp_path / "does-not-exist.jsonl"),
    }
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    assert captured["json"]["commands"] is None


# --- Judge gate transcript extraction ---------------------------------------


def test_codex_judge_transcript_extraction_keeps_only_assistant_output_text(tmp_path: Path) -> None:
    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text(
        "\n".join(
            [
                _response_item(
                    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "do it"}]}
                ),
                _response_item(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Reviewing now."}],
                    }
                ),
                _response_item({"type": "function_call", "name": "Bash", "arguments": "{}"}),
                _response_item(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Found an issue."}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert hook_cli._hook_extract_codex_judge_transcript(transcript) == "Reviewing now.\nFound an issue."


def test_codex_judge_transcript_extraction_returns_empty_for_an_unreadable_file(tmp_path: Path) -> None:
    assert hook_cli._hook_extract_codex_judge_transcript(tmp_path / "missing.jsonl") == ""


# --- otari hook setup --harness codex ---------------------------------------


@pytest.fixture(autouse=True)
def _fixed_otari_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hook_cli, "_otari_binary_path", lambda: _FAKE_OTARI_PATH)


def _invoke_setup(*args: str, input: str | None = None) -> Any:  # noqa: A002 - matches CliRunner's own kwarg name
    return CliRunner().invoke(hook_cli.hook, ["setup", "--harness", "codex", *args], input=input)


def test_setup_writes_codex_hooks_json_not_claude_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    result = _invoke_setup("--api-key", "k", input="n\n")
    assert result.exit_code == 0, result.output
    assert not (tmp_path / ".claude").exists()
    settings = json.loads((tmp_path / ".codex" / "hooks.json").read_text(encoding="utf-8"))
    entry = settings["hooks"]["PreToolUse"][0]
    assert entry["matcher"] == "apply_patch"
    assert entry["hooks"][0]["command"] == f"{_FAKE_OTARI_PATH} hook --harness codex --api-key k"
    assert settings["hooks"]["Stop"][0]["hooks"][0]["command"] == f"{_FAKE_OTARI_PATH} hook --harness codex --api-key k"


def test_setup_matcher_covers_bash_and_exec_when_a_command_gate_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".otari-guardrails.yml").write_text(
        'schema_version: "1.0"\npolicy:\n  id: x\ngates:\n'
        "  - id: g\n    type: command\n    runs: [pre_tool_use.command]\n    enforcement: required\n"
        '    forbidden: ["npm"]\n    message: m\n',
        encoding="utf-8",
    )
    result = _invoke_setup("--api-key", "k")
    assert result.exit_code == 0, result.output
    settings = json.loads((tmp_path / ".codex" / "hooks.json").read_text(encoding="utf-8"))
    assert settings["hooks"]["PreToolUse"][0]["matcher"] == "apply_patch|Bash|exec|code_mode_exec"
