"""Unit tests for `otari hook`, the native hook-protocol transport.

Mocks the network boundary (httpx.post) and the Git boundary (subprocess.run)
so these run with no server and no real repository; otari_agent.domain's
own evaluation is covered separately in tests/unit/agent_gates/ and
tests/integration/test_hooks_route.py.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

import otari_agent.hook as hook_cli
from otari_agent.settings import HookSettings

_GATES_YAML = "schema_version: '1.0'\npolicy:\n  id: test\ngates: []\n"


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._payload


@pytest.fixture(autouse=True)
def _judge_log_in_tmp_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirect the judge-call audit log away from the real ~/.otari/, for every test in this module."""
    monkeypatch.setattr(hook_cli, "_hook_judge_log_path", lambda: tmp_path / "judge-calls.log")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".otari-gates.yml").write_text(_GATES_YAML, encoding="utf-8")
    return tmp_path


@pytest.fixture
def config_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_settings(config_path: str | None = None) -> HookSettings:
        return HookSettings(master_key="test-master-key")

    monkeypatch.setattr(hook_cli, "load_settings", fake_load_settings)


def _invoke(payload: dict[str, Any], **extra_args: str) -> Any:
    args = ["--api-key", "test-key"]
    for key, value in extra_args.items():
        args += [f"--{key.replace('_', '-')}", value]
    return CliRunner().invoke(hook_cli.hook, args, input=json.dumps(payload))


def test_pretooluse_blocks_a_forbidden_edit(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _FakeResponse(
            {
                "blocked": True,
                "results": [{"gate_id": "g", "enforcement": "required", "outcome": "fail", "message": "no"}],
            }
        ),
    )
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "Edit",
        "tool_input": {"file_path": str(repo / "CHANGELOG.md")},
    }
    result = _invoke(payload)
    assert result.exit_code == 2, result.output
    assert "no" in result.output


def test_pretooluse_allows_when_not_blocked(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResponse({"blocked": False, "results": []}))
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "Edit",
        "tool_input": {"file_path": str(repo / "README.md")},
    }
    result = _invoke(payload)
    assert result.exit_code == 0, result.output


def test_pretooluse_ignores_unhandled_tools(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    called = False

    def fake_post(*args: object, **kwargs: object) -> _FakeResponse:
        nonlocal called
        called = True
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "Read",
        "tool_input": {"file_path": str(repo / "README.md")},
    }
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    assert not called, "a tool call this integration does not name must never reach the Hook Server"


def test_pretooluse_submits_a_bash_command_for_command_match(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
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
    assert captured["json"]["changed_paths"] == []


def test_an_oversize_bash_command_is_truncated_rather_than_rejected(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    """The Hook Server 422s a command over its limit, and a 422 fails the whole
    check open, taking every changed_path gate in the same policy with it. A
    Bash call carrying a heredoc clears that limit routinely, so the head is
    sent (where a tool name lives) instead of the request being lost.
    """
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    command = "npm install " + "x" * 8000
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "Bash",
        "tool_input": {"command": command},
    }
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    sent = captured["json"]["commands"]
    assert len(sent[0]) == hook_cli._HOOK_MAX_COMMAND_LENGTH
    assert sent[0].startswith("npm install ")
    assert "checking only the first" in result.output


def test_a_rejected_check_does_not_block_and_does_not_blame_the_network(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    """A 4xx means the request arrived and was answered. Reporting it as
    "could not reach" sends whoever debugs it to the network rather than to
    the status and body that say what was actually wrong.
    """

    class _RejectingResponse:
        status_code = 422
        text = "commands entry exceeds 4096 characters."

        def raise_for_status(self) -> None:
            request = httpx.Request("POST", "http://gw.test/api/v1/hooks/check")
            response = httpx.Response(422, text=self.text, request=request)
            raise httpx.HTTPStatusError("422", request=request, response=response)

        def json(self) -> dict[str, Any]:  # pragma: no cover - never reached
            raise AssertionError("json() must not be called on a rejected check")

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _RejectingResponse())
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "Bash",
        "tool_input": {"command": "npm install"},
    }
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    assert "rejected the check (422" in result.output
    assert "could not reach" not in result.output


def test_pretooluse_ignores_a_bash_call_with_no_command(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    called = False

    def fake_post(*args: object, **kwargs: object) -> _FakeResponse:
        nonlocal called
        called = True
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    payload = {"hook_event_name": "PreToolUse", "cwd": str(repo), "tool_name": "Bash", "tool_input": {}}
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    assert not called


def test_stop_event_blocks_on_git_status(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        # -z's real shape: NUL-delimited, no trailing newline per record.
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=" M CHANGELOG.md\0", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
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
    payload = {"hook_event_name": "Stop", "cwd": str(repo)}
    result = _invoke(payload)
    assert result.exit_code == 2, result.output
    assert captured["json"]["changed_paths"] == ["CHANGELOG.md"]


def test_stop_event_evaluates_locally_and_blocks_on_git_status(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    """The default, no-flag path's own full Stop-event pipeline: real Git

    evidence collection feeding the real `run_policy_check`, not a mocked
    `httpx.post` standing in for the evaluator. Every other Stop-event test
    in this module opts into the remote mode (`--api-key`) and mocks the
    network boundary instead; this is the only one that proves the default
    path's evidence collection and evaluation are wired together correctly
    end to end.
    """
    (repo / ".otari-gates.yml").write_text(
        'schema_version: "1.0"\npolicy:\n  id: test\ngates:\n'
        "  - id: g\n    type: changed_path\n    enforcement: required\n"
        '    forbidden: ["CHANGELOG.md"]\n    message: forbidden\n',
        encoding="utf-8",
    )

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=" M CHANGELOG.md\0", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: pytest.fail("httpx.post should not be called"))
    payload = {"hook_event_name": "Stop", "cwd": str(repo)}
    result = CliRunner().invoke(hook_cli.hook, [], input=json.dumps(payload))
    assert result.exit_code == 2, result.output
    assert "forbidden" in result.output


def _transcript_line(
    *, command: str | None = None, text: str | None = None, side_chain: bool = False, tool_use_id: str = "toolu_1"
) -> str:
    """One JSONL line shaped like a real Claude Code transcript record."""
    content: list[dict[str, Any]]
    if command is not None:
        content = [{"type": "tool_use", "id": tool_use_id, "name": "Bash", "input": {"command": command}}]
    else:
        content = [{"type": "text", "text": text or "hello"}]
    record = {
        "type": "assistant",
        "isSidechain": side_chain,
        "message": {"role": "assistant", "content": content},
    }
    return json.dumps(record)


def _tool_result_line(*, tool_use_id: str, is_error: bool, content: str) -> str:
    """One JSONL line shaped like a real Claude Code tool_result record.

    Confirmed against a real transcript: a PreToolUse denial's `content` is a
    bare string like "PreToolUse:Bash hook error: [...]: {...}", not a list
    of content blocks.
    """
    record = {
        "type": "user",
        "isSidechain": False,
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "is_error": is_error, "content": content}],
        },
    }
    return json.dumps(record)


def test_stop_event_submits_commands_collected_from_the_transcript(
    monkeypatch: pytest.MonkeyPatch, repo: Path, tmp_path: Path
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        "\n".join(
            [
                _transcript_line(text="thinking..."),
                _transcript_line(command="make postman"),
                _transcript_line(command="git status"),
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
    assert captured["json"]["commands"] == ["make postman", "git status"]


def test_stop_event_excludes_sidechain_commands(monkeypatch: pytest.MonkeyPatch, repo: Path, tmp_path: Path) -> None:
    """A subagent's own Bash calls (isSidechain: true) are not this policy's own agent's."""

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        "\n".join(
            [
                _transcript_line(command="make postman"),
                _transcript_line(command="rm -rf /", side_chain=True),
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
    assert captured["json"]["commands"] == ["make postman"]


def test_stop_event_excludes_a_command_a_pretooluse_hook_denied(
    monkeypatch: pytest.MonkeyPatch, repo: Path, tmp_path: Path
) -> None:
    """A denied `npm install` followed by an allowed `pnpm install` must not

    keep failing every later Stop for an attempt that never executed: the
    transcript records the denied call as a tool_use like any other, and the
    only trace of the denial is the paired tool_result naming the same
    tool_use_id, is_error, with content matching a PreToolUse hook block.
    """

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        "\n".join(
            [
                _transcript_line(command="npm install", tool_use_id="toolu_denied"),
                _tool_result_line(
                    tool_use_id="toolu_denied",
                    is_error=True,
                    content="PreToolUse:Bash hook error: [otari hook]: otari hook: blocked (claude-code, PreToolUse)",
                ),
                _transcript_line(command="pnpm install", tool_use_id="toolu_allowed"),
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
    assert captured["json"]["commands"] == ["pnpm install"]


def test_stop_event_includes_a_command_that_ran_but_exited_nonzero(
    monkeypatch: pytest.MonkeyPatch, repo: Path, tmp_path: Path
) -> None:
    """A command that actually ran, and merely failed, is not a PreToolUse

    denial: excluding every is_error tool_result regardless of content would
    let a forbidden command that happened to also fail evade command_match,
    the opposite of what excluding a denial is for.
    """

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        "\n".join(
            [
                _transcript_line(command="npm install", tool_use_id="toolu_ran"),
                _tool_result_line(tool_use_id="toolu_ran", is_error=True, content="npm error code ENOENT"),
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
    assert captured["json"]["commands"] == ["npm install"]


def test_stop_event_submits_no_commands_when_aggregate_evidence_is_oversize(
    monkeypatch: pytest.MonkeyPatch, repo: Path, tmp_path: Path
) -> None:
    """Per-command truncation alone doesn't bound the total: 501 commands,

    each safely under the per-command cap, already clear the Hook Server's
    own aggregate _MAX_TOTAL_COMMAND_CHARS. Submitting an arbitrary subset
    (dropping the oldest) risks a false pass or false fail on whichever
    command that subset happened to lose, so this submits no command
    evidence at all (None) rather than a partial one: a required
    command_match/command_if_changed gate then resolves unknown and blocks,
    instead of risking either outcome on data known to be incomplete.
    """

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    transcript = tmp_path / "session.jsonl"
    # Index at the front, comfortably under _HOOK_MAX_COMMAND_LENGTH per
    # command (4008 chars), so only the aggregate bound is exercised here.
    lines = [_transcript_line(command=f"cmd{i:04d} " + "x" * 4000, tool_use_id=f"toolu_{i}") for i in range(501)]
    transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    payload = {"hook_event_name": "Stop", "cwd": str(repo), "transcript_path": str(transcript)}
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    assert captured["json"]["commands"] is None
    assert "submitting no command evidence" in result.output


def test_stop_event_submits_no_commands_when_there_are_too_many(
    monkeypatch: pytest.MonkeyPatch, repo: Path, tmp_path: Path
) -> None:
    """Same reasoning as the aggregate-characters bound, for the count bound:

    an arbitrary subset of way too many commands is not evidence a required
    gate should trust either.
    """

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    transcript = tmp_path / "session.jsonl"
    lines = [
        _transcript_line(command=f"cmd{i}", tool_use_id=f"toolu_{i}") for i in range(hook_cli._HOOK_MAX_COMMANDS + 1)
    ]
    transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    payload = {"hook_event_name": "Stop", "cwd": str(repo), "transcript_path": str(transcript)}
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    assert captured["json"]["commands"] is None
    assert "submitting no command evidence" in result.output


def test_stop_event_submits_no_commands_when_transcript_path_is_missing(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    """No transcript_path at all submits None (unresolved), not `[]` (collected, none)."""

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    payload = {"hook_event_name": "Stop", "cwd": str(repo)}
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    assert captured["json"]["commands"] is None


def test_stop_event_submits_no_commands_when_transcript_is_unreadable(
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
    missing_transcript = tmp_path / "does-not-exist.jsonl"
    payload = {"hook_event_name": "Stop", "cwd": str(repo), "transcript_path": str(missing_transcript)}
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    assert captured["json"]["commands"] is None


def test_stop_event_skips_a_malformed_transcript_line(
    monkeypatch: pytest.MonkeyPatch, repo: Path, tmp_path: Path
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        "\n".join(["not json at all", _transcript_line(command="make postman")]) + "\n", encoding="utf-8"
    )
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    payload = {"hook_event_name": "Stop", "cwd": str(repo), "transcript_path": str(transcript)}
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    assert captured["json"]["commands"] == ["make postman"]


def test_stop_event_skips_a_bash_call_whose_input_is_not_a_mapping(
    monkeypatch: pytest.MonkeyPatch, repo: Path, tmp_path: Path
) -> None:
    """A tool_use block's `input` is a mapping in every real transcript, but this

    reads a caller-controlled file it does not otherwise validate; a line
    that deviates (input as a bare string, say) must be skipped like any
    other malformed line, not crash the whole Stop event with an
    AttributeError from treating a non-dict as one.
    """

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    transcript = tmp_path / "session.jsonl"
    malformed = json.dumps(
        {
            "type": "assistant",
            "isSidechain": False,
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": "not-a-mapping"}],
            },
        }
    )
    transcript.write_text("\n".join([malformed, _transcript_line(command="make postman")]) + "\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    payload = {"hook_event_name": "Stop", "cwd": str(repo), "transcript_path": str(transcript)}
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    assert captured["json"]["commands"] == ["make postman"]


def test_pretooluse_submits_a_posix_relative_path(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    """A nested target is submitted "/"-separated, whatever the platform.

    A forbidden glob is a repo-relative POSIX path and the evaluator splits it
    on "/", so a WindowsPath's native ``docs\\foo.md`` spelling would match
    nothing and every PreToolUse gate would pass on Windows. This asserts the
    separator directly rather than the equality alone, so the intent survives a
    reader on a POSIX box, where ``str()`` and ``as_posix()`` agree and only a
    Windows run can tell the two apart.
    """
    nested = repo / "docs" / "guide" / "page.md"
    nested.parent.mkdir(parents=True)
    nested.write_text("x", encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "Edit",
        "tool_input": {"file_path": str(nested)},
    }
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    submitted = captured["json"]["changed_paths"]
    assert submitted == ["docs/guide/page.md"]
    assert "\\" not in submitted[0]


def test_stop_event_parses_a_rename_as_its_new_path(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    """-z reports a rename/copy as two consecutive tokens: new path, then old path."""
    captured: dict[str, Any] = {}

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="R  renamed.txt\0original.txt\0", stderr="")

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(httpx, "post", fake_post)
    result = _invoke({"hook_event_name": "Stop", "cwd": str(repo)})
    assert result.exit_code == 0, result.output
    assert captured["json"]["changed_paths"] == ["renamed.txt"]


def test_stop_event_does_not_misparse_a_filename_containing_an_arrow(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    """An untracked file literally named 'a -> b.txt' is one token, not a false rename.

    The old human-readable-format parser split any entry containing the
    substring " -> " as though it were a rename's "old -> new", which would
    have truncated this filename to whatever followed the last " -> " in it.
    """
    captured: dict[str, Any] = {}

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="?? weird -> name.txt\0", stderr="")

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(httpx, "post", fake_post)
    result = _invoke({"hook_event_name": "Stop", "cwd": str(repo)})
    assert result.exit_code == 0, result.output
    assert captured["json"]["changed_paths"] == ["weird -> name.txt"]


def test_stop_event_reports_a_non_ascii_filename_unescaped(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    """-z never quotes/octal-escapes a path, unlike the human-readable format."""
    captured: dict[str, Any] = {}

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="?? café.txt\0", stderr="")

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(httpx, "post", fake_post)
    result = _invoke({"hook_event_name": "Stop", "cwd": str(repo)})
    assert result.exit_code == 0, result.output
    assert captured["json"]["changed_paths"] == ["café.txt"]


def test_stop_event_does_not_block_when_git_status_fails(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="not a git repository")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = _invoke({"hook_event_name": "Stop", "cwd": str(repo)})
    assert result.exit_code == 0, result.output
    assert "could not read Git state" in result.output


def test_unrecognized_event_is_a_no_op(repo: Path) -> None:
    result = _invoke({"hook_event_name": "PostToolUse", "cwd": str(repo)})
    assert result.exit_code == 0, result.output


def test_no_policy_file_is_a_no_op(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    result = _invoke({"hook_event_name": "Stop", "cwd": str(tmp_path)})
    assert result.exit_code == 0, result.output


def test_a_non_utf8_policy_file_does_not_block(tmp_path: Path) -> None:
    """`gates_file.is_file()` does not guarantee the read right after it succeeds

    (a race, a permissions change, a non-UTF-8 file): an uncaught
    `UnicodeDecodeError` there used to exit `otari hook` nonzero before it
    ever reached `httpx.post`, breaking the fail-open contract every other
    evidence-collection failure in this command already has.
    """
    (tmp_path / ".git").mkdir()
    (tmp_path / ".otari-gates.yml").write_bytes(b'schema_version: "1.0"\npolicy:\n  id: x\n# caf\xe9\ngates: []\n')
    result = _invoke({"hook_event_name": "Stop", "cwd": str(tmp_path)})
    assert result.exit_code == 0, result.output
    assert "could not read" in result.output


def test_outside_a_git_repo_is_a_no_op(tmp_path: Path) -> None:
    result = _invoke({"hook_event_name": "Stop", "cwd": str(tmp_path)})
    assert result.exit_code == 0, result.output


def test_malformed_stdin_is_a_no_op() -> None:
    result = CliRunner().invoke(hook_cli.hook, ["--api-key", "test-key"], input="not json")
    assert result.exit_code == 0, result.output


def test_no_flags_evaluates_locally_with_no_credential_needed(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    """No `--api-key`/`--url` is the default now, not a missing-setup case:

    `otari hook` evaluates `.otari-gates.yml` in process
    (`otari_agent.domain.check.run_policy_check`) and calls `httpx.post`
    only when either flag opts into the other, HTTP-backed mode. A required
    gate still blocks with no credential, no config, and no server at all.
    """

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("httpx.post should not be called for the default, local evaluation path")

    monkeypatch.setattr(httpx, "post", fail_if_called)
    (repo / ".otari-gates.yml").write_text(
        'schema_version: "1.0"\npolicy:\n  id: test\ngates:\n'
        "  - id: g\n    type: changed_path\n    enforcement: required\n"
        '    forbidden: ["CHANGELOG.md"]\n    message: forbidden\n',
        encoding="utf-8",
    )
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "Edit",
        "tool_input": {"file_path": str(repo / "CHANGELOG.md")},
    }
    result = CliRunner().invoke(hook_cli.hook, [], input=json.dumps(payload))
    assert result.exit_code == 2, result.output
    assert "forbidden" in result.output


def test_malformed_local_policy_does_not_block(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    """The local evaluation path's own fail-open contract: a policy

    `run_policy_check` cannot parse must report and exit 0, the same as
    every other evidence-collection failure this command handles, not raise.
    """
    monkeypatch.setattr(httpx, "post", lambda *a, **k: pytest.fail("httpx.post should not be called"))
    (repo / ".otari-gates.yml").write_text("not: valid: yaml: at: all:\n  - [", encoding="utf-8")
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "Edit",
        "tool_input": {"file_path": str(repo / "CHANGELOG.md")},
    }
    result = CliRunner().invoke(hook_cli.hook, [], input=json.dumps(payload))
    assert result.exit_code == 0, result.output
    assert "could not evaluate" in result.output


def test_url_alone_without_a_resolvable_credential_does_not_block(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    """The opt-in remote mode still needs a credential from somewhere:

    `--url` alone opts in, but with no `--api-key` and no configured
    `master_key`, there is nothing to authenticate the request with, and
    that must fail open rather than block.
    """

    def fake_load_settings(config_path: str | None = None) -> HookSettings:
        return HookSettings(master_key=None)

    monkeypatch.setattr(hook_cli, "load_settings", fake_load_settings)
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "Edit",
        "tool_input": {"file_path": str(repo / "CHANGELOG.md")},
    }
    result = CliRunner().invoke(hook_cli.hook, ["--url", "http://gw.example:9000"], input=json.dumps(payload))
    assert result.exit_code == 0, result.output
    assert "no API key or master key resolved" in result.output


def test_invalid_config_does_not_block(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    """load_settings raises ValueError on an unreadable or malformed config file.

    That is a setup problem, not a required gate failing, so it must fail
    open like every other setup failure this command handles, not surface as
    an unhandled traceback.
    """

    def fake_load_settings(config_path: str | None = None) -> HookSettings:
        raise ValueError("Hybrid mode (legacy value 'platform') requires OTARI_AI_TOKEN to be set.")

    monkeypatch.setattr(hook_cli, "load_settings", fake_load_settings)
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "Edit",
        "tool_input": {"file_path": str(repo / "CHANGELOG.md")},
    }
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    assert "could not load config" in result.output


@pytest.mark.parametrize(
    ("body", "case"),
    [
        (ValueError("Expecting value: line 1 column 1 (char 0)"), "not JSON at all"),
        ({"results": [{"gate_id": "g", "outcome": "fail"}]}, "JSON with no 'blocked'"),
        ({"blocked": True}, "JSON with no 'results'"),
        ({"results": "not-a-list", "blocked": True}, "'results' of the wrong type"),
    ],
)
def test_unreadable_response_does_not_block(
    monkeypatch: pytest.MonkeyPatch, repo: Path, body: object, case: str
) -> None:
    """A response this command cannot read fails open like an unreachable one.

    The command's contract is that it blocks on a required gate failing and on
    nothing else. A body that is not JSON, or JSON of an unexpected shape, used
    to escape the ``httpx.HTTPError`` handler as a bare ValueError/KeyError and
    surface as a traceback.
    """

    class _Unreadable:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> Any:
            if isinstance(body, Exception):
                raise body
            return body

    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: _Unreadable())
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "Edit",
        "tool_input": {"file_path": str(repo / "CHANGELOG.md")},
    }
    result = _invoke(payload)
    assert result.exit_code == 0, f"{case}: {result.output}"
    assert "unreadable response" in result.output, case


def test_a_gate_result_missing_display_fields_does_not_crash(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    """The try/except around reading the response only covers what builds

    `failing` (result["results"], gate["outcome"]); it does not, on its own,
    cover the later step that formats each failing gate for display, which
    reads gate['enforcement']/['gate_id']/['message']. A gate result that is
    well-formed enough to build `failing` (has 'outcome') but is missing one
    of those other fields, as an older or otherwise mismatched otari serve
    behind --url might send, must not raise KeyError there and surface as a
    traceback instead of this command's own fail-open contract.
    """
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _FakeResponse({"blocked": True, "results": [{"outcome": "fail"}]}),
    )
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "Edit",
        "tool_input": {"file_path": str(repo / "CHANGELOG.md")},
    }
    result = _invoke(payload)
    assert result.exit_code == 2, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_unreachable_gateway_does_not_block(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    def fake_post(*args: object, **kwargs: object) -> _FakeResponse:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", fake_post)
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "Edit",
        "tool_input": {"file_path": str(repo / "CHANGELOG.md")},
    }
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    assert "could not reach" in result.output


def test_api_key_alone_opts_into_remote_and_falls_back_to_configured_localhost(
    monkeypatch: pytest.MonkeyPatch, repo: Path, config_stub: None
) -> None:
    """`--api-key` with no `--url` is enough to opt into the HTTP-backed mode:

    the credential is the given one, but the gateway's own URL still falls
    back to the configured host/port, exactly as it did before local
    evaluation existed.
    """
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["url"] = url
        captured["headers"] = kwargs.get("headers")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "Edit",
        "tool_input": {"file_path": str(repo / "CHANGELOG.md")},
    }
    result = CliRunner().invoke(hook_cli.hook, ["--api-key", "given-key"], input=json.dumps(payload))
    assert result.exit_code == 0, result.output
    assert captured["url"] == "http://localhost:8000/api/v1/hooks/check"
    assert captured["headers"]["Otari-Key"] == "given-key"


def test_url_alone_opts_into_remote_and_falls_back_to_configured_master_key(
    monkeypatch: pytest.MonkeyPatch, repo: Path, config_stub: None
) -> None:
    """`--url` with no `--api-key` is likewise enough to opt in: the URL is

    the given one, but the credential still falls back to the configured
    ``master_key``.
    """
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["url"] = url
        captured["headers"] = kwargs.get("headers")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "Edit",
        "tool_input": {"file_path": str(repo / "CHANGELOG.md")},
    }
    result = CliRunner().invoke(hook_cli.hook, ["--url", "http://gw.example:9000"], input=json.dumps(payload))
    assert result.exit_code == 0, result.output
    assert captured["url"] == "http://gw.example:9000/api/v1/hooks/check"
    # The bare token, not a ``Bearer `` prefix: deps.extract_credential_token
    # tolerates the prefix for back-compat, but a header named for the key
    # carries the raw token.
    assert captured["headers"]["Otari-Key"] == "test-master-key"


def test_strip_judge_code_fence_recovers_json_wrapped_in_a_json_fence() -> None:
    fenced = '```json\n{"outcome": "pass", "reasoning": "fine"}\n```'
    assert hook_cli._hook_strip_judge_code_fence(fenced) == '{"outcome": "pass", "reasoning": "fine"}'


def test_strip_judge_code_fence_recovers_json_wrapped_in_a_bare_fence() -> None:
    fenced = '```\n{"outcome": "pass", "reasoning": "fine"}\n```'
    assert hook_cli._hook_strip_judge_code_fence(fenced) == '{"outcome": "pass", "reasoning": "fine"}'


def test_strip_judge_code_fence_leaves_unfenced_json_unchanged() -> None:
    unfenced = '{"outcome": "pass", "reasoning": "fine"}'
    assert hook_cli._hook_strip_judge_code_fence(unfenced) == unfenced


_JUDGE_GATES_YAML = (
    "schema_version: '1.0'\n"
    "policy:\n  id: test\n"
    "gates:\n"
    "  - id: follows-pattern\n"
    "    type: judge\n"
    "    enforcement: advisory\n"
    "    rubric: Does this change follow the repository's error-handling conventions?\n"
    "    message: Does not follow the pattern.\n"
)


@pytest.fixture
def judge_repo(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".otari-gates.yml").write_text(_JUDGE_GATES_YAML, encoding="utf-8")
    return tmp_path


def _git_status_and_diff_run(git_status_stdout: str = "", git_diff_stdout: str = "") -> Any:
    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=git_status_stdout, stderr="")
        if cmd[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=git_diff_stdout, stderr="")
        raise AssertionError(f"unexpected subprocess.run call before claude -p: {cmd}")

    return fake_run


def test_stop_event_submits_a_judge_verdict_from_claude_p(
    monkeypatch: pytest.MonkeyPatch, judge_repo: Path, tmp_path: Path
) -> None:
    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="+ changed line\n", stderr="")
        if cmd[0] == "/usr/bin/claude":
            assert cmd[1:3] == ["--model", hook_cli._HOOK_JUDGE_DEFAULT_MODEL], (
                "a judge call defaults to the cheaper model, not the session's own"
            )
            assert cmd[3:6] == ["--tools", "", "--strict-mcp-config"], (
                "a judge call never needs a tool or an MCP server, and its prompt embeds "
                "attacker-influenceable diff/transcript text that must not reach either"
            )
            assert cmd[-1] == "-p" and "input" in kwargs, (
                "the prompt goes over stdin, not as a trailing argv element: an embedded "
                "NUL byte (which a diff or transcript can carry) raises ValueError as an argv "
                "element but not as stdin input"
            )
            assert kwargs.get("cwd") == hook_cli._hook_judge_workdir(), (
                "must run outside the repo it is judging, or its own Stop hook "
                "(this same otari hook command) recurses into itself"
            )
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=json.dumps({"outcome": "fail", "reasoning": "does not match"}),
                stderr="",
            )
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)

    transcript = tmp_path / "session.jsonl"
    transcript.write_text(_transcript_line(text="did some work") + "\n", encoding="utf-8")

    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse(
            {
                "blocked": False,
                "results": [
                    {
                        "gate_id": "follows-pattern",
                        "enforcement": "advisory",
                        "outcome": "fail",
                        "message": "Does not follow the pattern.",
                    }
                ],
            }
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    payload = {"hook_event_name": "Stop", "cwd": str(judge_repo), "transcript_path": str(transcript)}
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    assert captured["json"]["judge_results"] == [
        {"gate_id": "follows-pattern", "outcome": "fail", "reasoning": "does not match"}
    ]


def test_stop_event_locally_evaluates_a_judge_verdict_and_warns(
    monkeypatch: pytest.MonkeyPatch, judge_repo: Path, tmp_path: Path
) -> None:
    """The default path's own judge-gate flow, no `httpx.post` mock: the

    locally-collected verdict must reach `run_policy_check` and come back as
    an advisory, non-blocking `systemMessage`, not just get built correctly
    for a mocked network call (`test_stop_event_submits_a_judge_verdict_from_claude_p`
    covers that half already).
    """

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="+ changed line\n", stderr="")
        if cmd[0] == "/usr/bin/claude":
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=json.dumps({"outcome": "fail", "reasoning": "does not match"}), stderr=""
            )
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: pytest.fail("httpx.post should not be called"))

    transcript = tmp_path / "session.jsonl"
    transcript.write_text(_transcript_line(text="did some work") + "\n", encoding="utf-8")

    payload = {"hook_event_name": "Stop", "cwd": str(judge_repo), "transcript_path": str(transcript)}
    result = CliRunner().invoke(hook_cli.hook, [], input=json.dumps(payload))
    assert result.exit_code == 0, result.output
    stdout_payload = json.loads(result.stdout)
    assert "does not match" in stdout_payload["systemMessage"]


def test_judge_model_is_overridable_via_flag(monkeypatch: pytest.MonkeyPatch, judge_repo: Path) -> None:
    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[0] == "/usr/bin/claude":
            assert cmd[1:3] == ["--model", "claude-sonnet-5"]
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=json.dumps({"outcome": "pass", "reasoning": "ok"}), stderr=""
            )
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResponse({"blocked": False, "results": []}))

    result = _invoke({"hook_event_name": "Stop", "cwd": str(judge_repo)}, judge_model="claude-sonnet-5")
    assert result.exit_code == 0, result.output


def test_judge_dry_run_never_calls_claude_but_still_counts_and_logs(
    monkeypatch: pytest.MonkeyPatch, judge_repo: Path
) -> None:
    """--judge-dry-run runs the whole applicability/diff/transcript pipeline for real,

    but skips the actual `claude -p` call: `shutil.which("claude")` and
    `subprocess.run` for `claude` must never be reached, yet the submitted
    verdict still carries a reasoning that estimates the prompt size, and the
    audit log still gets both the "invoking" and the completed line (so
    counting log lines tells you how many real calls this would have made).
    """

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="+ changed line\n", stderr="")
        raise AssertionError(f"claude must never be invoked in a dry run: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    def fake_which(name: str) -> str | None:
        raise AssertionError("shutil.which('claude') must never be called in a dry run")

    monkeypatch.setattr(shutil, "which", fake_which)

    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    args = ["--api-key", "test-key", "--judge-dry-run"]
    result = CliRunner().invoke(
        hook_cli.hook, args, input=json.dumps({"hook_event_name": "Stop", "cwd": str(judge_repo)})
    )
    assert result.exit_code == 0, result.output

    [judge_result] = captured["json"]["judge_results"]
    assert judge_result["gate_id"] == "follows-pattern"
    assert judge_result["outcome"] == "error"
    assert "--judge-dry-run" in judge_result["reasoning"]
    assert "tokens estimated" in judge_result["reasoning"]

    log_lines = hook_cli._hook_judge_log_path().read_text(encoding="utf-8").splitlines()
    assert sum(1 for line in log_lines if "outcome=invoking" in line) == 1
    assert sum(1 for line in log_lines if "detail=" in line) == 1


def test_stop_event_parses_a_verdict_wrapped_in_a_markdown_code_fence(
    monkeypatch: pytest.MonkeyPatch, judge_repo: Path
) -> None:
    """A real `claude -p --model claude-haiku-4-5-20251001` call, prompted with this exact

    "no markdown fence" instruction, still wrapped its JSON verdict in a
    ```json fence (confirmed manually against the real CLI). Parsing this
    as a failure would report "error" on what was, in substance, a
    perfectly good verdict, so `_hook_strip_judge_code_fence` must recover it.
    """

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[0] == "/usr/bin/claude":
            fenced = (
                '```json\n{"outcome": "fail", "reasoning": "The comment narrates the change '
                'itself rather than explaining non-obvious logic."}\n```\n'
            )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=fenced, stderr="")
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)

    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    result = _invoke({"hook_event_name": "Stop", "cwd": str(judge_repo)})
    assert result.exit_code == 0, result.output
    assert captured["json"]["judge_results"] == [
        {
            "gate_id": "follows-pattern",
            "outcome": "fail",
            "reasoning": "The comment narrates the change itself rather than explaining non-obvious logic.",
        }
    ]


def test_stop_event_reports_error_when_claude_is_not_on_path(monkeypatch: pytest.MonkeyPatch, judge_repo: Path) -> None:
    monkeypatch.setattr(subprocess, "run", _git_status_and_diff_run())
    monkeypatch.setattr(shutil, "which", lambda name: None)

    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    result = _invoke({"hook_event_name": "Stop", "cwd": str(judge_repo)})
    assert result.exit_code == 0, result.output
    assert captured["json"]["judge_results"] == [
        {
            "gate_id": "follows-pattern",
            "outcome": "error",
            "reasoning": "none of the configured judge CLI(s) were found on PATH: claude",
        }
    ]


def test_stop_event_reports_error_when_claude_p_output_is_not_valid_json(
    monkeypatch: pytest.MonkeyPatch, judge_repo: Path
) -> None:
    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[0] == "/usr/bin/claude":
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="Sure, I'll check that.", stderr="")
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)

    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    result = _invoke({"hook_event_name": "Stop", "cwd": str(judge_repo)})
    assert result.exit_code == 0, result.output
    [judge_result] = captured["json"]["judge_results"]
    assert judge_result["gate_id"] == "follows-pattern"
    assert judge_result["outcome"] == "error"
    assert "not return valid JSON" in judge_result["reasoning"]


def test_stop_event_warns_when_the_diff_is_truncated(monkeypatch: pytest.MonkeyPatch, judge_repo: Path) -> None:
    oversize_diff = "x" * (hook_cli._HOOK_JUDGE_MAX_DIFF_CHARS + 1)

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=oversize_diff, stderr="")
        if cmd[0] == "/usr/bin/claude":
            prompt = str(kwargs.get("input"))
            assert "... (diff truncated)" in prompt
            assert oversize_diff not in prompt, "the prompt must not carry the whole oversize diff"
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=json.dumps({"outcome": "pass", "reasoning": "ok"}), stderr=""
            )
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResponse({"blocked": False, "results": []}))

    result = _invoke({"hook_event_name": "Stop", "cwd": str(judge_repo)})
    assert result.exit_code == 0, result.output
    assert "diff is" in result.output
    assert "over the" in result.output


def test_stop_event_warns_when_the_transcript_is_truncated(
    monkeypatch: pytest.MonkeyPatch, judge_repo: Path, tmp_path: Path
) -> None:
    transcript = tmp_path / "session.jsonl"
    big_text = "x" * (hook_cli._HOOK_JUDGE_MAX_TRANSCRIPT_CHARS + 1)
    transcript.write_text(_transcript_line(text=big_text) + "\n", encoding="utf-8")

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[0] == "/usr/bin/claude":
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=json.dumps({"outcome": "pass", "reasoning": "ok"}), stderr=""
            )
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResponse({"blocked": False, "results": []}))

    result = _invoke({"hook_event_name": "Stop", "cwd": str(judge_repo), "transcript_path": str(transcript)})
    assert result.exit_code == 0, result.output
    assert "transcript is" in result.output
    assert "over the" in result.output


def test_judge_transcript_extraction_keeps_only_assistant_text(tmp_path: Path) -> None:
    """A judge gate's transcript evidence is the assistant's own `text` replies,

    never a `tool_use`/`tool_result` payload (a Bash call's own stdout, a
    Read's file contents, ...): bytes that dominate a raw transcript's size
    but carry no "why was this change made" signal, and are why the flat
    char cap needed a real ratio measurement (see
    `_HOOK_JUDGE_MAX_TRANSCRIPT_CHARS`'s own comment). A sidechain (subagent)
    turn is excluded too, same as command evidence.
    """
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        "\n".join(
            [
                _transcript_line(text="Adding a helper for p95 latency."),
                _transcript_line(command="cat very-large-file.txt"),
                _tool_result_line(tool_use_id="toolu_1", is_error=False, content="x" * 10_000),
                _transcript_line(text="Done.", side_chain=True),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    extracted = hook_cli._hook_extract_judge_transcript(transcript)
    assert extracted == "Adding a helper for p95 latency."


def test_judge_transcript_extraction_returns_empty_for_an_unreadable_file(tmp_path: Path) -> None:
    assert hook_cli._hook_extract_judge_transcript(tmp_path / "missing.jsonl") == ""


def test_judge_workdir_is_not_the_repo_being_judged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A judge call's own `cwd` is never the caller's repo, or a real early version of

    this recurses into itself: that repo's own `.claude/settings.local.json`
    registers `otari hook` for `Stop`, so an unguarded call whose own `Stop`
    hook is this same command triggers it again.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    workdir = hook_cli._hook_judge_workdir()
    assert workdir == tmp_path / ".otari" / "judge-workdir"
    assert workdir.is_dir()
    assert workdir != tmp_path


def test_stop_event_bounds_judge_reasoning_and_a_required_gate_still_blocks(
    monkeypatch: pytest.MonkeyPatch, judge_repo: Path
) -> None:
    """An oversize `reasoning` is truncated before it ever reaches `judge_results`, or

    it would 422 the whole `/hooks/check` request server-side
    (`JudgeVerdictRequest.reasoning`, capped at the same
    `_HOOK_MAX_JUDGE_REASONING_LENGTH`), and this command's own fail-open
    handling for a rejected request would then silently skip every other
    gate in the same policy along with it, mechanical and required ones
    included. Modeled here by a mocked response that still reports `blocked`
    (standing in for a required gate the real server would have evaluated
    independently): a bug that dropped the whole request on the floor would
    never reach that response at all.
    """
    oversize_reasoning = "x" * (hook_cli._HOOK_MAX_JUDGE_REASONING_LENGTH + 1)

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[0] == "/usr/bin/claude":
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=json.dumps({"outcome": "fail", "reasoning": oversize_reasoning}),
                stderr="",
            )
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)

    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse(
            {
                "blocked": True,
                "results": [
                    {
                        "gate_id": "some-other-required-gate",
                        "enforcement": "required",
                        "outcome": "fail",
                        "message": "m",
                    }
                ],
            }
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = _invoke({"hook_event_name": "Stop", "cwd": str(judge_repo)})
    assert result.exit_code == 2, result.output

    [judge_result] = captured["json"]["judge_results"]
    assert len(judge_result["reasoning"]) == hook_cli._HOOK_MAX_JUDGE_REASONING_LENGTH


def test_stop_event_survives_a_judge_setup_failure_and_still_blocks(
    monkeypatch: pytest.MonkeyPatch, judge_repo: Path
) -> None:
    """`_hook_judge_workdir()`'s own `mkdir` (permissions, disk full) or the

    subprocess launch itself can raise `OSError`, and an embedded NUL byte
    in the diff or transcript raises `ValueError`; both used to propagate
    uncaught, exiting `otari hook` before it ever reached `httpx.post` and
    taking every other gate in the same policy, mechanical and required
    ones included, down with it. Modeled the same way as the oversize-
    reasoning test above: a mocked response that still reports `blocked`
    proves the request was actually submitted, which a crash before this
    point would never let happen.
    """

    def fake_workdir() -> Path:
        raise OSError("Permission denied: ~/.otari/judge-workdir")

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)
    monkeypatch.setattr(hook_cli, "_hook_judge_workdir", fake_workdir)

    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse(
            {
                "blocked": True,
                "results": [
                    {
                        "gate_id": "some-other-required-gate",
                        "enforcement": "required",
                        "outcome": "fail",
                        "message": "m",
                    }
                ],
            }
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = _invoke({"hook_event_name": "Stop", "cwd": str(judge_repo)})
    assert result.exit_code == 2, result.output

    [judge_result] = captured["json"]["judge_results"]
    assert judge_result["outcome"] == "error"
    assert "permission denied" in judge_result["reasoning"].lower()


def test_stop_event_never_calls_the_model_when_diff_collection_fails(
    monkeypatch: pytest.MonkeyPatch, judge_repo: Path
) -> None:
    """`_hook_collect_diff` returning `None` (collection genuinely failed, not

    "collected, and there is none") must never reach `claude -p` at all: a
    diff-less prompt is indistinguishable from a real empty diff, and a
    model asked to judge a change it cannot see can still say "pass"
    (verified against a real call). The gate resolves "error" instead,
    without spending a model call on a judgment that cannot mean anything.
    """
    claude_call_count = 0

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal claude_call_count
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="fatal: bad revision")
        if cmd[0] == "/usr/bin/claude":
            claude_call_count += 1
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=json.dumps({"outcome": "pass", "reasoning": "looks fine"}), stderr=""
            )
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)

    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse(
            {
                "blocked": True,
                "results": [
                    {
                        "gate_id": "some-other-required-gate",
                        "enforcement": "required",
                        "outcome": "fail",
                        "message": "m",
                    }
                ],
            }
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = _invoke({"hook_event_name": "Stop", "cwd": str(judge_repo)})
    assert result.exit_code == 2, result.output

    assert claude_call_count == 0, "a diff-collection failure must never reach claude -p"
    [judge_result] = captured["json"]["judge_results"]
    assert judge_result["outcome"] == "error"
    assert "diff" in judge_result["reasoning"].lower()


def test_stop_event_retries_the_judge_diff_only_when_the_prompt_is_too_long(
    monkeypatch: pytest.MonkeyPatch, judge_repo: Path, tmp_path: Path
) -> None:
    """`claude -p`'s own "prompt is too long" rejection (confirmed against a real

    call: nonzero exit, message on stdout, zero usage billed) retries once
    with the transcript dropped, since the diff is the primary evidence a
    judge rubric needs and the transcript is only supplementary. The retry's
    prompt must not carry the transcript at all.
    """
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(_transcript_line(text="a decision the diff alone would not explain") + "\n")

    claude_calls: list[str] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="+ changed line\n", stderr="")
        if cmd[0] == "/usr/bin/claude":
            prompt = str(kwargs.get("input"))
            claude_calls.append(prompt)
            if len(claude_calls) == 1:
                assert "a decision the diff alone would not explain" in prompt
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=1,
                    stdout="Prompt is too long · the request is ~290782 tokens (limit 200000)",
                    stderr="",
                )
            assert "a decision the diff alone would not explain" not in prompt, (
                "the retry must drop the transcript, not resend it"
            )
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=json.dumps({"outcome": "pass", "reasoning": "diff-only ok"}), stderr=""
            )
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)

    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    result = _invoke({"hook_event_name": "Stop", "cwd": str(judge_repo), "transcript_path": str(transcript)})
    assert result.exit_code == 0, result.output
    assert len(claude_calls) == 2
    assert captured["json"]["judge_results"] == [
        {"gate_id": "follows-pattern", "outcome": "pass", "reasoning": "diff-only ok"}
    ]


def test_stop_event_does_not_retry_when_there_is_no_transcript_to_drop(
    monkeypatch: pytest.MonkeyPatch, judge_repo: Path
) -> None:
    """Nothing to drop, so a "prompt is too long" rejection with an empty transcript

    reports "error" on the first call rather than repeating the exact same
    call a second time.
    """
    claude_call_count = 0

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal claude_call_count
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="+ changed line\n", stderr="")
        if cmd[0] == "/usr/bin/claude":
            claude_call_count += 1
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="Prompt is too long · the request is ~290782 tokens", stderr=""
            )
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)

    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    result = _invoke({"hook_event_name": "Stop", "cwd": str(judge_repo)})
    assert result.exit_code == 0, result.output
    assert claude_call_count == 1
    [judge_result] = captured["json"]["judge_results"]
    assert judge_result["outcome"] == "error"
    assert "prompt is too long" in judge_result["reasoning"].lower()


def test_stop_event_bounds_total_judge_time_so_a_required_gate_still_reaches_the_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A per-call timeout does not bound the total: `_HOOK_JUDGE_TOTAL_BUDGET_SECONDS`

    caps every judge gate and retry in one run to one shared deadline, so a
    few slow gates cannot themselves consume Claude Code's own outer hook
    timeout (~600s, past which it kills `otari hook` and discards its output
    entirely) and take a required mechanical gate down with them by keeping
    the request from ever reaching `/hooks/check`. Modeled with a fake clock
    rather than a real sleep: the first gate's own check finds time left and
    runs for real; by the second gate's check the deadline has already
    passed, so it (and every gate after it) reports "error" without ever
    calling `claude -p`.
    """
    (tmp_path / ".git").mkdir()
    gates_yaml = (
        "schema_version: '1.0'\npolicy:\n  id: test\ngates:\n"
        "  - id: no-hand-edited-changelog\n    type: changed_path\n    enforcement: required\n"
        '    forbidden: ["CHANGELOG.md"]\n    message: do not hand-edit\n'
        "  - id: judge-0\n    type: judge\n    enforcement: advisory\n    rubric: r0\n    message: m0\n"
        "  - id: judge-1\n    type: judge\n    enforcement: advisory\n    rubric: r1\n    message: m1\n"
        "  - id: judge-2\n    type: judge\n    enforcement: advisory\n    rubric: r2\n    message: m2\n"
    )
    (tmp_path / ".otari-gates.yml").write_text(gates_yaml, encoding="utf-8")

    claude_call_count = 0

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal claude_call_count
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=" M CHANGELOG.md\0", stderr="")
        if cmd[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[0] == "/usr/bin/claude":
            claude_call_count += 1
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=json.dumps({"outcome": "pass", "reasoning": "ok"}), stderr=""
            )
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)

    # One call to compute the shared deadline, then one per gate's own remaining-time
    # check: [deadline base, gate-0 check (budget left), gate-1 check (past deadline),
    # gate-2 check (still past deadline)].
    fake_clock = iter([0.0, 100.0, 600.0, 700.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(fake_clock))

    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse(
            {
                "blocked": True,
                "results": [
                    {
                        "gate_id": "no-hand-edited-changelog",
                        "enforcement": "required",
                        "outcome": "fail",
                        "message": "m",
                    }
                ],
            }
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = _invoke({"hook_event_name": "Stop", "cwd": str(tmp_path)})
    assert result.exit_code == 2, result.output

    assert claude_call_count == 1, "only the gate whose check ran before the deadline should call claude -p"
    outcomes = {entry["gate_id"]: entry["outcome"] for entry in captured["json"]["judge_results"]}
    assert outcomes["judge-0"] == "pass"
    assert outcomes["judge-1"] == "error"
    assert outcomes["judge-2"] == "error"
    assert "budget" in next(
        entry["reasoning"] for entry in captured["json"]["judge_results"] if entry["gate_id"] == "judge-1"
    )
    assert captured["json"]["changed_paths"] == ["CHANGELOG.md"]


def test_stop_event_with_a_non_utf8_diff_still_blocks_a_required_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`git diff HEAD` emits a tracked file's own content bytes, not necessarily

    valid UTF-8 (a Latin-1-encoded tracked file, confirmed against a real
    repo). Real `git`, not a mocked `subprocess.run`, is the point: this
    reproduces the actual `UnicodeDecodeError` `subprocess.run(...,
    encoding="utf-8")` raises from inside itself on such a file, which used
    to crash `otari hook` before it ever reached `httpx.post`, taking the
    unrelated required `changed_path` gate down with it. `--judge-dry-run`
    keeps this test from needing a real (or mocked) `claude` call: it still
    runs the real diff collection this bug lives in, only skipping the
    model call itself.
    """
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "src" / "gateway").mkdir(parents=True)
    (tmp_path / "src" / "gateway" / "latin.py").write_bytes(b"hello\n")
    (tmp_path / "CHANGELOG.md").write_text("v1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    with (tmp_path / "src" / "gateway" / "latin.py").open("ab") as f:
        f.write(b"\n# caf\xe9 latin1 comment\n")
    (tmp_path / "CHANGELOG.md").write_text("v1\nv2\n", encoding="utf-8")

    gates_yaml = (
        "schema_version: '1.0'\npolicy:\n  id: test\ngates:\n"
        "  - id: no-hand-edited-changelog\n    type: changed_path\n    enforcement: required\n"
        '    forbidden: ["CHANGELOG.md"]\n    message: do not hand-edit\n'
        "  - id: follows-pattern\n    type: judge\n    enforcement: advisory\n"
        '    rubric: r\n    when_changed: ["src/**"]\n    message: m\n'
    )
    (tmp_path / ".otari-gates.yml").write_text(gates_yaml, encoding="utf-8")

    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse(
            {
                "blocked": True,
                "results": [
                    {
                        "gate_id": "no-hand-edited-changelog",
                        "enforcement": "required",
                        "outcome": "fail",
                        "message": "m",
                    }
                ],
            }
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = CliRunner().invoke(
        hook_cli.hook,
        ["--api-key", "test-key", "--judge-dry-run"],
        input=json.dumps({"hook_event_name": "Stop", "cwd": str(tmp_path)}),
    )
    assert result.exit_code == 2, result.output
    # .otari-gates.yml itself is untracked here (written after the initial commit,
    # for a self-contained test repo) and so is real, expected changed-path evidence
    # too, alongside the two files this test cares about.
    assert sorted(captured["json"]["changed_paths"]) == [".otari-gates.yml", "CHANGELOG.md", "src/gateway/latin.py"]


def test_a_policy_with_no_judge_gates_submits_no_judge_results(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    """`repo`'s policy (`_GATES_YAML`) declares no gates at all, so no `claude`
    call should ever be attempted.
    """

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    def fake_which(name: str) -> str | None:
        raise AssertionError("shutil.which('claude') must not be called when there are no judge gates")

    monkeypatch.setattr(shutil, "which", fake_which)

    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    result = _invoke({"hook_event_name": "Stop", "cwd": str(repo)})
    assert result.exit_code == 0, result.output
    assert captured["json"]["judge_results"] == []


def test_stop_event_caps_the_number_of_judge_gates_evaluated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Each judge gate costs one sequential model call, unlike the other gate

    types' near-instant pattern matching, so an unbounded gate count would mean
    unbounded wall-clock on a single Stop event. Only the first
    _HOOK_JUDGE_MAX_GATES_PER_RUN gates (declaration order) get a `claude -p`
    call; the rest are skipped with a stderr message naming which.
    """
    (tmp_path / ".git").mkdir()
    gate_count = hook_cli._HOOK_JUDGE_MAX_GATES_PER_RUN + 2
    gates_yaml = "schema_version: '1.0'\npolicy:\n  id: test\ngates:\n" + "".join(
        f"  - id: judge-{i}\n    type: judge\n    enforcement: advisory\n    rubric: r{i}\n    message: m{i}\n"
        for i in range(gate_count)
    )
    (tmp_path / ".otari-gates.yml").write_text(gates_yaml, encoding="utf-8")

    claude_call_count = 0

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal claude_call_count
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[0] == "/usr/bin/claude":
            claude_call_count += 1
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=json.dumps({"outcome": "pass", "reasoning": "ok"}), stderr=""
            )
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)

    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    result = _invoke({"hook_event_name": "Stop", "cwd": str(tmp_path)})
    assert result.exit_code == 0, result.output

    submitted_ids = [entry["gate_id"] for entry in captured["json"]["judge_results"]]
    assert submitted_ids == [f"judge-{i}" for i in range(hook_cli._HOOK_JUDGE_MAX_GATES_PER_RUN)]
    assert claude_call_count == hook_cli._HOOK_JUDGE_MAX_GATES_PER_RUN
    assert "over the" in result.output
    for skipped_id in (f"judge-{i}" for i in range(hook_cli._HOOK_JUDGE_MAX_GATES_PER_RUN, gate_count)):
        assert skipped_id in result.output


def test_stop_event_skips_a_when_changed_judge_gate_that_does_not_apply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A judge gate scoped by `when_changed` costs nothing when it does not apply.

    No diff/transcript read and no `claude -p` call, since none of that work
    is needed to know the gate resolves `not_applicable`; only `git status`
    runs. Mirrors `test_a_policy_with_no_judge_gates_submits_no_judge_results`,
    but for a gate that exists and is simply out of scope for this session.
    """
    (tmp_path / ".git").mkdir()
    gates_yaml = (
        "schema_version: '1.0'\npolicy:\n  id: test\ngates:\n"
        "  - id: judge-src-only\n    type: judge\n    enforcement: advisory\n"
        "    rubric: r\n    when_changed: [src/**]\n    message: m\n"
    )
    (tmp_path / ".otari-gates.yml").write_text(gates_yaml, encoding="utf-8")

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=" M docs/README.md\0", stderr="")
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    def fake_which(name: str) -> str | None:
        raise AssertionError("shutil.which('claude') must not be called for an out-of-scope judge gate")

    monkeypatch.setattr(shutil, "which", fake_which)

    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    result = _invoke({"hook_event_name": "Stop", "cwd": str(tmp_path)})
    assert result.exit_code == 0, result.output
    assert captured["json"]["judge_results"] == []


def test_stop_event_runs_a_when_changed_judge_gate_that_applies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".git").mkdir()
    gates_yaml = (
        "schema_version: '1.0'\npolicy:\n  id: test\ngates:\n"
        "  - id: judge-src-only\n    type: judge\n    enforcement: advisory\n"
        "    rubric: r\n    when_changed: [src/**]\n    message: m\n"
    )
    (tmp_path / ".otari-gates.yml").write_text(gates_yaml, encoding="utf-8")

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=" M src/module.py\0", stderr="")
        if cmd[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[0] == "/usr/bin/claude":
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=json.dumps({"outcome": "pass", "reasoning": "ok"}), stderr=""
            )
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)

    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    result = _invoke({"hook_event_name": "Stop", "cwd": str(tmp_path)})
    assert result.exit_code == 0, result.output
    assert captured["json"]["judge_results"] == [{"gate_id": "judge-src-only", "outcome": "pass", "reasoning": "ok"}]


def test_pretooluse_submits_no_judge_results(monkeypatch: pytest.MonkeyPatch, judge_repo: Path) -> None:
    """Only a Stop event has a real diff and finished transcript to judge against.

    `None`, not `[]`: an empty list means "ran judge gates, found none
    applicable", which resolves a when_changed-matched judge gate `unknown`
    server-side rather than `not_applicable` (evaluate_judge's own
    docstring) -- an advisory warning on every single matching PreToolUse
    edit otherwise.
    """
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(judge_repo),
        "tool_name": "Edit",
        "tool_input": {"file_path": str(judge_repo / "README.md")},
    }
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    assert captured["json"]["judge_results"] is None


def test_pretooluse_shows_no_advisory_warning_for_a_not_applicable_judge_gate(
    monkeypatch: pytest.MonkeyPatch, judge_repo: Path
) -> None:
    """The real server's own response for this exact request shape (omitted

    `judge_results`, a matching edit): a `not_applicable` judge gate result,
    the fixed `evaluate_judge` behavior this whole change exists for. Passed
    through the CLI's own advisory-warning logic unmocked, confirming the
    fix reaches the thing a developer actually sees, not just the response
    the server sends.
    """
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _FakeResponse(
            {
                "blocked": False,
                "results": [
                    {
                        "gate_id": "follows-pattern",
                        "enforcement": "advisory",
                        "outcome": "not_applicable",
                        "message": "This event does not evaluate judge gates.",
                    }
                ],
            }
        ),
    )
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(judge_repo),
        "tool_name": "Edit",
        "tool_input": {"file_path": str(judge_repo / "README.md")},
    }
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    assert result.output == "", "a not_applicable outcome must never print an advisory warning"


def test_advisory_only_failure_warns_without_blocking(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    """blocked=False is set whenever nothing required failed, even if an advisory gate did.

    Checking only `blocked` before deciding whether to print anything would
    silently drop that advisory warning: it is never true on its own.
    """
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _FakeResponse(
            {
                "blocked": False,
                "results": [
                    {"gate_id": "g", "enforcement": "advisory", "outcome": "fail", "message": "please reconsider"}
                ],
            }
        ),
    )
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "Edit",
        "tool_input": {"file_path": str(repo / "README.md")},
    }
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    # Plain stderr text is invisible to Claude Code on a non-blocking hook
    # (it only reaches its own debug log): the message must be a
    # `systemMessage` on stdout, the field Claude Code's hook protocol
    # surfaces to the user for exactly this case.
    assert result.stderr == ""
    stdout_payload = json.loads(result.stdout)
    assert "please reconsider" in stdout_payload["systemMessage"]
    assert "advisory" in stdout_payload["systemMessage"].lower()


def test_advisory_warning_includes_the_judge_models_own_reasoning(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    """`message` is the gate's own fixed policy text, the same for every failure;

    `detail` is what a judge model actually found (`evaluate_judge`'s own
    `detail=verdict.reasoning`), specific to this one verdict. Modeled on the
    real evaluator's own response shape (both fields present, as it always
    sends them), not a synthetic one: a fix that only worked against a
    hand-picked payload shape would not prove much.
    """
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _FakeResponse(
            {
                "blocked": False,
                "results": [
                    {
                        "gate_id": "no-narrative-comments",
                        "enforcement": "advisory",
                        "outcome": "fail",
                        "message": "This diff may add a comment that restates the code.",
                        "detail": "The comment at src/module.py:42 narrates the change rather than explaining why.",
                    }
                ],
            }
        ),
    )
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "Edit",
        "tool_input": {"file_path": str(repo / "README.md")},
    }
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    stdout_payload = json.loads(result.stdout)
    assert "src/module.py:42" in stdout_payload["systemMessage"]
    assert "This diff may add a comment that restates the code." in stdout_payload["systemMessage"]


def test_pretooluse_submits_call_scoped_evidence(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    """One tool call's own command is call-scoped, which is what lets the

    server judge it with command_match and skip command_if_changed, rather
    than inferring either from an empty list.
    """
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "Bash",
        "tool_input": {"command": "npm install"},
    }
    assert _invoke(payload).exit_code == 0
    assert captured["json"]["command_scope"] == "call"


def test_stop_event_submits_session_scoped_evidence(
    monkeypatch: pytest.MonkeyPatch, repo: Path, tmp_path: Path
) -> None:
    """A Stop event really has seen every command the session ran, and saying

    so is what lets command_if_changed resolve at all and takes command_match
    out of the picture (where a cumulative match could never be cleared).
    """

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(_transcript_line(command="make postman") + "\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    payload = {"hook_event_name": "Stop", "cwd": str(repo), "transcript_path": str(transcript)}
    assert _invoke(payload).exit_code == 0
    assert captured["json"]["command_scope"] == "session"


def test_a_repeat_stop_block_says_the_block_is_finite(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    """Claude Code overrides a Stop hook after 8 consecutive blocks and lets the

    turn end. Blocking silently through that budget leaves a required gate
    looking clean at exactly the moment it is firing hardest, so a repeat
    block says what the budget is and that it is running out.
    """

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _FakeResponse(
            {
                "blocked": True,
                "results": [{"gate_id": "g", "enforcement": "required", "outcome": "fail", "message": "no"}],
            }
        ),
    )
    payload = {"hook_event_name": "Stop", "cwd": str(repo), "stop_hook_active": True}
    result = _invoke(payload)
    assert result.exit_code == 2
    assert "already blocked once this turn" in result.output
    assert "8 consecutive blocks" in result.output


def test_a_first_stop_block_does_not_mention_the_budget(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _FakeResponse(
            {
                "blocked": True,
                "results": [{"gate_id": "g", "enforcement": "required", "outcome": "fail", "message": "no"}],
            }
        ),
    )
    payload = {"hook_event_name": "Stop", "cwd": str(repo)}
    result = _invoke(payload)
    assert result.exit_code == 2
    assert "already blocked once" not in result.output


def _write_verifier(tmp_path: Path, name: str, body: str) -> Path:
    script = tmp_path / name
    script.write_text(f"#!/usr/bin/env bash\n{body}\n", encoding="utf-8")
    script.chmod(0o755)
    return script


def test_hook_run_check_verifier_passes_on_real_exit_zero(tmp_path: Path) -> None:
    """No mocking: a real script, run as a real subprocess, exiting 0."""
    _write_verifier(tmp_path, "v.sh", "exit 0")
    outcome, detail = hook_cli._hook_run_check_verifier(tmp_path, "v.sh", deadline=time.monotonic() + 10)
    assert outcome == "pass"
    assert detail == ""


def test_hook_run_check_verifier_fails_on_real_exit_one_and_captures_stdout(tmp_path: Path) -> None:
    _write_verifier(tmp_path, "v.sh", 'echo "conflicted.txt:2"\nexit 1')
    outcome, detail = hook_cli._hook_run_check_verifier(tmp_path, "v.sh", deadline=time.monotonic() + 10)
    assert outcome == "fail"
    assert detail == "conflicted.txt:2\n"


@pytest.mark.parametrize("exit_code", [2, 7, 255])
def test_hook_run_check_verifier_errors_on_other_exit_codes(tmp_path: Path, exit_code: int) -> None:
    _write_verifier(tmp_path, "v.sh", f"exit {exit_code}")
    outcome, _detail = hook_cli._hook_run_check_verifier(tmp_path, "v.sh", deadline=time.monotonic() + 10)
    assert outcome == "error"


def test_hook_run_check_verifier_errors_when_the_script_does_not_exist(tmp_path: Path) -> None:
    outcome, detail = hook_cli._hook_run_check_verifier(
        tmp_path, "does-not-exist.sh", deadline=time.monotonic() + 10
    )
    assert outcome == "error"
    assert "does not exist" in detail


def test_hook_run_check_verifier_errors_when_the_script_is_not_executable(tmp_path: Path) -> None:
    script = tmp_path / "v.sh"
    script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    # Deliberately not chmod +x: exec must raise PermissionError (an OSError).
    outcome, detail = hook_cli._hook_run_check_verifier(tmp_path, "v.sh", deadline=time.monotonic() + 10)
    assert outcome == "error"
    assert "v.sh" in detail


def test_hook_run_check_verifier_rejects_a_verifier_that_resolves_outside_the_repo_root(tmp_path: Path) -> None:
    """A relative path with enough `..` segments could otherwise climb out of the repo.

    domain.policy already rejects an absolute verifier at parse time, but a
    relative one is validated again here, against the real filesystem, since
    parsing has no filesystem to check against.
    """
    # "../outside.sh" from repo_root resolves to tmp_path/outside.sh: a real,
    # executable, existing script that a broken guard would happily run.
    outside = tmp_path / "outside.sh"
    outside.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    outside.chmod(0o755)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outcome, detail = hook_cli._hook_run_check_verifier(
        repo_root, f"../{outside.name}", deadline=time.monotonic() + 10
    )
    assert outcome == "error"
    assert "outside the repo root" in detail


def test_hook_run_check_verifier_errors_when_the_deadline_has_already_passed(tmp_path: Path) -> None:
    _write_verifier(tmp_path, "v.sh", "exit 0")
    outcome, detail = hook_cli._hook_run_check_verifier(tmp_path, "v.sh", deadline=time.monotonic() - 1)
    assert outcome == "error"
    assert "budget exhausted" in detail


def test_hook_run_check_verifier_times_out_on_a_real_slow_script(tmp_path: Path) -> None:
    _write_verifier(tmp_path, "v.sh", "sleep 5\nexit 0")
    # A near-zero remaining budget forces subprocess.run's own `timeout=` well
    # under the script's real 5s sleep, without waiting for _HOOK_CHECK_TIMEOUT_SECONDS.
    outcome, detail = hook_cli._hook_run_check_verifier(tmp_path, "v.sh", deadline=time.monotonic() + 0.05)
    assert outcome == "error"
    assert "did not respond" in detail


def test_hook_run_check_verifier_timeout_also_kills_a_background_child(tmp_path: Path) -> None:
    """A timed-out verifier takes anything it backgrounded with it.

    The verifier leaves `sleep 30` running, and that child inherits the
    captured pipes. Killing the verifier alone leaves the child holding them,
    outliving both this call's timeout and the whole run's shared budget, so
    the verifier runs in a process group of its own and the timeout kills the
    group.
    """
    _write_verifier(tmp_path, "v.sh", "sleep 30 &\necho $! > child.pid\nsleep 5\nexit 0")
    # A whole second, not the 0.05s the plain timeout test uses: the script has
    # to reach `echo $!` before the kill, or there is no recorded child to
    # assert about.
    outcome, detail = hook_cli._hook_run_check_verifier(tmp_path, "v.sh", deadline=time.monotonic() + 1)
    assert outcome == "error"
    assert "did not respond" in detail

    child_pid = int((tmp_path / "child.pid").read_text().strip())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:  # the signal is delivered asynchronously
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    pytest.fail(f"background child {child_pid} survived the verifier's timeout")


def test_hook_run_check_verifier_replaces_undecodable_output(tmp_path: Path) -> None:
    """Bytes that are not valid UTF-8 become replacement characters, not a crash.

    Strict decoding raises `UnicodeDecodeError` from inside `subprocess`
    itself, which is neither of the exceptions this function catches: it would
    escape and take every other gate in the policy down with it.
    """
    _write_verifier(tmp_path, "v.sh", r"""printf 'bad: \xff\xfe'""" + "\nexit 1")
    outcome, detail = hook_cli._hook_run_check_verifier(tmp_path, "v.sh", deadline=time.monotonic() + 10)
    assert outcome == "fail"
    assert detail.startswith("bad: ")
    assert "\ufffd" in detail


def test_hook_run_check_verifier_caps_detail_length(tmp_path: Path) -> None:
    _write_verifier(tmp_path, "v.sh", 'printf "%0.sx" {1..10000}\nexit 1')
    outcome, detail = hook_cli._hook_run_check_verifier(tmp_path, "v.sh", deadline=time.monotonic() + 10)
    assert outcome == "fail"
    assert len(detail) == hook_cli._HOOK_MAX_CHECK_DETAIL_LENGTH


def test_check_passed_gates_run_concurrently_not_sequentially(tmp_path: Path) -> None:
    """Five check_passed gates, each a real script sleeping ~0.3s, must finish in
    well under 5 * 0.3s: `_hook_collect_check_verdicts` runs verifiers through a
    `ThreadPoolExecutor` (`_HOOK_GATE_MAX_WORKERS`), not one after another. No
    mocking: real scripts, run as real subprocesses, the same as the
    `_hook_run_check_verifier` tests above.
    """
    gate_count = 5
    per_gate_seconds = 0.3
    for i in range(gate_count):
        _write_verifier(tmp_path, f"v{i}.sh", f"sleep {per_gate_seconds}\nexit 0")
    gates_yaml = "schema_version: '1.0'\npolicy:\n  id: test\ngates:\n" + "".join(
        f"  - id: g{i}\n    type: check_passed\n    enforcement: required\n"
        f"    verifier: v{i}.sh\n    message: m{i}\n"
        for i in range(gate_count)
    )

    start = time.monotonic()
    results = hook_cli._hook_collect_check_verdicts(gates_yaml, tmp_path / ".otari-gates.yml", tmp_path, [])
    elapsed = time.monotonic() - start

    assert [result["gate_id"] for result in results] == [f"g{i}" for i in range(gate_count)]
    assert all(result["outcome"] == "pass" for result in results)
    # gate_count * per_gate_seconds is the sleep time alone a fully
    # sequential run could not possibly finish under, real subprocess
    # spawn overhead on top of that not even counted; no fudge factor
    # needed for this bound to be sound.
    assert elapsed < gate_count * per_gate_seconds, (
        f"took {elapsed:.2f}s for {gate_count} gates at {per_gate_seconds}s each -- looks sequential"
    )


_CHECK_GATES_YAML_TEMPLATE = (
    "schema_version: '1.0'\n"
    "policy:\n  id: test\n"
    "gates:\n"
    "  - id: no-leftover-conflict-markers\n"
    "    type: check_passed\n"
    "    enforcement: required\n"
    "    verifier: {verifier}\n"
    "    message: A tracked file still carries a Git merge-conflict marker.\n"
)


@pytest.fixture
def check_repo(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    _write_verifier(tmp_path, "verify.sh", "exit 0")
    (tmp_path / ".otari-gates.yml").write_text(
        _CHECK_GATES_YAML_TEMPLATE.format(verifier="verify.sh"), encoding="utf-8"
    )
    return tmp_path


def _git_status_only_run(git_status_stdout: str = "") -> Any:
    """Fake `git status`; every other call (the verifier script itself) runs for real.

    Unlike the judge tests' own dispatchers, which mock every subprocess.run
    call including `claude -p`, this leaves the check_passed verifier's own
    execution real: the point of these tests is to exercise a real script
    under a real subprocess, not a second copy of `_hook_run_check_verifier`
    that just returns a canned result.
    """
    real_run = subprocess.run

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=git_status_stdout, stderr="")
        return real_run(cmd, **kwargs)

    return fake_run


def test_stop_event_submits_a_check_verdict_from_the_verifier_script(
    monkeypatch: pytest.MonkeyPatch, check_repo: Path
) -> None:
    """End to end through `hook()`, with a real verifier script actually executed

    (only `git status` is mocked): the Stop event runs the check_passed
    gate's verifier and submits its real exit-code-derived verdict as
    `check_results`.
    """
    monkeypatch.setattr(subprocess, "run", _git_status_only_run())
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    result = _invoke({"hook_event_name": "Stop", "cwd": str(check_repo)})
    assert result.exit_code == 0, result.output
    assert captured["json"]["check_results"] == [
        {"gate_id": "no-leftover-conflict-markers", "outcome": "pass", "detail": ""}
    ]


def test_stop_event_submits_a_failing_check_verdict_and_blocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    _write_verifier(tmp_path, "verify.sh", 'echo "conflicted.txt:2"\nexit 1')
    (tmp_path / ".otari-gates.yml").write_text(
        _CHECK_GATES_YAML_TEMPLATE.format(verifier="verify.sh"), encoding="utf-8"
    )

    monkeypatch.setattr(subprocess, "run", _git_status_only_run())
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse(
            {
                "blocked": True,
                "results": [
                    {
                        "gate_id": "no-leftover-conflict-markers",
                        "enforcement": "required",
                        "outcome": "fail",
                        "message": "A tracked file still carries a Git merge-conflict marker.",
                        "detail": "conflicted.txt:2",
                    }
                ],
            }
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = _invoke({"hook_event_name": "Stop", "cwd": str(tmp_path)})
    assert result.exit_code == 2, result.output
    assert captured["json"]["check_results"] == [
        {"gate_id": "no-leftover-conflict-markers", "outcome": "fail", "detail": "conflicted.txt:2\n"}
    ]


def test_pretooluse_submits_no_check_results(monkeypatch: pytest.MonkeyPatch, check_repo: Path) -> None:
    """A PreToolUse call has no finished session for a verifier to check yet: `check_results`

    must be omitted (None), not an empty list, so a required check_passed
    gate resolves not_applicable rather than the unknown a genuinely missing
    verdict would (see docs/agent-gates.md).
    """
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(check_repo),
        "tool_name": "Edit",
        "tool_input": {"file_path": str(check_repo / "README.md")},
    }
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    assert captured["json"]["check_results"] is None


def test_stop_event_skips_check_passed_gates_that_when_changed_excludes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A local `when_changed` skip means the verifier is never even run, not just excluded

    from the wire result: the script here would fail if it ran (`exit 1`),
    and the assertion below confirms the gate resolves via an empty
    check_results list, never having invoked it.
    """
    (tmp_path / ".git").mkdir()
    _write_verifier(tmp_path, "verify.sh", "exit 1")
    policy = (
        "schema_version: '1.0'\n"
        "policy:\n  id: test\n"
        "gates:\n"
        "  - id: g\n"
        "    type: check_passed\n"
        "    enforcement: required\n"
        "    verifier: verify.sh\n"
        "    when_changed: ['src/**']\n"
        "    message: m\n"
    )
    (tmp_path / ".otari-gates.yml").write_text(policy, encoding="utf-8")

    monkeypatch.setattr(subprocess, "run", _git_status_only_run(git_status_stdout=" M docs/README.md\0"))
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        captured["json"] = kwargs.get("json")
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    result = _invoke({"hook_event_name": "Stop", "cwd": str(tmp_path)})
    assert result.exit_code == 0, result.output
    assert captured["json"]["check_results"] == []


def test_collect_check_verdicts_skips_gates_over_the_per_run_limit(tmp_path: Path) -> None:
    _write_verifier(tmp_path, "verify.sh", "exit 0")
    gates_yaml = ["schema_version: '1.0'\npolicy:\n  id: test\ngates:"]
    gates_yaml.extend(
        f"  - id: g{i}\n    type: check_passed\n    enforcement: required\n    verifier: verify.sh\n    message: m"
        for i in range(hook_cli._HOOK_CHECK_MAX_GATES_PER_RUN + 1)
    )
    gates_file = tmp_path / ".otari-gates.yml"
    policy_yaml = "\n".join(gates_yaml) + "\n"
    gates_file.write_text(policy_yaml, encoding="utf-8")

    results = hook_cli._hook_collect_check_verdicts(policy_yaml, gates_file, tmp_path, [])
    assert len(results) == hook_cli._HOOK_CHECK_MAX_GATES_PER_RUN
    assert {r["outcome"] for r in results} == {"pass"}


def test_collect_check_verdicts_returns_empty_for_an_unparseable_policy(tmp_path: Path) -> None:
    gates_file = tmp_path / ".otari-gates.yml"
    assert hook_cli._hook_collect_check_verdicts("not: valid: yaml: at: all:", gates_file, tmp_path, []) == []
