"""Unit tests for `otari hook`, the native hook-protocol transport.

Mocks the network boundary (httpx.post) and the Git boundary (subprocess.run)
so these run with no server and no real repository; gateway.agent_runtime's
own evaluation is covered separately in tests/unit/agent_runtime/ and
tests/integration/test_hooks_route.py.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

import gateway.cli as gateway_cli
from gateway.core.config import GatewayConfig

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
    monkeypatch.setattr(gateway_cli, "_hook_judge_log_path", lambda: tmp_path / "judge-calls.log")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".otari-gates.yml").write_text(_GATES_YAML, encoding="utf-8")
    return tmp_path


@pytest.fixture
def config_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_config(config_path: str | None = None) -> GatewayConfig:
        return GatewayConfig(master_key="test-master-key")

    monkeypatch.setattr(gateway_cli, "load_config", fake_load_config)


def _invoke(payload: dict[str, Any], **extra_args: str) -> Any:
    args = ["--api-key", "test-key"]
    for key, value in extra_args.items():
        args += [f"--{key.replace('_', '-')}", value]
    return CliRunner().invoke(gateway_cli.hook, args, input=json.dumps(payload))


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
    assert len(sent[0]) == gateway_cli._HOOK_MAX_COMMAND_LENGTH
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
        _transcript_line(command=f"cmd{i}", tool_use_id=f"toolu_{i}") for i in range(gateway_cli._HOOK_MAX_COMMANDS + 1)
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


def test_outside_a_git_repo_is_a_no_op(tmp_path: Path) -> None:
    result = _invoke({"hook_event_name": "Stop", "cwd": str(tmp_path)})
    assert result.exit_code == 0, result.output


def test_malformed_stdin_is_a_no_op() -> None:
    result = CliRunner().invoke(gateway_cli.hook, ["--api-key", "test-key"], input="not json")
    assert result.exit_code == 0, result.output


def test_missing_credential_does_not_block(monkeypatch: pytest.MonkeyPatch, repo: Path, config_stub: None) -> None:
    def fake_load_config(config_path: str | None = None) -> GatewayConfig:
        return GatewayConfig(master_key=None)

    monkeypatch.setattr(gateway_cli, "load_config", fake_load_config)
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "Edit",
        "tool_input": {"file_path": str(repo / "CHANGELOG.md")},
    }
    result = CliRunner().invoke(gateway_cli.hook, [], input=json.dumps(payload))
    assert result.exit_code == 0, result.output
    assert "no API key or master key resolved" in result.output


def test_invalid_config_does_not_block(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    """load_config runs GatewayConfig.validate_mode_selection(), which raises

    ValueError on a real misconfiguration (e.g. OTARI_MODE=hybrid with no
    OTARI_AI_TOKEN set). That is a setup problem, not a required gate
    failing, so it must fail open like every other setup failure this
    command handles, not surface as an unhandled traceback.
    """

    def fake_load_config(config_path: str | None = None) -> GatewayConfig:
        raise ValueError("Hybrid mode (legacy value 'platform') requires OTARI_AI_TOKEN to be set.")

    monkeypatch.setattr(gateway_cli, "load_config", fake_load_config)
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


def test_falls_back_to_configured_master_key_and_localhost(
    monkeypatch: pytest.MonkeyPatch, repo: Path, config_stub: None
) -> None:
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
    result = CliRunner().invoke(gateway_cli.hook, [], input=json.dumps(payload))
    assert result.exit_code == 0, result.output
    assert captured["url"] == "http://localhost:8000/api/v1/hooks/check"
    # The bare token, not a ``Bearer `` prefix: deps._extract_bearer_token
    # tolerates the prefix for back-compat, but a header named for the key
    # carries the raw token.
    assert captured["headers"]["Otari-Key"] == "test-master-key"


def test_strip_judge_code_fence_recovers_json_wrapped_in_a_json_fence() -> None:
    fenced = '```json\n{"outcome": "pass", "reasoning": "fine"}\n```'
    assert gateway_cli._hook_strip_judge_code_fence(fenced) == '{"outcome": "pass", "reasoning": "fine"}'


def test_strip_judge_code_fence_recovers_json_wrapped_in_a_bare_fence() -> None:
    fenced = '```\n{"outcome": "pass", "reasoning": "fine"}\n```'
    assert gateway_cli._hook_strip_judge_code_fence(fenced) == '{"outcome": "pass", "reasoning": "fine"}'


def test_strip_judge_code_fence_leaves_unfenced_json_unchanged() -> None:
    unfenced = '{"outcome": "pass", "reasoning": "fine"}'
    assert gateway_cli._hook_strip_judge_code_fence(unfenced) == unfenced


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
            assert cmd[1:3] == ["--model", gateway_cli._HOOK_JUDGE_DEFAULT_MODEL], (
                "a judge call defaults to the cheaper model, not the session's own"
            )
            assert kwargs.get("cwd") == gateway_cli._hook_judge_workdir(), (
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
        gateway_cli.hook, args, input=json.dumps({"hook_event_name": "Stop", "cwd": str(judge_repo)})
    )
    assert result.exit_code == 0, result.output

    [judge_result] = captured["json"]["judge_results"]
    assert judge_result["gate_id"] == "follows-pattern"
    assert judge_result["outcome"] == "error"
    assert "--judge-dry-run" in judge_result["reasoning"]
    assert "tokens estimated" in judge_result["reasoning"]

    log_lines = gateway_cli._hook_judge_log_path().read_text(encoding="utf-8").splitlines()
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


def test_stop_event_reports_error_when_claude_is_not_on_path(
    monkeypatch: pytest.MonkeyPatch, judge_repo: Path
) -> None:
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
        {"gate_id": "follows-pattern", "outcome": "error", "reasoning": "the `claude` CLI was not found on PATH"}
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
    oversize_diff = "x" * (gateway_cli._HOOK_JUDGE_MAX_DIFF_CHARS + 1)

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if cmd[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=oversize_diff, stderr="")
        if cmd[0] == "/usr/bin/claude":
            prompt = cmd[-1]
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
    big_text = "x" * (gateway_cli._HOOK_JUDGE_MAX_TRANSCRIPT_CHARS + 1)
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

    extracted = gateway_cli._hook_extract_judge_transcript(transcript)
    assert extracted == "Adding a helper for p95 latency."


def test_judge_transcript_extraction_returns_empty_for_an_unreadable_file(tmp_path: Path) -> None:
    assert gateway_cli._hook_extract_judge_transcript(tmp_path / "missing.jsonl") == ""


def test_judge_workdir_is_not_the_repo_being_judged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A judge call's own `cwd` is never the caller's repo, or a real early version of

    this recurses into itself: that repo's own `.claude/settings.local.json`
    registers `otari hook` for `Stop`, so an unguarded call whose own `Stop`
    hook is this same command triggers it again.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    workdir = gateway_cli._hook_judge_workdir()
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
    oversize_reasoning = "x" * (gateway_cli._HOOK_MAX_JUDGE_REASONING_LENGTH + 1)

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
    assert len(judge_result["reasoning"]) == gateway_cli._HOOK_MAX_JUDGE_REASONING_LENGTH


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
            prompt = cmd[-1]
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


def test_a_policy_with_no_judge_gates_submits_no_judge_results(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
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


def test_stop_event_caps_the_number_of_judge_gates_evaluated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each judge gate costs one sequential model call, unlike the other gate

    types' near-instant pattern matching, so an unbounded gate count would mean
    unbounded wall-clock on a single Stop event. Only the first
    _HOOK_JUDGE_MAX_GATES_PER_RUN gates (declaration order) get a `claude -p`
    call; the rest are skipped with a stderr message naming which.
    """
    (tmp_path / ".git").mkdir()
    gate_count = gateway_cli._HOOK_JUDGE_MAX_GATES_PER_RUN + 2
    gates_yaml = "schema_version: '1.0'\npolicy:\n  id: test\ngates:\n" + "".join(
        f"  - id: judge-{i}\n    type: judge\n    enforcement: advisory\n"
        f"    rubric: r{i}\n    message: m{i}\n"
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
    assert submitted_ids == [f"judge-{i}" for i in range(gateway_cli._HOOK_JUDGE_MAX_GATES_PER_RUN)]
    assert claude_call_count == gateway_cli._HOOK_JUDGE_MAX_GATES_PER_RUN
    assert "over the" in result.output
    for skipped_id in (f"judge-{i}" for i in range(gateway_cli._HOOK_JUDGE_MAX_GATES_PER_RUN, gate_count)):
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
    """Only a Stop event has a real diff and finished transcript to judge against."""
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
    assert captured["json"]["judge_results"] == []


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
