"""Unit tests for `otari hook`, the native hook-protocol transport.

Mocks the network boundary (httpx.post) and the Git boundary (subprocess.run)
so these run with no server and no real repository; gateway.agent_runtime's
own evaluation is covered separately in tests/unit/agent_runtime/ and
tests/integration/test_hooks_route.py.
"""

import json
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


def test_pretooluse_ignores_non_edit_tools(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    called = False

    def fake_post(*args: object, **kwargs: object) -> _FakeResponse:
        nonlocal called
        called = True
        return _FakeResponse({"blocked": False, "results": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    payload = {
        "hook_event_name": "PreToolUse",
        "cwd": str(repo),
        "tool_name": "Bash",
        "tool_input": {"command": "echo hi"},
    }
    result = _invoke(payload)
    assert result.exit_code == 0, result.output
    assert not called, "a non-edit tool call must never reach the Hook Server"


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
