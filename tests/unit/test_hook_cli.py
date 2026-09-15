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
            {"blocked": True, "results": [{"gate_id": "g", "outcome": "fail", "message": "no"}]}
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
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=" M CHANGELOG.md\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _FakeResponse(
            {"blocked": True, "results": [{"gate_id": "g", "outcome": "fail", "message": "no"}]}
        ),
    )
    payload = {"hook_event_name": "Stop", "cwd": str(repo)}
    result = _invoke(payload)
    assert result.exit_code == 2, result.output


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
    assert captured["headers"]["Otari-Key"] == "Bearer test-master-key"
