"""Unit tests for the code-execution conformance script.

The script is what a third-party backend runs to show it implements contract
version 1, so its own failure modes matter as much as its happy path: a checker
that passes a non-conforming backend is worse than no checker. Each test here
serves one deliberately broken backend and asserts the script catches it.

The backends are `httpx.MockTransport` handlers rather than a live container, so
these run in the unit suite with no Docker.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_code_execution_conformance.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_code_execution_conformance", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


conformance = _load()

_MARKER = conformance._MARKER
_FILE_BODY = f"{_MARKER}\n".encode()
_SESSION_ID = "3f6c1a9e8b2d4c7f"
_EXEC_PATH = re.compile(rf"^/sessions/{_SESSION_ID}/exec$")


@dataclass
class FakeBackend:
    """A conforming backend, with a knob per way of being wrong."""

    envelope: bool = True
    serve_files: bool = True
    confine_paths: bool = True
    refuse_unknown_tool: bool = True
    unknown_tool_status: int = 400
    escape_status: int = 403
    # Drops the session the moment the file operations are reached, which is the
    # other thing a 404 on a file route can mean.
    lose_session_at_files: bool = False
    reported_idle_timeout: int = 300
    download_body: bytes = _FILE_BODY
    calls: list[tuple[str, str]] = field(default_factory=list)
    _session_lost: bool = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append((request.method, path))

        if self._session_lost and _SESSION_ID in path:
            return httpx.Response(404, json={"detail": "no such session"})
        if request.method == "POST" and path == "/sessions":
            return httpx.Response(
                201,
                json={
                    "session_id": _SESSION_ID,
                    "created_at": 1786000000.0,
                    "last_activity_at": 1786000000.0,
                    "idle_timeout_seconds": self.reported_idle_timeout,
                    "max_lifetime_seconds": 3600,
                },
            )
        if request.method == "DELETE" and path == f"/sessions/{_SESSION_ID}":
            return httpx.Response(204)
        if request.method == "POST" and _EXEC_PATH.match(path):
            return self._exec(json.loads(request.content))
        if path == f"/sessions/{_SESSION_ID}/files/list":
            return self._list_files()
        if path == f"/sessions/{_SESSION_ID}/files":
            return self._upload() if request.method == "POST" else self._download(request)
        return httpx.Response(404, json={"detail": f"no route for {request.method} {path}"})

    def _exec(self, body: dict[str, Any]) -> httpx.Response:
        tool = body["tool"]
        if tool not in ("code_execution", "bash_code_execution", "text_editor_code_execution"):
            if self.refuse_unknown_tool:
                return httpx.Response(self.unknown_tool_status, json={"detail": f"unknown tool: {tool}"})
            return httpx.Response(200, json=self._envelope(body, "ran it anyway"))
        # A text-editor `create` reports the write; every other call the script
        # makes echoes the marker, which is how it knows code actually ran.
        creating = tool == "text_editor_code_execution" and body["input"]["command"] == "create"
        stdout = "File created successfully" if creating else f"{_MARKER}\n"
        return httpx.Response(200, json=self._envelope(body, stdout))

    def _envelope(self, request_body: dict[str, Any], stdout: str) -> dict[str, Any]:
        # Echoed when the request carried one, generated otherwise, as the
        # contract has it.
        tool_use_id = request_body.get("tool_use_id") or "srvtoolu_generated"
        block = {
            "type": "code_execution_tool_result",
            "tool_use_id": tool_use_id,
            "content": {
                "type": "code_execution_result",
                "stdout": stdout,
                "stderr": "",
                "return_code": 0,
                "content": [],
            },
        }
        if not self.envelope:
            # The mistake the typed client was written to catch: the result block
            # returned bare, at the top level, instead of under `result_block`.
            return block
        return {"tool_use_id": tool_use_id, "execution_time_ms": 12, "result_block": block}

    def _upload(self) -> httpx.Response:
        if self.lose_session_at_files:
            self._session_lost = True
            return httpx.Response(404, json={"detail": "no such session"})
        if not self.serve_files:
            return httpx.Response(404, json={"detail": "not implemented"})
        return httpx.Response(201, json={"path": "conformance.txt", "size_bytes": len(_FILE_BODY)})

    def _list_files(self) -> httpx.Response:
        if not self.serve_files:
            return httpx.Response(404, json={"detail": "not implemented"})
        return httpx.Response(
            200,
            json={
                "files": [
                    {
                        "path": "conformance.txt",
                        "size_bytes": len(_FILE_BODY),
                        "mime_type": "text/plain",
                        "modified_at": 1786000000.0,
                    }
                ]
            },
        )

    def _download(self, request: httpx.Request) -> httpx.Response:
        if not self.serve_files:
            return httpx.Response(404, json={"detail": "not implemented"})
        escaping = ".." in (request.url.params.get("path") or "")
        if escaping and self.confine_paths:
            return httpx.Response(self.escape_status, json={"detail": "path escapes the workspace"})
        return httpx.Response(200, content=self.download_body)


def _run(backend: FakeBackend) -> list[Any]:
    spec = conformance.load_spec()
    with httpx.Client(base_url="http://sandbox", transport=httpx.MockTransport(backend.handler)) as client:
        checks: list[Any] = conformance.run_checks(client, spec, timeout_seconds=5)
    return checks


def _failures(checks: list[Any]) -> dict[str, str]:
    return {check.name: check.detail for check in checks if check.status == conformance.FAIL}


def _statuses(checks: list[Any]) -> dict[str, str]:
    return {check.name: check.status for check in checks}


def test_conforming_backend_passes() -> None:
    checks = _run(FakeBackend())
    assert not _failures(checks)
    assert conformance.report(checks) == 0
    # Every operation the contract requires was actually exercised.
    assert {"CreateSession", "DestroySession", "PutFile", "ListFiles", "GetFile"} <= set(_statuses(checks))
    assert sum(1 for check in checks if check.name.startswith("Execute ")) == 5


def test_bare_result_block_fails_execute() -> None:
    """The latent break the typed client closed, now caught at the source."""
    failures = _failures(_run(FakeBackend(envelope=False)))
    assert "Execute code_execution" in failures
    assert "does not match ExecResponse" in failures["Execute code_execution"]


def test_missing_file_operations_are_skipped_not_failed() -> None:
    checks = _run(FakeBackend(serve_files=False))
    statuses = _statuses(checks)
    assert [statuses[name] for name in ("PutFile", "ListFiles", "GetFile")] == [conformance.SKIP] * 3
    assert not _failures(checks)
    assert conformance.report(checks) == 0


def test_serving_a_path_outside_the_workspace_fails() -> None:
    failures = _failures(_run(FakeBackend(confine_paths=False)))
    assert "GetFile confines access to the session workspace" in failures


def test_running_an_unknown_tool_kind_fails() -> None:
    failures = _failures(_run(FakeBackend(refuse_unknown_tool=False)))
    assert "Execute refuses an unknown tool kind" in failures


def test_falling_over_on_an_unknown_tool_kind_is_not_a_refusal() -> None:
    """A 500 means the backend broke on input it was supposed to reject."""
    failures = _failures(_run(FakeBackend(unknown_tool_status=500)))
    assert "Execute refuses an unknown tool kind" in failures


def test_falling_over_on_a_traversal_attempt_fails() -> None:
    failures = _failures(_run(FakeBackend(escape_status=500)))
    assert "GetFile confines access to the session workspace" in failures


def test_a_lost_session_is_not_reported_as_absent_file_operations() -> None:
    """A 404 on a file route means one of two things, and only one is a skip.

    A backend that dropped the session mid-run would otherwise pass as one that
    simply does not implement the optional operations.
    """
    checks = _run(FakeBackend(lose_session_at_files=True))
    assert "the session was gone" in _failures(checks)["File operations"]
    assert conformance.SKIP not in set(_statuses(checks).values())


def test_lifetime_hint_reported_above_the_request_fails() -> None:
    """A backend may clamp a lifetime hint downward, never upward.

    Upward is what breaks a client: the handle is its only statement of how long
    the session lives.
    """
    failures = _failures(_run(FakeBackend(reported_idle_timeout=99_999)))
    assert "CreateSession honours lifetime hints" in failures


def test_downloaded_bytes_must_match_what_was_written() -> None:
    failures = _failures(_run(FakeBackend(download_body=b"something else")))
    assert "GetFile" in failures


def test_session_is_released_even_when_a_check_fails() -> None:
    """A checker that leaks sessions cannot be run against a live backend twice."""
    backend = FakeBackend(envelope=False)
    checks = _run(backend)
    assert _failures(checks)
    assert ("DELETE", f"/sessions/{_SESSION_ID}") in backend.calls


def test_unreachable_backend_reports_a_failure_rather_than_raising() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    spec = conformance.load_spec()
    with httpx.Client(base_url="http://sandbox", transport=httpx.MockTransport(refuse)) as client:
        checks = conformance.run_checks(client, spec, timeout_seconds=5)
    assert list(_failures(checks)) == ["CreateSession"]
    assert conformance.report(checks) == 1


def test_main_reports_and_exits_nonzero_on_a_broken_backend(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    backend = FakeBackend(envelope=False)
    real_client = httpx.Client

    def client_with_fake_transport(**kwargs: Any) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(backend.handler)
        return real_client(**kwargs)

    monkeypatch.setattr(conformance.httpx, "Client", client_with_fake_transport)
    exit_code = conformance.main(["--base-url", "http://sandbox/", "--auth-token", "secret"])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "does not conform to code-execution contract version 1" in output
    assert "secret" not in output
