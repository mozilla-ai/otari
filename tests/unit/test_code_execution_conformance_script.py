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
    # A read-only backend: reads and listings served, writes not. Conforming, since
    # each file operation is independently optional.
    serve_writes: bool = True
    confine_paths: bool = True
    refuse_unknown_tool: bool = True
    unknown_tool_status: int = 400
    escape_status: int = 403
    # Drops the session the moment the file operations are reached, which is the
    # other thing a 404 on a file route can mean.
    lose_session_at_files: bool = False
    # A ceiling of the backend's own, refused rather than clamped, as the contract
    # allows. `None` accepts any budget.
    budget_ceiling_seconds: int | None = None
    # Typed loosely on purpose: JSON has one numeric type, so a conforming backend
    # may serialise an integer timeout as `99999.0`.
    reported_idle_timeout: float = 300
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
        ceiling = self.budget_ceiling_seconds
        if ceiling is not None and int(body.get("timeout_seconds") or 0) > ceiling:
            return httpx.Response(422, json={"detail": f"timeout_seconds exceeds the {ceiling}s ceiling"})
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
        if not self.serve_files or not self.serve_writes:
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


def _run(backend: FakeBackend, *, timeout_seconds: int = 5) -> list[Any]:
    spec = conformance.load_spec()
    with httpx.Client(base_url="http://sandbox", transport=httpx.MockTransport(backend.handler)) as client:
        checks: list[Any] = conformance.run_checks(client, spec, timeout_seconds=timeout_seconds)
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


def test_a_read_only_backend_keeps_its_served_operations_checked() -> None:
    """Each file operation is independently optional, so each is probed on its own.

    Gating all three on the upload skipped a read-only backend's served
    operations, and took the workspace-confinement check down with them: the one
    check here that is about safety rather than shape.
    """
    checks = _run(FakeBackend(serve_writes=False))
    statuses = _statuses(checks)

    assert statuses["PutFile"] == conformance.SKIP
    assert statuses["ListFiles"] == conformance.PASS
    assert statuses["GetFile"] == conformance.PASS
    assert statuses["GetFile confines access to the session workspace"] == conformance.PASS
    assert not _failures(checks)


def test_a_backend_serving_no_reads_skips_confinement_rather_than_passing_it() -> None:
    """Nothing to confine is a skip, never a pass: a pass would claim a check that never ran."""
    statuses = _statuses(_run(FakeBackend(serve_files=False)))
    assert statuses["GetFile confines access to the session workspace"] == conformance.SKIP


def test_missing_file_operations_are_skipped_not_failed() -> None:
    checks = _run(FakeBackend(serve_files=False))
    statuses = _statuses(checks)
    assert [statuses[name] for name in ("PutFile", "ListFiles", "GetFile")] == [conformance.SKIP] * 3
    assert not _failures(checks)


def test_a_backend_that_caps_the_execution_budget_still_conforms() -> None:
    """Refusing an over-cap budget is contract-permitted, so it cannot be a failure.

    Without the retry, `--timeout-seconds 300` reported the reference backend,
    which caps at 120 and refuses rather than clamping, as non-conforming to the
    contract it is the reference for.
    """
    checks = _run(FakeBackend(budget_ceiling_seconds=120), timeout_seconds=300)

    assert not _failures(checks)
    assert conformance.report(checks) == 0
    # The refusal is reported rather than hidden: the run says which budget held.
    executed = [check for check in checks if check.name == "Execute code_execution"]
    assert "refused a 300s budget" in executed[0].detail
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


def test_a_float_lifetime_hint_above_the_request_also_fails() -> None:
    """JSON has one numeric type, so an integer field can arrive as `99999.0`.

    That backend is conforming to the schema (a zero-fraction float is an integer
    in JSON Schema) and arrives here as a Python float, which an `isinstance(int)`
    guard skipped, in precisely the case the check exists to catch.
    """
    failures = _failures(_run(FakeBackend(reported_idle_timeout=99_999.0)))
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


def test_a_non_positive_execution_budget_is_rejected_at_the_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rejected before a session is leased, and blamed on the invocation.

    The contract's minimum is 1 second, so a 0 would otherwise fail on the first
    request as "this script built a non-conforming payload". Both halves are
    asserted: a nonzero exit, since `SystemExit(0)` would be a success the CLI
    never reached, and a backend that saw no traffic, since a session leased
    before the argument was refused is one the run never releases.
    """
    seen: list[tuple[str, str]] = []
    real_client = httpx.Client

    def record(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        return httpx.Response(500)

    def client_with_recording_transport(**kwargs: Any) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(record)
        return real_client(**kwargs)

    monkeypatch.setattr(conformance.httpx, "Client", client_with_recording_transport)

    with pytest.raises(SystemExit) as exit_info:
        conformance.main(["--base-url", "http://sandbox", "--timeout-seconds", "0"])

    assert exit_info.value.code != 0
    assert seen == []


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
    # Both streams: a credential that leaked into a traceback or a warning would
    # go to stderr, which an stdout-only assertion would wave through.
    captured = capsys.readouterr()
    assert "does not conform to code-execution contract version 1" in captured.out
    assert "secret" not in captured.out
    assert "secret" not in captured.err
