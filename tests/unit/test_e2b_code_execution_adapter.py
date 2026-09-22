"""Unit tests for the E2B adapter, against a stub SDK.

The real SDK is an optional extra and talks to a paid service, so what these
cover is the translation: what E2B reports becomes the contract's result block,
its filesystem becomes the workspace the backend above the port diffs, and its
failures become the two errors every adapter raises.
"""

from __future__ import annotations

import posixpath
import sys
import types
from datetime import UTC, datetime
from typing import Any

import pytest

from gateway.adapters import e2b_code_execution_adapter as e2b_adapter
from gateway.adapters.code_execution_adapter import (
    ProtocolCodeExecutionAdapter,
    build_code_execution_port,
    verify_code_execution_ready,
)
from gateway.core.config import GatewayConfig
from gateway.ports.code_execution_port import (
    OutputOverBudget,
    SandboxNotReachableError,
    SandboxSessionGoneError,
    SandboxUnavailableError,
)

ROOT = e2b_adapter.WORKSPACE_ROOT


class _AuthenticationException(Exception): ...


class _RateLimitException(Exception): ...


class _ServiceBusyException(Exception): ...


class _TimeoutException(Exception): ...


class _SandboxException(Exception): ...


class _NotFoundException(Exception): ...


class _FileType:
    DIR = "dir"
    FILE = "file"


class _Entry:
    def __init__(self, path: str, size: int, *, is_dir: bool = False, modified: datetime | None = None) -> None:
        self.path = path
        self.size = size
        self.type = _FileType.DIR if is_dir else _FileType.FILE
        self.modified_time = modified


class _Logs:
    def __init__(self, stdout: list[str], stderr: list[str]) -> None:
        self.stdout = stdout
        self.stderr = stderr


class _Result:
    def __init__(self, text: str | None) -> None:
        self.text = text


class _Error:
    def __init__(self, name: str, value: str, traceback: str | None) -> None:
        self.name = name
        self.value = value
        self.traceback = traceback


class _Execution:
    def __init__(
        self,
        stdout: list[str] | None = None,
        stderr: list[str] | None = None,
        results: list[_Result] | None = None,
        error: _Error | None = None,
    ) -> None:
        self.logs = _Logs(stdout or [], stderr or [])
        self.results = results or []
        self.error = error


class _Files:
    def __init__(self, tree: dict[str, bytes]) -> None:
        self.tree = tree
        self.written: list[tuple[str, bytes]] = []

    async def write(self, path: str, data: bytes) -> None:
        self.tree[path] = data
        self.written.append((path, data))

    async def read(self, path: str, format: str = "bytes") -> bytes:  # noqa: A002 - the SDK's own name
        del format
        if path not in self.tree:
            raise _NotFoundException(path)
        return self.tree[path]

    async def list(self, path: str, depth: int = 1) -> list[_Entry]:
        del depth
        prefix = path.rstrip("/") + "/"
        return [_Entry(p, len(b)) for p, b in self.tree.items() if p.startswith(prefix)]


class _Sandbox:
    sandbox_id = "sbx_test"

    def __init__(self, execution: _Execution | Exception | None = None, tree: dict[str, bytes] | None = None) -> None:
        self._execution = execution or _Execution(stdout=["ok\n"])
        self.files = _Files(tree if tree is not None else {})
        self.killed = False

    async def run_code(self, code: str, timeout: int, request_timeout: int) -> _Execution:
        del code, timeout, request_timeout
        if isinstance(self._execution, Exception):
            raise self._execution
        return self._execution

    async def kill(self) -> None:
        self.killed = True

    async def set_timeout(self, timeout: int) -> None:
        self.held_for = timeout


def _stub_sdk(
    sandbox: _Sandbox | None = None,
    *,
    create_error: Exception | None = None,
    connect_error: Exception | None = None,
) -> Any:
    made = sandbox or _Sandbox()

    class _AsyncSandbox:
        @staticmethod
        async def create(**kwargs: Any) -> _Sandbox:
            made.created_with = kwargs  # type: ignore[attr-defined]
            if create_error is not None:
                raise create_error
            return made

        @staticmethod
        async def connect(sandbox_id: str, **kwargs: Any) -> _Sandbox:
            made.connected_with = (sandbox_id, kwargs)  # type: ignore[attr-defined]
            if connect_error is not None:
                raise connect_error
            return made

    class _Sdk:
        AsyncSandbox = _AsyncSandbox
        AuthenticationException = _AuthenticationException
        FileNotFoundException = _NotFoundException
        FileType = _FileType
        NotFoundException = _NotFoundException
        RateLimitException = _RateLimitException
        SandboxException = _SandboxException
        ServiceBusyException = _ServiceBusyException
        TimeoutException = _TimeoutException

    _Sdk.made = made  # type: ignore[attr-defined]
    return _Sdk


def _adapter(monkeypatch: pytest.MonkeyPatch, sdk: Any) -> e2b_adapter.E2BCodeExecutionAdapter:
    monkeypatch.setattr(e2b_adapter, "_sdk", lambda: sdk)
    return e2b_adapter.E2BCodeExecutionAdapter()


@pytest.mark.asyncio
async def test_a_session_is_one_sandbox_and_is_killed_on_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = _stub_sdk()
    async with _adapter(monkeypatch, sdk).open_session(timeout_s=42, session_ttl_s=400) as session:
        assert session.session_id == "sbx_test"
    assert sdk.made.killed
    # E2B's ``timeout`` is how long it keeps the sandbox, which has to be the
    # whole lease and never one execute's budget: a request runs code once per
    # tool-loop round, and the sandbox has to outlast all of them.
    assert sdk.made.created_with["timeout"] == 400


@pytest.mark.asyncio
async def test_a_pinned_container_image_is_ignored_rather_than_sent_as_a_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _stub_sdk()
    adapter = _adapter(monkeypatch, sdk)
    async with adapter.open_session(image="mzdotai/otari-sandbox-container:latest", timeout_s=30, session_ttl_s=120):
        pass
    # sandbox_session_image is a container image reference, the published
    # protocol's vocabulary; E2B names templates of its own, so sending it
    # through would fail every creation.
    assert "template" not in sdk.made.created_with


@pytest.mark.asyncio
async def test_stdout_stderr_and_a_bare_expression_become_one_result_block(monkeypatch: pytest.MonkeyPatch) -> None:
    # Jupyter carries the value of a trailing expression as a result rather
    # than on stdout, and a terminal would have shown it.
    execution = _Execution(stdout=["counting\n"], stderr=["a warning\n"], results=[_Result("42")])
    sdk = _stub_sdk(_Sandbox(execution))
    async with _adapter(monkeypatch, sdk).open_session(timeout_s=30, session_ttl_s=120) as session:
        block = await session.execute("6 * 7", timeout_s=30)
    assert block.content.stdout == "counting\n42\n"
    assert block.content.stderr == "a warning\n"
    assert block.content.return_code == 0
    # Which files a run produced is worked out above the port, by diffing.
    assert block.content.content == []


@pytest.mark.asyncio
async def test_a_program_that_raises_is_a_successful_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    execution = _Execution(error=_Error("ValueError", "boom", "Traceback...\nValueError: boom\n"))
    sdk = _stub_sdk(_Sandbox(execution))
    async with _adapter(monkeypatch, sdk).open_session(timeout_s=30, session_ttl_s=120) as session:
        block = await session.execute("raise ValueError('boom')", timeout_s=30)
    assert block.content.return_code == 1
    assert "ValueError: boom" in block.content.stderr


@pytest.mark.asyncio
async def test_a_timeout_is_reported_in_the_block_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = _stub_sdk(_Sandbox(_TimeoutException()))
    async with _adapter(monkeypatch, sdk).open_session(timeout_s=5, session_ttl_s=20) as session:
        block = await session.execute("while True: pass", timeout_s=5)
    assert block.content.return_code == 124
    assert "timed out" in block.content.stderr


@pytest.mark.asyncio
async def test_capacity_and_credential_failures_map_to_the_ports_two_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    busy = _adapter(monkeypatch, _stub_sdk(create_error=_ServiceBusyException("full")))
    with pytest.raises(SandboxUnavailableError) as caught:
        async with busy.open_session(timeout_s=30, session_ttl_s=120):
            pass
    assert caught.value.retry_after == "5"

    rejected = _adapter(monkeypatch, _stub_sdk(create_error=_AuthenticationException("bad key")))
    with pytest.raises(SandboxNotReachableError) as refused:
        async with rejected.open_session(timeout_s=30, session_ttl_s=120):
            pass
    # The deployment's own credential: the operator reads the log, the caller does not.
    assert "bad key" not in str(refused.value)


@pytest.mark.asyncio
async def test_the_workspace_listing_is_relative_and_leaves_directories_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = {posixpath.join(ROOT, "chart.png"): b"\x89PNG", posixpath.join(ROOT, "data.csv"): b"a,b\n"}
    sandbox = _Sandbox(tree=tree)
    when = datetime(2026, 9, 21, tzinfo=UTC)
    walked: list[int] = []

    def _list(path: str, depth: int = 1) -> Any:
        walked.append(depth)
        return _listing(tree, when)

    sandbox.files.list = _list  # type: ignore[method-assign]
    sdk = _stub_sdk(sandbox)
    async with _adapter(monkeypatch, sdk).open_session(timeout_s=30, session_ttl_s=120) as session:
        listed = await session.list_files()
    # Relative, because that is what the model names and what GetFile takes,
    # and no directory, because nothing above the port can fetch one.
    assert sorted((e.path, e.size_bytes) for e in listed) == [("chart.png", 4), ("data.csv", 4)]
    assert all(e.modified_at == when.timestamp() for e in listed)
    # Deeper than one level, so a run that writes into a subdirectory still has
    # its output collected.
    assert walked == [e2b_adapter._LIST_DEPTH]


async def _listing(tree: dict[str, bytes], when: datetime) -> list[_Entry]:
    entries: list[_Entry] = [_Entry(p, len(b), modified=when) for p, b in tree.items()]
    entries.append(_Entry(posixpath.join(ROOT, "outdir"), 0, is_dir=True))
    return entries


@pytest.mark.asyncio
async def test_a_produced_file_comes_back_and_respects_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = _Sandbox(tree={posixpath.join(ROOT, "chart.png"): b"\x89PNG bytes"})
    sdk = _stub_sdk(sandbox)
    async with _adapter(monkeypatch, sdk).open_session(timeout_s=30, session_ttl_s=120) as session:
        chunks = [chunk async for chunk in session.read_file("chart.png", budget_bytes=1000)]
        assert b"".join(chunks) == b"\x89PNG bytes"

        # Refused from the listing, before the SDK is asked for the bytes: it
        # hands a file over whole, so measuring after the read would mean
        # holding all of whatever the run decided to write.
        reads: list[str] = []
        original_read = sandbox.files.read

        async def _recorded(path: str, format: str = "bytes") -> bytes:  # noqa: A002 - the SDK's own name
            reads.append(path)
            return await original_read(path, format)

        sandbox.files.read = _recorded  # type: ignore[method-assign]
        with pytest.raises(OutputOverBudget):
            async for _ in session.read_file("chart.png", budget_bytes=2):
                pass
        assert reads == []

        with pytest.raises(SandboxNotReachableError):
            async for _ in session.read_file("missing.png", budget_bytes=1000):
                pass


@pytest.mark.asyncio
async def test_a_file_whose_size_the_provider_will_not_report_is_not_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """The SDK hands a file over whole, so an unknown size is refused, not read on trust.

    Reading it to then measure it is the thing the size check exists to avoid:
    what a run writes is untrusted and can be arbitrarily large.
    """
    sandbox = _Sandbox(tree={posixpath.join(ROOT, "chart.png"): b"\x89PNG bytes"})
    sdk = _stub_sdk(sandbox)
    reads: list[str] = []

    async def _unlistable(path: str, depth: int = 1) -> list[_Entry]:
        del path, depth
        raise RuntimeError("listing is unavailable")

    async def _recorded(path: str, format: str = "bytes") -> bytes:  # noqa: A002 - the SDK's own name
        reads.append(path)
        return b""

    async with _adapter(monkeypatch, sdk).open_session(timeout_s=30, session_ttl_s=120) as session:
        sandbox.files.list = _unlistable  # type: ignore[method-assign]
        sandbox.files.read = _recorded  # type: ignore[method-assign]
        with pytest.raises(SandboxNotReachableError):
            async for _ in session.read_file("chart.png", budget_bytes=1000):
                pass
    assert reads == []


@pytest.mark.asyncio
async def test_a_path_that_leaves_the_workspace_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = _stub_sdk()
    async with _adapter(monkeypatch, sdk).open_session(timeout_s=30, session_ttl_s=120) as session:
        with pytest.raises(ValueError, match="escapes"):
            await session.put_file("../../etc/passwd", b"x", mime_type="text/plain")


@pytest.mark.asyncio
async def test_an_upload_lands_under_the_workspace_root(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = _stub_sdk()
    async with _adapter(monkeypatch, sdk).open_session(timeout_s=30, session_ttl_s=120) as session:
        await session.put_file("data.csv", b"a,b\n", mime_type="text/csv")
    assert sdk.made.files.written == [(posixpath.join(ROOT, "data.csv"), b"a,b\n")]


def test_the_deployment_picks_its_adapter_and_e2b_needs_no_url() -> None:
    protocol = build_code_execution_port(GatewayConfig(sandbox_url="http://sandbox:8080"))
    assert isinstance(protocol, ProtocolCodeExecutionAdapter)
    assert protocol.label == "http://sandbox:8080"

    hosted = GatewayConfig(sandbox_provider="e2b")
    assert build_code_execution_port(hosted).label == "e2b"
    # No URL, and code execution is still configured: that is the whole point.
    assert hosted.sandbox_configured()

    with pytest.raises(ValueError, match="sandbox_url is required"):
        build_code_execution_port(GatewayConfig())


def test_an_unknown_provider_is_refused_at_config_load() -> None:
    with pytest.raises(ValueError, match="sandbox_provider must be one of"):
        GatewayConfig(sandbox_provider="not-a-provider")


def test_a_provider_that_cannot_be_reached_is_refused_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hosted provider is advertised on the strength of the setting alone, so
    a missing extra or credential has to fail the boot rather than every request."""
    hosted = GatewayConfig(sandbox_provider="e2b")

    def _no_sdk() -> Any:
        raise SandboxNotReachableError("sandbox_provider 'e2b' requires the E2B SDK")

    monkeypatch.setattr(e2b_adapter, "_sdk", _no_sdk)
    with pytest.raises(ValueError, match="requires the E2B SDK"):
        verify_code_execution_ready(hosted)

    monkeypatch.setattr(e2b_adapter, "_sdk", lambda: _stub_sdk())
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    with pytest.raises(ValueError, match="E2B_API_KEY"):
        verify_code_execution_ready(hosted)

    monkeypatch.setenv("E2B_API_KEY", "e2b_test_key")
    verify_code_execution_ready(hosted)


def test_startup_says_nothing_about_a_deployment_that_configured_no_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Most deployments run no sandbox at all, and none of them should fail to boot."""
    monkeypatch.delenv("OTARI_SANDBOX_URL", raising=False)
    verify_code_execution_ready(GatewayConfig())
    verify_code_execution_ready(GatewayConfig(sandbox_url="http://sandbox:8080"))


def _fake_sdk_modules() -> tuple[types.ModuleType, types.ModuleType]:
    """Stand-ins for the two real modules, so `_sdk` itself can be run."""
    e2b = types.ModuleType("e2b")
    for name in (
        "AuthenticationException",
        "FileNotFoundException",
        "NotFoundException",
        "RateLimitException",
        "SandboxException",
        "ServiceBusyException",
        "TimeoutException",
    ):
        setattr(e2b, name, type(name, (Exception,), {}))
    e2b.FileType = _FileType  # type: ignore[attr-defined]
    code_interpreter = types.ModuleType("e2b_code_interpreter")
    code_interpreter.AsyncSandbox = _Sandbox  # type: ignore[attr-defined]
    return e2b, code_interpreter


def test_the_sdk_namespace_is_built_from_the_real_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every other test here hands the adapter a stub in place of `_sdk`, so this
    is the one that runs `_sdk` itself: what it returns is only ever exercised by
    a deployment that booted with the extra installed."""
    e2b, code_interpreter = _fake_sdk_modules()
    monkeypatch.setitem(sys.modules, "e2b", e2b)
    monkeypatch.setitem(sys.modules, "e2b_code_interpreter", code_interpreter)

    sdk = e2b_adapter._sdk()

    assert sdk.AsyncSandbox is code_interpreter.AsyncSandbox
    assert sdk.FileType is e2b.FileType
    for name in ("AuthenticationException", "SandboxException", "TimeoutException"):
        assert getattr(sdk, name) is getattr(e2b, name)


def test_a_missing_sdk_names_the_extra_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    # None in sys.modules is what makes an import of a present module fail.
    monkeypatch.setitem(sys.modules, "e2b", None)
    with pytest.raises(SandboxNotReachableError, match=r"pip install otari\[e2b\]"):
        e2b_adapter._sdk()


@pytest.mark.asyncio
async def test_a_kept_sandbox_is_held_for_the_idle_timeout_instead_of_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reuse across requests: the sandbox outlives the block, on E2B's own timer."""
    sdk = _stub_sdk()
    async with _adapter(monkeypatch, sdk).open_session(timeout_s=30, session_ttl_s=300, keep_alive_s=600):
        pass
    assert not sdk.made.killed
    assert sdk.made.held_for == 600


@pytest.mark.asyncio
async def test_a_resumed_sandbox_is_connected_by_id_and_not_created(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = _stub_sdk()
    async with _adapter(monkeypatch, sdk).open_session(timeout_s=30, session_ttl_s=300, resume="sbx_test") as session:
        assert session.session_id == "sbx_test"
    sandbox_id, kwargs = sdk.made.connected_with
    assert sandbox_id == "sbx_test"
    # This request's whole lease, so the sandbox outlasts every call it makes.
    assert kwargs["timeout"] == 300
    assert not hasattr(sdk.made, "created_with")


@pytest.mark.asyncio
async def test_resuming_a_sandbox_e2b_no_longer_has_is_gone_not_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The caller's id is stale, which is its 400 to fix, not a backend outage."""
    sdk = _stub_sdk(connect_error=_NotFoundException("no such sandbox"))
    with pytest.raises(SandboxSessionGoneError):
        async with _adapter(monkeypatch, sdk).open_session(timeout_s=30, session_ttl_s=300, resume="sbx_gone"):
            pass
