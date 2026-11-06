"""Unit tests for the Monty adapter, against the real interpreter the dev group installs."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from gateway.adapters import monty_code_execution_adapter
from gateway.adapters.code_execution_adapter import ProtocolCodeExecutionAdapter, build_code_execution_port
from gateway.adapters.monty_code_execution_adapter import MontyCodeExecutionAdapter
from gateway.core.config import GatewayConfig
from gateway.models.tools import SandboxProvider
from gateway.ports.code_execution_port import (
    OutputOverBudget,
    SandboxNotReachableError,
    SandboxSessionGoneError,
    SandboxUnavailableError,
)
from gateway.services.sandbox_backend import SandboxBackend


def _adapter() -> MontyCodeExecutionAdapter:
    return MontyCodeExecutionAdapter(max_memory_bytes=64 * 1024 * 1024, write_bytes_limit=1024 * 1024)


def test_with_no_sandbox_url_the_monty_extra_is_the_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    assert not GatewayConfig().sandbox_configured()

    monkeypatch.setattr("gateway.core.config.monty_installed", lambda: True)
    default = GatewayConfig()
    assert default.sandbox_configured()
    assert build_code_execution_port(default).label == "monty"

    # A URL names a backend the operator runs, and an explicit provider is an operator's choice.
    assert isinstance(
        build_code_execution_port(GatewayConfig(sandbox_url="http://sandbox:8080")), ProtocolCodeExecutionAdapter
    )
    assert not GatewayConfig(sandbox_provider=SandboxProvider.PROTOCOL).sandbox_configured()


@pytest.mark.asyncio
async def test_a_session_keeps_its_state_and_files_across_runs() -> None:
    async with _adapter().open_session(timeout_s=10, session_ttl_s=60) as session:
        await session.put_file("data.csv", b"a,b\n1,2\n", mime_type="text/csv")

        first = await session.execute(
            "from pathlib import Path\nrows = Path('data.csv').read_text().splitlines()\nprint(len(rows))",
            timeout_s=10,
        )
        second = await session.execute("Path('out.txt').write_text(rows[1])", timeout_s=10)

        assert (first.content.stdout, first.content.return_code) == ("2\n", 0)
        assert second.content.return_code == 0
        assert {entry.path for entry in await session.list_files()} == {"data.csv", "out.txt"}
        assert b"".join([chunk async for chunk in session.read_file("out.txt", budget_bytes=100)]) == b"1,2"


@pytest.mark.asyncio
async def test_code_that_fails_is_reported_as_a_program_would_report_it() -> None:
    adapter = MontyCodeExecutionAdapter(max_memory_bytes=64 * 1024 * 1024, write_bytes_limit=1024)
    async with adapter.open_session(timeout_s=1, session_ttl_s=60) as session:
        raised = await session.execute("1 / 0", timeout_s=1)
        unsupported = await session.execute("import numpy", timeout_s=1)
        endless = await session.execute("while True:\n    pass", timeout_s=1)

    assert raised.content.return_code == 1
    assert "ZeroDivisionError" in raised.content.stderr
    # Refused before it runs, with a reason the model can act on.
    assert unsupported.content.return_code == 1
    assert "numpy" in unsupported.content.stderr
    assert endless.content.return_code == 124


@pytest.mark.asyncio
async def test_a_session_is_never_resumed_and_reads_only_what_it_holds() -> None:
    with pytest.raises(SandboxSessionGoneError):
        async with _adapter().open_session(timeout_s=10, session_ttl_s=60, resume="otari-monty-old"):
            pass

    # A pinned image is a container's, so Monty runs the code without it.
    async with _adapter().open_session(image="python:3.13", timeout_s=10, session_ttl_s=60) as session:
        assert not session.holds_across_requests
        await session.put_file("big.txt", b"0123456789", mime_type="text/plain")
        with pytest.raises(ValueError, match="escapes"):
            await session.put_file("../outside.txt", b"x", mime_type="text/plain")
        with pytest.raises(OutputOverBudget):
            async for _ in session.read_file("big.txt", budget_bytes=2):
                pass
        with pytest.raises(SandboxNotReachableError):
            async for _ in session.read_file("missing.txt", budget_bytes=100):
                pass


@pytest.mark.asyncio
async def test_a_full_pool_asks_the_caller_to_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FullPool:
        @asynccontextmanager
        async def checkout(self, **_: object) -> AsyncIterator[None]:
            raise TimeoutError
            yield

    monkeypatch.setattr(monty_code_execution_adapter, "_pool", _FullPool())
    with pytest.raises(SandboxUnavailableError) as caught:
        async with _adapter().open_session(timeout_s=10, session_ttl_s=60):
            pass
    assert caught.value.retry_after == "5"


def test_a_missing_monty_names_the_extra_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    # None in sys.modules is what makes an import of a present module fail.
    monkeypatch.setitem(sys.modules, "pydantic_monty", None)
    with pytest.raises(SandboxNotReachableError, match=r"uv sync --extra monty"):
        monty_code_execution_adapter.verify_ready()


def test_the_model_is_told_it_runs_a_python_subset() -> None:
    backend = SandboxBackend(port=_adapter())
    [(_, hint)] = backend.purpose_hints()
    [tool] = backend.openai_tools

    assert "standard library only" in hint
    assert "numpy" not in hint
    assert "standard library only" in tool["function"]["description"]
