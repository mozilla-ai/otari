"""Unit tests for the Monty adapter, against the real interpreter the dev group installs."""

from __future__ import annotations

import pytest

from gateway.adapters.code_execution_adapter import ProtocolCodeExecutionAdapter, build_code_execution_port
from gateway.adapters.monty_code_execution_adapter import MontyCodeExecutionAdapter
from gateway.core.config import GatewayConfig
from gateway.models.tools import SandboxProvider
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


def test_the_model_is_told_it_runs_a_python_subset() -> None:
    backend = SandboxBackend(port=_adapter())
    [(_, hint)] = backend.purpose_hints()
    [tool] = backend.openai_tools

    assert "standard library only" in hint
    assert "numpy" not in hint
    assert "standard library only" in tool["function"]["description"]
