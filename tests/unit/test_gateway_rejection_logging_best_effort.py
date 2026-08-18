"""Writing a rejection row must never change the response the caller gets.

Every gateway-side rejection site logs the drop and then re-raises the refusal
it was already going to return (see ``log_gateway_rejection``). If the log write
escaped as an exception, an unhealthy log writer would turn a clean 403 or 400
into a 500 and look to the client like a broken gateway. These pin that the
write is best-effort in both directions: a failure is swallowed, and a healthy
writer still gets its row.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.api.routes._pipeline import log_gateway_rejection
from gateway.models.entities import UsageLog


class _BoomWriter:
    """A log writer whose put fails the way a broken session or queue would."""

    async def put(self, log: UsageLog) -> None:
        raise RuntimeError("log writer is down")

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class _RecordingWriter:
    def __init__(self) -> None:
        self.rows: list[UsageLog] = []

    async def put(self, log: UsageLog) -> None:
        self.rows.append(log)

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


def _kwargs(log_writer: Any) -> dict[str, Any]:
    # Touched once even for a rejection: every usage row carries a workspace, and
    # the key names it. The scalar resolves to None, which is the "key not found"
    # arm and falls back to the default workspace, so the row is still built.
    db = MagicMock()
    db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None)))
    return {
        "db": db,
        "log_writer": log_writer,
        "api_key_id": "key-1",
        "user_id": "user-1",
        "model": "gpt-4o",
        "provider": "openai",
        "endpoint": "/v1/chat/completions",
        "detail": "User 'user-1' has exceeded budget limit",
        # The status this rejection returns, recorded so the failure taxonomy can
        # tell a refusal from a provider fault (see UsageLog.status_code).
        "status_code": 403,
        "started_at": None,
    }


@pytest.mark.asyncio
async def test_writer_failure_does_not_propagate() -> None:
    """A failing writer must not escape and mask the caller's rejection."""
    await log_gateway_rejection(**_kwargs(_BoomWriter()))


@pytest.mark.asyncio
async def test_healthy_writer_still_records_the_rejection() -> None:
    """The control: swallowing failures must not mean swallowing everything."""
    writer = _RecordingWriter()
    await log_gateway_rejection(**_kwargs(writer))

    assert len(writer.rows) == 1
    row = writer.rows[0]
    assert row.status == "error"
    assert row.cost is None
    assert row.counts_toward_budget is True
    assert row.user_id == "user-1"
    # Without this the row classifies as "unknown" in errors_by_status_code,
    # indistinguishable from a row written before the column existed.
    assert row.status_code == 403
