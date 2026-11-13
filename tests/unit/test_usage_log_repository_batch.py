from typing import cast
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.usage import UsageLog
from gateway.repositories.usage_and_telemetry import UsageLogRepository


@pytest.mark.asyncio
async def test_add_many_stages_usage_rows_and_flushes_once() -> None:
    session = AsyncMock(spec=AsyncSession)
    repository = UsageLogRepository(cast(AsyncSession, session))
    rows = [UsageLog(id="batch-row", model="test-model", endpoint="/v1/batches/results", status="success")]

    await repository.add_many(rows)

    session.add_all.assert_called_once_with(rows)
    session.flush.assert_awaited_once()
