"""The timings the catalog reads an offering's speed from."""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.usage import UsageLog
from gateway.repositories.usage_and_telemetry import UsageLogRepository
from gateway.services.workspace_scope import default_workspace_id

_OFFERING = ("nebius:glm", "nebius", "glm")


@pytest.mark.asyncio
async def test_both_spellings_of_an_offering_share_its_cap_and_unmeasurable_rows_take_no_place(
    async_db: AsyncSession,
) -> None:
    workspace_id = await default_workspace_id(async_db)
    now = datetime.now(UTC)

    def served(minutes_ago: int, **overrides: Any) -> UsageLog:
        fields: dict[str, Any] = {
            "workspace_id": workspace_id,
            "timestamp": now - timedelta(minutes=minutes_ago),
            "model": "glm",
            "provider": "nebius",
            "endpoint": "/v1/chat/completions",
            "status": "success",
            "completion_tokens": 100,
            "latency_ms": 1000,
        }
        fields.update(overrides)
        return UsageLog(**fields)

    async_db.add_all(
        [
            # Newest: a rate with no duration to divide by, which must not take a place.
            *(served(0, latency_ms=0) for _ in range(3)),
            served(1),
            served(2, provider=None, model="nebius:glm"),
            served(3),
            served(4, provider=None, model="nebius:glm"),
            # A named instance whose model happens to spell the selector.
            served(0, provider="home_lab", model="nebius:glm"),
        ]
    )
    await async_db.commit()

    uow = UnitOfWork(async_db)
    async with uow:
        samples = await UsageLogRepository(uow).speed_samples(
            workspace_ids=[workspace_id], offerings=[_OFFERING], since=now - timedelta(days=1), per_offering=3
        )

    assert len(samples) == 3
    assert all(sample.latency_ms == 1000 for sample in samples)
    assert sorted(sample.model for sample in samples) == ["glm", "glm", "nebius:glm"]
