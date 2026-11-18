"""Data access for the one attempt to connect this deployment to otari.ai. Flushes, never commits."""

import uuid
from datetime import datetime
from typing import Any, Never, cast

from sqlalchemy import CursorResult, delete, update

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.providers import (
    PLATFORM_CONNECT_PENDING,
    PLATFORM_CONNECT_ROW_ID,
    PlatformConnectAttempt,
    PlatformConnectState,
)
from gateway.repositories.base_repository import BaseRepository


class PlatformConnectRepository(BaseRepository[PlatformConnectAttempt, Never, Never]):
    """Repository for the ``platform_connect_attempts`` row."""

    def __init__(self, uow: UnitOfWork):
        super().__init__(uow, PlatformConnectAttempt)

    async def current(self) -> PlatformConnectAttempt | None:
        """The attempt as the database holds it now, or None where there is none."""
        return await self.db.get(PlatformConnectAttempt, PLATFORM_CONNECT_ROW_ID, populate_existing=True)

    async def replace(self, **fields: Any) -> PlatformConnectAttempt:
        """Put a new attempt in place of whatever the row held."""
        await self.db.execute(delete(PlatformConnectAttempt).execution_options(synchronize_session=False))
        self.db.expire_all()
        row = PlatformConnectAttempt(id=PLATFORM_CONNECT_ROW_ID, **fields)
        self.db.add(row)
        await self.db.flush()
        return row

    async def claim_poll(self, attempt_id: uuid.UUID, now: datetime, next_poll_at: datetime) -> bool:
        """Move ``next_poll_at`` forward if the pending attempt is due a poll. Whether this caller may poll.

        One statement, so of two status reads at once only one asks otari.ai.
        """
        return await self._update_pending(
            attempt_id, PlatformConnectAttempt.next_poll_at <= now, next_poll_at=next_poll_at
        )

    async def slow_down(self, attempt_id: uuid.UUID, interval_seconds: int, next_poll_at: datetime) -> bool:
        """Poll the pending attempt less often from now on."""
        return await self._update_pending(attempt_id, interval_seconds=interval_seconds, next_poll_at=next_poll_at)

    async def finish(self, attempt_id: uuid.UUID, status: PlatformConnectState) -> bool:
        """End the pending attempt as ``status`` and drop its device code. Whether it was still pending."""
        return await self._update_pending(attempt_id, status=status, encrypted_device_code=None)

    async def clear(self) -> None:
        """Forget the attempt, pending or ended."""
        await self.db.execute(delete(PlatformConnectAttempt).execution_options(synchronize_session=False))

    async def _update_pending(self, attempt_id: uuid.UUID, *conditions: Any, **values: Any) -> bool:
        result = cast(
            CursorResult[Any],
            await self.db.execute(
                update(PlatformConnectAttempt)
                .where(
                    PlatformConnectAttempt.id == PLATFORM_CONNECT_ROW_ID,
                    PlatformConnectAttempt.attempt_id == attempt_id,
                    PlatformConnectAttempt.status == PLATFORM_CONNECT_PENDING,
                    *conditions,
                )
                .values(**values)
                .execution_options(synchronize_session=False)
            ),
        )
        return result.rowcount == 1
