"""Data access for the marker that names the deployment's bootstrap operator."""

import uuid
from typing import Never

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.platform import BOOTSTRAP_IDENTITY_KEY, RuntimeSetting
from gateway.repositories.base_repository import BaseRepository


class BootstrapMarkerRepository(BaseRepository[RuntimeSetting, Never, Never]):
    """Reads and moves the ``tenancy_bootstrap_user_id`` marker."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, RuntimeSetting)

    async def user_id(self) -> uuid.UUID | None:
        """The identity the marker names, read past the session's identity map, or None without a readable one."""
        marker = (
            await self.db.execute(
                select(RuntimeSetting)
                .where(col(RuntimeSetting.key) == BOOTSTRAP_IDENTITY_KEY)
                .execution_options(populate_existing=True)
            )
        ).scalar_one_or_none()
        if marker is None:
            return None
        try:
            return uuid.UUID(marker.value)
        except ValueError:
            return None

    async def move(self, *, from_user_id: uuid.UUID, to_user_id: uuid.UUID) -> bool:
        """Point the marker at ``to_user_id`` if it still names ``from_user_id``, under a row lock.

        Flush-only. False when the marker names anybody else once the lock is held,
        so two handovers racing from the same identity cannot both land.
        """
        marker = (
            await self.db.execute(
                select(RuntimeSetting)
                .where(col(RuntimeSetting.key) == BOOTSTRAP_IDENTITY_KEY)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
        ).scalar_one_or_none()
        if marker is None or marker.value != str(from_user_id):
            return False
        marker.value = str(to_user_id)
        await self.db.flush()
        return True
