"""Null Object growth-signal adapter, for a build with no outside vendor.

Satisfies :class:`gateway.ports.growth_signal_port.GrowthSignalPort` with the
honest answer for a deployment that runs its own users on its own channels:
nothing is scheduled, so no HTTP call, background task, or vendor credential is
ever reached for want of a growth or support integration.
"""

import uuid
from datetime import datetime

from fastapi import BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.ports.growth_signal_port import GrowthActivationEvent


class NullGrowthSignalAdapter:
    """``GrowthSignalPort`` adapter for a build with no CRM or messenger vendor.

    Holds no state, so the request's database session is unused.
    """

    def __init__(self, session: AsyncSession | None) -> None:
        # Accepted to match the container's per-request factory; unused, because
        # there is nothing to schedule.
        del session

    async def record_signup(
        self,
        *,
        background_tasks: BackgroundTasks,
        user_id: uuid.UUID,
        email: str,
        full_name: str | None,
        created_at: datetime,
    ) -> None:
        return None

    async def record_activation(
        self,
        *,
        background_tasks: BackgroundTasks,
        event: GrowthActivationEvent,
        user_id: uuid.UUID,
        email: str,
    ) -> None:
        return None

    async def record_profile_updated(
        self,
        *,
        background_tasks: BackgroundTasks,
        user_id: uuid.UUID,
        email: str,
        full_name: str | None,
    ) -> None:
        return None

    async def record_account_deleted(
        self,
        *,
        background_tasks: BackgroundTasks,
        user_id: uuid.UUID,
        email: str,
    ) -> None:
        return None
