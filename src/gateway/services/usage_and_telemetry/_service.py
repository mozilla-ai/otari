"""The usage domain's reads, each in a block of its own on the Unit of Work it was built with."""

import uuid
from collections.abc import Collection
from datetime import datetime

from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.usage_and_telemetry import ServedTraffic, UsageLogRepository


class UsageService:
    """Usage rows, read inside a tenant scope the caller derived."""

    def __init__(self, uow: UnitOfWork, usage_logs: UsageLogRepository) -> None:
        self._uow = uow
        self._usage_logs = usage_logs

    async def served_traffic(self, *, workspace_ids: Collection[uuid.UUID], since: datetime) -> list[ServedTraffic]:
        """The requests served here since ``since`` in these workspaces, by name sent and offering reached."""
        async with self._uow:
            return await self._usage_logs.served_traffic(workspace_ids=workspace_ids, since=since)
