"""Data access for ``usage_logs``, one row per attempt the gateway served or imported. Flushes, never commits."""

import uuid
from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime
from typing import Never

from sqlalchemy import and_, case, func, select

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.usage import SERVED_HERE_SLUG, UsageLog
from gateway.repositories.base_repository import BaseRepository

# The statuses that end a request. ``absorbed`` is an attempt a later one recovered
# from, so counting it would count one request twice.
_REQUEST_STATUSES = ("success", "error")


@dataclass(frozen=True)
class ServedTraffic:
    """The requests this gateway served for one name sent and one offering reached.

    ``model`` is the provider's own id, or a whole selector on a row written before
    ``provider`` was recorded. ``fell_back`` counts the successful requests a
    later attempt served.
    """

    provider: str | None
    model: str
    policy_name: str | None
    requests: int
    fell_back: int


class UsageLogRepository(BaseRepository[UsageLog, Never, Never]):
    """Read usage rows inside a set of workspaces."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, UsageLog)

    async def served_traffic(self, *, workspace_ids: Collection[uuid.UUID], since: datetime) -> list[ServedTraffic]:
        """The requests served here since ``since`` in these workspaces, grouped by name sent and offering reached.

        One grouped read. Imported usage is left out, because it was never served here.
        """
        if not workspace_ids:
            return []
        fell_back = func.sum(case((and_(UsageLog.status == "success", UsageLog.attempt_position > 1), 1), else_=0))
        stmt = (
            select(
                UsageLog.provider,
                UsageLog.model,
                UsageLog.policy_name,
                func.count().label("requests"),
                func.coalesce(fell_back, 0).label("fell_back"),
            )
            .where(
                UsageLog.workspace_id.in_(sorted(workspace_ids)),
                UsageLog.source == SERVED_HERE_SLUG,
                UsageLog.status.in_(_REQUEST_STATUSES),
                UsageLog.timestamp >= since,
            )
            .group_by(UsageLog.provider, UsageLog.model, UsageLog.policy_name)
        )
        return [
            ServedTraffic(
                provider=row.provider,
                model=row.model,
                policy_name=row.policy_name,
                requests=int(row.requests),
                fell_back=int(row.fell_back),
            )
            for row in (await self.db.execute(stmt)).all()
        ]
