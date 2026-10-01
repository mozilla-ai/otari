import uuid
from dataclasses import dataclass
from typing import Never

from sqlalchemy import select
from sqlmodel import col

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.usage import UsageLog
from gateway.repositories.base_repository import BaseRepository


@dataclass(frozen=True)
class RoutedRequest:
    """How one served request was routed, as its settling usage row recorded it.

    Both fields are ``None`` for a request no learning router backend decided.
    """

    routing_backend: str | None
    routing_decision_id: str | None


class RoutedRequestRepository(BaseRepository[UsageLog, Never, Never]):
    """Read requests back by the id their caller received, in the open block of a Unit of Work."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, UsageLog)

    async def find(self, request_id: str, workspace_id: uuid.UUID) -> RoutedRequest | None:
        """Return the request ``request_id`` names in this workspace, or None when it has no row there.

        A request can write several rows (an absorbed attempt before the one that
        served), and every one carries the routing decision; the settling row is
        read, newest first, so an absorbed row never answers.
        """
        result = await self.db.execute(
            select(UsageLog.routing_backend, UsageLog.routing_decision_id)
            .where(
                UsageLog.request_id == request_id,
                UsageLog.workspace_id == workspace_id,
                UsageLog.status != "absorbed",
            )
            .order_by(col(UsageLog.timestamp).desc())
            .limit(1)
        )
        row = result.one_or_none()
        if row is None:
            return None
        backend, decision_id = row
        return RoutedRequest(routing_backend=backend, routing_decision_id=decision_id)
