"""Data access for the subagent model recommendations the gateway answered.

Built on the Unit of Work. Flushes, never commits.
"""

import uuid
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Never

from sqlalchemy import Select, case, func, select

from gateway.core.sql import BucketGrain, bucket_expr, canonical_bucket, dialect_name, utc_bound
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.routing import SUBAGENT_OUTCOME_ESTIMATED, SUBAGENT_OUTCOME_MEASURED, SubagentDecision
from gateway.repositories.base_repository import BaseRepository

Workspaces = Collection[uuid.UUID] | Select[uuid.UUID]
"""The workspaces a read may cover: their ids, or a subquery selecting them."""


@dataclass(frozen=True)
class DecisionWindow:
    """The decisions a read covers. ``workspaces`` comes from the caller's scope, never from a client filter."""

    workspaces: Workspaces
    start: datetime
    end: datetime
    user_ids: Sequence[str] | None = None


@dataclass(frozen=True)
class DecisionSums:
    """What a group of decisions comes to.

    ``measured`` and ``estimated`` count the decisions whose outcome carries
    both costs, which are the ones ``baseline_cost`` and ``actual_cost`` sum.
    ``router_cost`` sums every decision in the group.
    """

    key: tuple[Any, ...]
    decisions: int
    with_outcome: int
    measured: int
    estimated: int
    baseline_cost: Decimal
    actual_cost: Decimal
    router_cost: Decimal


_PRICED = SubagentDecision.baseline_cost.is_not(None) & SubagentDecision.actual_cost.is_not(None)


def _count(condition: Any) -> Any:
    return func.coalesce(func.sum(case((condition, 1), else_=0)), 0)


def _total(column: Any, condition: Any = None) -> Any:
    return func.coalesce(func.sum(column if condition is None else case((condition, column))), 0)


_SUMS = (
    func.count(),
    _count(SubagentDecision.outcome_at.is_not(None)),
    _count(_PRICED & (SubagentDecision.outcome_kind == SUBAGENT_OUTCOME_MEASURED)),
    _count(_PRICED & (SubagentDecision.outcome_kind == SUBAGENT_OUTCOME_ESTIMATED)),
    _total(SubagentDecision.baseline_cost, _PRICED),
    _total(SubagentDecision.actual_cost, _PRICED),
    _total(SubagentDecision.router_cost),
)
# Largest net saving first: the baseline less what was spent on the subagents and on the router.
_BY_SAVING = (_SUMS[4] - _SUMS[5] - _SUMS[6]).desc()


class SubagentDecisionRepository(BaseRepository[SubagentDecision, Never, Never]):
    """Repository for `subagent_decisions` rows."""

    def __init__(self, uow: UnitOfWork):
        super().__init__(uow, SubagentDecision)

    async def insert(self, **fields: Any) -> SubagentDecision:
        """Add a row from the fields the service chose. Flushed, not committed."""
        row = SubagentDecision(**fields)
        self.db.add(row)
        await self.db.flush()
        return row

    async def get_for_caller(
        self,
        decision_id: uuid.UUID,
        *,
        user_id: str | None,
        workspace_id: uuid.UUID | None,
        api_key_id: str | None = None,
    ) -> SubagentDecision | None:
        """One decision by id, or None. A user, a workspace or a key given narrows it to that caller's own.

        The row is locked until the transaction ends, so a read-then-write on it does not interleave.
        """
        statement = select(SubagentDecision).where(SubagentDecision.id == decision_id).with_for_update()
        if user_id is not None:
            statement = statement.where(SubagentDecision.user_id == user_id)
        if api_key_id is not None:
            statement = statement.where(SubagentDecision.api_key_id == api_key_id)
        if workspace_id is not None:
            statement = statement.where(SubagentDecision.workspace_id == workspace_id)
        return (await self.db.execute(statement)).scalar_one_or_none()

    async def save(self, row: SubagentDecision) -> None:
        """Flush the changes made to ``row``."""
        await self.db.flush()

    async def totals(self, window: DecisionWindow) -> DecisionSums:
        """What every decision in the window comes to."""
        (sums,) = await self._sums(window)
        return sums

    async def by_outcome_kind(self, window: DecisionWindow) -> list[DecisionSums]:
        """The window's decisions, grouped by how much is known of their outcome."""
        return await self._sums(window, SubagentDecision.outcome_kind)

    async def by_agent_type(self, window: DecisionWindow, *, limit: int) -> list[DecisionSums]:
        """The subagent types that saved the most, largest saving first."""
        return await self._sums(window, SubagentDecision.agent_type, limit=limit)

    async def by_swap(self, window: DecisionWindow, *, limit: int) -> list[DecisionSums]:
        """The reported outcomes by baseline model and model run on, largest saving first.

        The baseline is the priced model it was read as, or the harness's own
        name for it where it was read as none.
        """
        baseline = func.coalesce(SubagentDecision.baseline_model_key, SubagentDecision.baseline_model)
        return await self._sums(
            window, baseline, SubagentDecision.ran_on, where=SubagentDecision.outcome_at.is_not(None), limit=limit
        )

    async def series(self, window: DecisionWindow, bucket: BucketGrain) -> dict[str, DecisionSums]:
        """The window's decisions per time bucket, by canonical bucket start. Empty buckets are absent."""
        expr = bucket_expr(dialect_name(self.db), bucket, SubagentDecision.created_at)
        return {canonical_bucket(sums.key[0], bucket): sums for sums in await self._sums(window, expr)}

    async def _sums(
        self, window: DecisionWindow, *groups: Any, where: Any = None, limit: int | None = None
    ) -> list[DecisionSums]:
        """The sums of the window's decisions per distinct value of ``groups``, or one row for none."""
        statement = select(*groups, *_SUMS).where(
            SubagentDecision.workspace_id.in_(_ids(window.workspaces)),
            SubagentDecision.created_at >= utc_bound(window.start),
            SubagentDecision.created_at < utc_bound(window.end),
        )
        if where is not None:
            statement = statement.where(where)
        if window.user_ids is not None:
            statement = statement.where(SubagentDecision.user_id.in_(sorted(window.user_ids)))
        if groups:
            statement = statement.group_by(*groups).order_by(_BY_SAVING, *groups)
        if limit is not None:
            statement = statement.limit(limit)
        width = len(groups)
        return [
            DecisionSums(tuple(row[:width]), *(_number(value) for value in row[width:]))
            for row in (await self.db.execute(statement)).all()
        ]


def _ids(workspaces: Workspaces) -> Any:
    return workspaces if isinstance(workspaces, Select) else sorted(workspaces)


def _number(value: Any) -> Any:
    """A sum as the driver returned it: a count as an int, a money total as a Decimal."""
    return value if isinstance(value, int) else Decimal(str(value))
