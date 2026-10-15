import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Never, cast

from sqlalchemy import and_, case, delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped
from sqlalchemy.sql.elements import ColumnElement

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.budget_exceptions import SpendCeilingAlreadyExistsError
from gateway.models.budgets import (
    SCOPE_API_TOKEN,
    SCOPE_ORG_MEMBER,
    SCOPE_ORGANIZATION,
    SCOPE_WORKSPACE,
    SCOPE_WORKSPACE_MEMBER,
    Budget,
    ScopedBudget,
    ScopeType,
)
from gateway.repositories.base_repository import BaseRepository

# The counters are ``NUMERIC(18, 6)``, so a bare ``0.0`` in a CASE arm or a ``.values()`` would make PostgreSQL
# resolve the expression as double precision and hand a binary-rounded amount back to an exact column.
_ZERO = Decimal(0)


@dataclass(frozen=True)
class ApplicableCeiling:
    """A ceiling on one of a request's scopes, with its provider, its budget's caps and when its window ends."""

    scoped_budget_id: str
    scope_type: ScopeType
    provider_key_id: str | None
    budget_duration_sec: int | None
    reset_alignment: str | None
    period_end: datetime | None
    token_limit: int | None
    request_limit: int | None


@dataclass(frozen=True)
class CeilingUsage:
    """A ceiling's cap, its recorded use and its holds, on each axis, and when its window ends."""

    max_budget: Decimal | None
    current_spend: Decimal
    reserved_spend: Decimal
    token_limit: int | None
    current_tokens: int
    reserved_tokens: int
    request_limit: int | None
    current_requests: int
    reserved_requests: int
    period_end: datetime | None


@dataclass(frozen=True)
class ScopeIdSets:
    """The IDs of the scopes of each kind that a query is about, as the strings a ceiling stores."""

    organization_ids: tuple[str, ...]
    workspace_ids: tuple[str, ...]
    organization_member_ids: tuple[str, ...]
    workspace_member_ids: tuple[str, ...]
    api_key_ids: tuple[str, ...]


def _in_scopes(scopes: ScopeIdSets) -> ColumnElement[bool]:
    """Match a ceiling whose scope ID is in the set for its own scope kind, so an ID never matches across kinds."""
    return or_(
        (ScopedBudget.scope_type == SCOPE_ORGANIZATION) & ScopedBudget.scope_id.in_(scopes.organization_ids),
        (ScopedBudget.scope_type == SCOPE_WORKSPACE) & ScopedBudget.scope_id.in_(scopes.workspace_ids),
        (ScopedBudget.scope_type == SCOPE_ORG_MEMBER) & ScopedBudget.scope_id.in_(scopes.organization_member_ids),
        (ScopedBudget.scope_type == SCOPE_WORKSPACE_MEMBER) & ScopedBudget.scope_id.in_(scopes.workspace_member_ids),
        (ScopedBudget.scope_type == SCOPE_API_TOKEN) & ScopedBudget.scope_id.in_(scopes.api_key_ids),
    )


def _for_provider(provider_key_id: str | None) -> ColumnElement[bool]:
    """Match a ceiling's provider, where None matches the ceiling that caps every provider."""
    if provider_key_id is None:
        return ScopedBudget.provider_key_id.is_(None)
    return ScopedBudget.provider_key_id == provider_key_id


def _member_scope_ids(member_ids: Sequence[uuid.UUID]) -> list[str]:
    return [str(member_id) for member_id in member_ids]


def _release_count_expression(column: Mapped[int], amount: int) -> object:
    """Subtract ``amount`` from a token or request hold, clamped at zero."""
    return case((column - amount < 0, 0), else_=column - amount)


def _release_expression(amount: Decimal) -> object:
    """Subtract ``amount`` from the money hold, clamped at zero.

    CASE rather than GREATEST, matching the per-user release, because SQLite has no GREATEST.
    Both arms are ``Decimal`` so the CASE resolves as ``numeric``.
    """
    return case(
        (ScopedBudget.reserved_spend - amount < _ZERO, _ZERO),
        else_=ScopedBudget.reserved_spend - amount,
    )


def _scope_match(scope_type: str, scope_id: str, provider_key_id: str | None) -> ColumnElement[bool]:
    """Match a ceiling on the scope and provider that the two partial unique indexes key on."""
    return (
        (ScopedBudget.scope_type == scope_type) & (ScopedBudget.scope_id == scope_id) & _for_provider(provider_key_id)
    )


class ScopedBudgetRepository(BaseRepository[ScopedBudget, Never, Never]):
    """Query and stage spend ceilings in the open block of a Unit of Work."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, ScopedBudget)

    async def add(self, ceiling: ScopedBudget) -> ScopedBudget:
        """Stage a new ceiling and return it with its generated values.

        The budget the ceiling names must exist, because every refusal of the insert is reported as a duplicate.

        Raises:
            SpendCeilingAlreadyExistsError: a ceiling already caps the same scope for the same provider.
        """
        self.db.add(ceiling)
        try:
            await self.db.flush()
        except IntegrityError:
            raise SpendCeilingAlreadyExistsError(ceiling.scope_type, ceiling.scope_id) from None
        await self.db.refresh(ceiling)
        return ceiling

    async def count_for_budget(self, budget_id: str) -> int:
        """Count the ceilings that name this budget."""
        result = await self.db.execute(
            select(func.count()).select_from(ScopedBudget).where(ScopedBudget.budget_id == budget_id)
        )
        return result.scalar_one()

    async def count_for_budgets(self, budget_ids: Sequence[str]) -> dict[str, int]:
        """Count the ceilings naming each of these budgets, omitting a budget that none names."""
        result = await self.db.execute(
            select(ScopedBudget.budget_id, func.count())
            .where(ScopedBudget.budget_id.in_(budget_ids))
            .group_by(ScopedBudget.budget_id)
        )
        return dict(result.tuples().all())

    async def count_in_scopes(self, scopes: ScopeIdSets) -> int:
        """Count the ceilings on these scopes."""
        result = await self.db.execute(select(func.count()).select_from(ScopedBudget).where(_in_scopes(scopes)))
        return result.scalar_one()

    async def delete_for_member(self, member_id: uuid.UUID) -> None:
        """Delete every ceiling keyed on this membership.

        ``scope_id`` is not a foreign key, so nothing cascades, and a ceiling left
        behind would refuse its budget's deletion.
        """
        await self.db.execute(
            delete(ScopedBudget).where(
                ScopedBudget.scope_type == SCOPE_WORKSPACE_MEMBER,
                ScopedBudget.scope_id == str(member_id),
            )
        )

    async def delete_for_workspace(self, workspace_id: uuid.UUID, member_ids: Sequence[uuid.UUID]) -> None:
        """Delete every ceiling keyed on this workspace or on one of these memberships.

        Precondition: ``member_ids`` is every membership of the workspace whatever its status, and the
        workspace's row lock is held.
        An invited or suspended membership owns a ceiling too, and one left behind is unreachable and
        refuses its budget's deletion for good.
        """
        await self.db.execute(
            delete(ScopedBudget)
            .where(
                or_(
                    (ScopedBudget.scope_type == SCOPE_WORKSPACE) & (ScopedBudget.scope_id == str(workspace_id)),
                    (ScopedBudget.scope_type == SCOPE_WORKSPACE_MEMBER)
                    & ScopedBudget.scope_id.in_(_member_scope_ids(member_ids)),
                )
            )
            .execution_options(synchronize_session=False)
        )

    async def get_usage(self, scoped_budget_id: str) -> CeilingUsage | None:
        """Return this ceiling's cap, recorded use, holds and window end, or None when it no longer exists."""
        row = (
            await self.db.execute(
                select(
                    Budget.max_budget,
                    ScopedBudget.current_spend,
                    ScopedBudget.reserved_spend,
                    Budget.token_limit,
                    ScopedBudget.current_tokens,
                    ScopedBudget.reserved_tokens,
                    Budget.request_limit,
                    ScopedBudget.current_requests,
                    ScopedBudget.reserved_requests,
                    ScopedBudget.period_end,
                )
                .join(Budget, Budget.budget_id == ScopedBudget.budget_id)
                .where(ScopedBudget.id == scoped_budget_id)
            )
        ).first()
        return None if row is None else CeilingUsage(*row)

    async def has_ceiling(self, scope_type: ScopeType, scope_id: str, provider_key_id: str | None) -> bool:
        """Report whether a ceiling caps this scope for this provider, where None means every provider."""
        return await self._exists(_scope_match(scope_type, scope_id, provider_key_id))

    async def hold(
        self,
        scoped_budget_id: str,
        amount: Decimal | None,
        *,
        tokens: int,
        requests: int,
        new_request: bool,
    ) -> bool:
        """Hold a request on one ceiling when every capped axis has room, and report whether it was held.

        ``new_request=False`` grows a request the ceiling has already admitted, and asks only whether the delta fits.
        ``amount=None`` neither holds nor gates the dollar axis.
        """

        # Each cap is a column on the budget this ceiling names, read as a correlated subquery so the whole check
        # stays inside the one conditional UPDATE. Reading them first and comparing in Python would reintroduce
        # the read-then-write race. Three subqueries over one row rather than a join, because a join here would
        # have to be an UPDATE ... FROM, which SQLite does not take.
        def cap_of(column: Mapped[Decimal | None] | Mapped[int | None]) -> Any:
            return select(column).where(Budget.budget_id == ScopedBudget.budget_id).scalar_subquery()

        def admits(committed: Any, held: Any, cap: Any) -> ColumnElement[bool]:
            """Whether one axis has room, or has no cap at all.

            An arrival is refused when the axis is already at or over its cap, which catches a request estimating
            nothing against an exhausted ceiling, and when this hold would push it past.
            A top-up is asked only the second.
            """
            fits: ColumnElement[bool] = committed + held <= cap
            if new_request:
                fits = and_(committed < cap, fits)
            return or_(cap.is_(None), fits)

        # An axis whose amount is None is not part of this request and is not asked.
        gates = [
            admits(committed, held, cap)
            for committed, held, cap in (
                (ScopedBudget.current_spend + ScopedBudget.reserved_spend, amount, cap_of(Budget.max_budget)),
                (ScopedBudget.current_tokens + ScopedBudget.reserved_tokens, tokens, cap_of(Budget.token_limit)),
                (
                    ScopedBudget.current_requests + ScopedBudget.reserved_requests,
                    requests,
                    cap_of(Budget.request_limit),
                ),
            )
            if held is not None
        ]
        result = await self.db.execute(
            update(ScopedBudget)
            .where(ScopedBudget.id == scoped_budget_id, *gates)
            .values(
                reserved_spend=ScopedBudget.reserved_spend + (amount if amount is not None else _ZERO),
                reserved_tokens=ScopedBudget.reserved_tokens + tokens,
                reserved_requests=ScopedBudget.reserved_requests + requests,
            )
            .execution_options(synchronize_session=False)
        )
        return bool(getattr(result, "rowcount", 0))

    async def insert_member_ceilings(self, ceilings: Sequence[ScopedBudget]) -> list[ScopedBudget]:
        """Stage these membership ceilings, skipping any the database already caps, and return those staged.

        A ceiling another writer placed first is skipped, because nothing locks the gap between deciding
        which memberships need one and inserting them.
        Any other refusal fails the step rather than being reported as a ceiling already in place.
        The two are told apart by re-reading the scope, which sees the other writer's row only under
        read-committed isolation, because the engines word the refusal differently.
        """
        if not ceilings:
            return []
        # A savepoint flushes what is already pending as it opens, so each attempt adds
        # its ceilings inside the block rather than before it.
        try:
            async with self.db.begin_nested():
                self.db.add_all(ceilings)
                await self.db.flush()
            return list(ceilings)
        except IntegrityError:
            pass

        staged: list[ScopedBudget] = []
        for ceiling in ceilings:
            # A failed flush expires the row, so the scope is read before it.
            collision = _scope_match(ceiling.scope_type, ceiling.scope_id, ceiling.provider_key_id)
            try:
                async with self.db.begin_nested():
                    self.db.add(ceiling)
                    await self.db.flush()
            except IntegrityError:
                if not await self._exists(collision):
                    raise
                continue
            staged.append(ceiling)
        return staged

    async def list_applicable(self, scopes: ScopeIdSets) -> list[ApplicableCeiling]:
        """Return the ceilings on these scopes, whichever provider each caps."""
        result = await self.db.execute(
            select(
                ScopedBudget.id,
                ScopedBudget.scope_type,
                ScopedBudget.provider_key_id,
                Budget.budget_duration_sec,
                Budget.reset_alignment,
                ScopedBudget.period_end,
                Budget.token_limit,
                Budget.request_limit,
            )
            .join(Budget, Budget.budget_id == ScopedBudget.budget_id)
            .where(_in_scopes(scopes))
        )
        # NOTE: The scope predicate matches every row on a known scope type, which is what makes the cast sound.
        return [
            ApplicableCeiling(
                scoped_budget_id=scoped_budget_id,
                scope_type=cast(ScopeType, scope_type),
                provider_key_id=provider_key_id,
                budget_duration_sec=budget_duration_sec,
                reset_alignment=reset_alignment,
                period_end=period_end,
                token_limit=token_limit,
                request_limit=request_limit,
            )
            for (
                scoped_budget_id,
                scope_type,
                provider_key_id,
                budget_duration_sec,
                reset_alignment,
                period_end,
                token_limit,
                request_limit,
            ) in result.tuples().all()
        ]

    async def list_in_scopes(
        self,
        scopes: ScopeIdSets,
        *,
        skip: int,
        limit: int,
        scope_type: ScopeType | None = None,
        scope_id: str | None = None,
    ) -> list[tuple[ScopedBudget, Budget]]:
        """Return a page of the ceilings on these scopes, each with the budget it names, oldest first.

        ``scope_type`` and ``scope_id`` narrow the page to one scope kind or one scope.
        """
        statement = (
            select(ScopedBudget, Budget)
            .join(Budget, Budget.budget_id == ScopedBudget.budget_id)
            .where(_in_scopes(scopes))
        )
        if scope_type is not None:
            statement = statement.where(ScopedBudget.scope_type == scope_type)
        if scope_id is not None:
            statement = statement.where(ScopedBudget.scope_id == scope_id)
        result = await self.db.execute(
            statement.order_by(ScopedBudget.created_at, ScopedBudget.id).offset(skip).limit(limit)
        )
        return list(result.tuples().all())

    async def get_in_scopes(self, scoped_budget_id: str, scopes: ScopeIdSets) -> ScopedBudget | None:
        """Return the ceiling with this ID when it is on one of these scopes, otherwise None."""
        result = await self.db.execute(
            select(ScopedBudget).where(ScopedBudget.id == scoped_budget_id, _in_scopes(scopes))
        )
        return result.scalar_one_or_none()

    async def lock(self, scoped_budget_ids: Sequence[str]) -> None:
        """Lock these ceilings until the transaction ends, in the order of their IDs."""
        await self.db.execute(
            select(ScopedBudget.id)
            .where(ScopedBudget.id.in_(list(scoped_budget_ids)))
            .order_by(ScopedBudget.id)
            .with_for_update()
        )

    async def member_ceiling(self, member_id: uuid.UUID, provider_key_id: str | None) -> ScopedBudget | None:
        """Return the ceiling capping this membership for this provider, where None means every provider."""
        result = await self.db.execute(
            select(ScopedBudget).where(_scope_match(SCOPE_WORKSPACE_MEMBER, str(member_id), provider_key_id))
        )
        return result.scalars().first()

    async def members_with_ceiling(
        self, member_ids: Sequence[uuid.UUID], provider_key_id: str | None
    ) -> set[uuid.UUID]:
        """Return the IDs of the memberships among these that a ceiling already caps for this provider."""
        result = await self.db.execute(
            select(ScopedBudget.scope_id).where(
                ScopedBudget.scope_type == SCOPE_WORKSPACE_MEMBER,
                ScopedBudget.scope_id.in_(_member_scope_ids(member_ids)),
                _for_provider(provider_key_id),
            )
        )
        return {uuid.UUID(scope_id) for scope_id in result.scalars().all()}

    async def release_holds(
        self, scoped_budget_ids: Sequence[str], amount: Decimal, *, tokens: int, requests: int
    ) -> None:
        """Give a held amount back to each of these ceilings, on every axis with a positive amount, clamped at zero."""
        values: dict[str, object] = {}
        if amount > _ZERO:
            values["reserved_spend"] = _release_expression(amount)
        if tokens > 0:
            values["reserved_tokens"] = _release_count_expression(ScopedBudget.reserved_tokens, tokens)
        if requests > 0:
            values["reserved_requests"] = _release_count_expression(ScopedBudget.reserved_requests, requests)
        await self._update_counters(scoped_budget_ids, values)

    async def remove(self, ceiling: ScopedBudget) -> None:
        """Stage the deletion of a ceiling."""
        await self.db.delete(ceiling)
        await self.db.flush()

    async def retime_for_budget(
        self, budget_id: str, *, period_start: datetime | None, period_end: datetime | None
    ) -> None:
        """Set this window on every ceiling naming the budget, leaving each ceiling's counters as they are."""
        await self.db.execute(
            update(ScopedBudget)
            .where(ScopedBudget.budget_id == budget_id)
            .values(period_start=period_start, period_end=period_end)
            .execution_options(synchronize_session=False)
        )

    async def roll_period(
        self, scoped_budget_id: str, *, now: datetime, period_start: datetime, period_end: datetime | None
    ) -> None:
        """Start this window on a ceiling whose window ended by ``now``, zeroing its recorded use and not its holds.

        The UPDATE is guarded on ``period_end``, so of several concurrent callers one rolls the window and the others
        change nothing.
        """
        await self.db.execute(
            update(ScopedBudget)
            .where(
                ScopedBudget.id == scoped_budget_id,
                ScopedBudget.period_end.is_not(None),
                ScopedBudget.period_end <= now,
            )
            .values(
                current_spend=_ZERO,
                current_tokens=0,
                current_requests=0,
                period_start=period_start,
                period_end=period_end,
            )
            .execution_options(synchronize_session=False)
        )

    async def settle(
        self,
        scoped_budget_ids: Sequence[str],
        *,
        spend: Decimal,
        held: Decimal,
        tokens: int,
        held_tokens: int,
        requests: int,
        held_requests: int,
    ) -> None:
        """Record use and release holds on each of these ceilings, on every axis with a positive amount."""
        values: dict[str, object] = {}
        if spend > _ZERO:
            values["current_spend"] = ScopedBudget.current_spend + spend
        if held > _ZERO:
            values["reserved_spend"] = _release_expression(held)
        if tokens > 0:
            values["current_tokens"] = ScopedBudget.current_tokens + tokens
        if held_tokens > 0:
            values["reserved_tokens"] = _release_count_expression(ScopedBudget.reserved_tokens, held_tokens)
        if requests > 0:
            values["current_requests"] = ScopedBudget.current_requests + requests
        if held_requests > 0:
            values["reserved_requests"] = _release_count_expression(ScopedBudget.reserved_requests, held_requests)
        await self._update_counters(scoped_budget_ids, values)

    async def scope_ceiling(self, scope_type: ScopeType, scope_id: str) -> ScopedBudget | None:
        """Return the ceiling that caps this scope across every provider, or None."""
        result = await self.db.execute(select(ScopedBudget).where(_scope_match(scope_type, scope_id, None)))
        return result.scalars().first()

    async def add_scope_ceiling_if_absent(
        self,
        *,
        scope_type: ScopeType,
        scope_id: str,
        budget_id: str,
        period_start: datetime | None,
        period_end: datetime | None,
    ) -> bool:
        """Stage a ceiling capping this scope across every provider, returning False when one already does.

        The insert runs in a SAVEPOINT, so losing that race to a concurrent writer rolls back this row alone.
        """
        try:
            async with self.db.begin_nested():
                self.db.add(
                    ScopedBudget(
                        scope_type=scope_type,
                        scope_id=scope_id,
                        budget_id=budget_id,
                        period_start=period_start,
                        period_end=period_end,
                    )
                )
        except IntegrityError:
            return False
        return True

    async def _exists(self, match: ColumnElement[bool]) -> bool:
        """Report whether any ceiling matches this predicate."""
        result = await self.db.execute(select(func.count()).select_from(ScopedBudget).where(match))
        return result.scalar_one() > 0

    async def _update_counters(self, scoped_budget_ids: Sequence[str], values: dict[str, object]) -> None:
        if not scoped_budget_ids or not values:
            return
        await self.db.execute(
            update(ScopedBudget)
            .where(ScopedBudget.id.in_(list(scoped_budget_ids)))
            .values(**values)
            .execution_options(synchronize_session=False)
        )
