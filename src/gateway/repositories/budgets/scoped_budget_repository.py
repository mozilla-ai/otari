import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Never

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError
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


def _scope_match(scope_type: str, scope_id: str, provider_key_id: str | None) -> ColumnElement[bool]:
    """Match a ceiling on the scope and provider that the two partial unique indexes key on."""
    return (
        (ScopedBudget.scope_type == scope_type) & (ScopedBudget.scope_id == scope_id) & _for_provider(provider_key_id)
    )


class ScopedBudgetRepository(BaseRepository[ScopedBudget, Never, Never]):
    """Query and stage spend ceilings in the open block of a Unit of Work."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, ScopedBudget)

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

    async def has_ceiling(self, scope_type: ScopeType, scope_id: str, provider_key_id: str | None) -> bool:
        """Report whether a ceiling caps this scope for this provider, where None means every provider."""
        return await self._exists(_scope_match(scope_type, scope_id, provider_key_id))

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

    async def list_in_scopes(self, scopes: ScopeIdSets, *, skip: int, limit: int) -> list[tuple[ScopedBudget, Budget]]:
        """Return a page of the ceilings on these scopes, each with the budget it names, oldest first."""
        result = await self.db.execute(
            select(ScopedBudget, Budget)
            .join(Budget, Budget.budget_id == ScopedBudget.budget_id)
            .where(_in_scopes(scopes))
            .order_by(ScopedBudget.created_at, ScopedBudget.id)
            .offset(skip)
            .limit(limit)
        )
        return list(result.tuples().all())

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

    async def _exists(self, match: ColumnElement[bool]) -> bool:
        """Report whether any ceiling matches this predicate."""
        result = await self.db.execute(select(func.count()).select_from(ScopedBudget).where(match))
        return result.scalar_one() > 0

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
