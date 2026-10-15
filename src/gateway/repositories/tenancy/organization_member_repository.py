"""Data access for organization memberships."""

import uuid
from collections.abc import Iterable
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.budgets import Budget, ScopedBudget
from gateway.models.tenancy import (
    Organization,
    OrganizationMember,
    OrganizationMemberCreate,
    OrganizationMemberUpdate,
    User,
    Workspace,
    WorkspaceMember,
)
from gateway.repositories.base_repository import BaseRepository
from gateway.repositories.tenancy.user_repository import user_alphabetical_order, user_matches_search

# Statuses a membership can hold and still belong on the roster. Removal
# suspends rather than deletes, so "suspended" is the one that drops off.
LISTABLE_STATUSES = ("active", "invited")


class OrganizationMemberRepository(
    BaseRepository[OrganizationMember, OrganizationMemberCreate, OrganizationMemberUpdate]
):
    """Repository for organization membership rows."""

    def __init__(self, db: AsyncSession | UnitOfWork):
        super().__init__(db, OrganizationMember)

    async def get_by_organization_and_user(
        self,
        organization_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> OrganizationMember | None:
        """Return the membership joining a user to an organization, whatever its status."""
        result = await self.db.execute(
            select(OrganizationMember).where(
                col(OrganizationMember.organization_id) == organization_id,
                col(OrganizationMember.user_id) == user_id,
            )
        )
        return result.scalars().first()

    async def get_active_by_organization_and_user(
        self,
        organization_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> OrganizationMember | None:
        """Return the active membership joining a user to an organization."""
        result = await self.db.execute(
            select(OrganizationMember).where(
                col(OrganizationMember.organization_id) == organization_id,
                col(OrganizationMember.user_id) == user_id,
                col(OrganizationMember.status) == "active",
            )
        )
        return result.scalars().first()

    async def get_by_id_and_organization(
        self,
        organization_member_id: uuid.UUID,
        organization_id: uuid.UUID,
    ) -> OrganizationMember | None:
        """Return a membership by id, scoped to one organization.

        The scoping is the cross-tenant boundary: a caller passing another
        organization's member id resolves to nothing rather than reading or
        mutating that organization's row.
        """
        result = await self.db.execute(
            select(OrganizationMember).where(
                col(OrganizationMember.id) == organization_member_id,
                col(OrganizationMember.organization_id) == organization_id,
            )
        )
        return result.scalars().first()

    async def get_by_id_and_user(
        self,
        organization_member_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> OrganizationMember | None:
        """Return a membership by id, scoped to the identity that holds it.

        ``get_by_id_and_organization``'s counterpart for the surfaces a caller
        reaches as the subject of a membership rather than as an administrator
        of the organization holding it (the invitee's own inbox). The scoping is
        the boundary either way: another identity's member id resolves to
        nothing rather than being read and then checked in Python.
        """
        result = await self.db.execute(
            select(OrganizationMember).where(
                col(OrganizationMember.id) == organization_member_id,
                col(OrganizationMember.user_id) == user_id,
            )
        )
        return result.scalars().first()

    async def get_by_user(self, user_id: uuid.UUID, *, active_only: bool = False) -> list[OrganizationMember]:
        """Return a user's memberships, oldest first."""
        statement = select(OrganizationMember).where(col(OrganizationMember.user_id) == user_id)
        if active_only:
            statement = statement.where(col(OrganizationMember.status) == "active")
        result = await self.db.execute(statement.order_by(col(OrganizationMember.created_at)))
        return list(result.scalars().all())

    async def get_first_active_for_user(self, user_id: uuid.UUID, *, lock: bool = False) -> OrganizationMember | None:
        """Return a user's oldest active membership, or None.

        The id breaks a tie, for the same reason ``get_active_owner`` does it:
        two memberships written in one transaction share a ``created_at``, and
        this decides which organization a stale pointer is repaired onto, so it
        must not be able to answer differently on two reads.

        ``lock`` holds that membership until the caller's
        transaction ends, for a write that is allowed only while one exists:
        a removal suspends the row, so it waits, and one that committed first
        fails the re-checked ``active`` condition.

        NOTE: The lock applies on PostgreSQL only, because SQLite has no row locks.
        """
        statement = (
            select(OrganizationMember)
            .where(
                col(OrganizationMember.user_id) == user_id,
                col(OrganizationMember.status) == "active",
            )
            .order_by(col(OrganizationMember.created_at), col(OrganizationMember.id))
            .limit(1)
        )
        if lock:
            statement = statement.with_for_update()
        result = await self.db.execute(statement)
        return result.scalars().first()

    async def get_organization_id(self, organization_member_id: uuid.UUID) -> uuid.UUID | None:
        """Return the ID of the organization a membership belongs to, or None."""
        result = await self.db.execute(
            select(col(OrganizationMember.organization_id)).where(col(OrganizationMember.id) == organization_member_id)
        )
        return result.scalar_one_or_none()

    async def get_ids_by_organization(self, organization_id: uuid.UUID) -> list[uuid.UUID]:
        """Return the ID of every membership in an organization, whatever its status."""
        result = await self.db.execute(
            select(col(OrganizationMember.id)).where(col(OrganizationMember.organization_id) == organization_id)
        )
        return list(result.scalars().all())

    async def create_membership(
        self,
        *,
        organization_id: uuid.UUID,
        user_id: uuid.UUID,
        role: str,
        status: str = "active",
    ) -> OrganizationMember:
        """Stage a new membership."""
        member = OrganizationMember(
            organization_id=organization_id,
            user_id=user_id,
            role=role,
            status=status,
        )
        self.db.add(member)
        await self.db.flush()
        await self.db.refresh(member)
        return member

    async def add_if_absent(
        self, *, organization_id: uuid.UUID, user_id: uuid.UUID, role: str
    ) -> tuple[OrganizationMember, bool]:
        """Stage an active membership unless one exists, and say whether this call created it.

        A concurrent insert of the same membership is settled in a savepoint, so the loser's conflict
        does not end the caller's transaction.
        """
        try:
            async with self.db.begin_nested():
                return await self.create_membership(organization_id=organization_id, user_id=user_id, role=role), True
        except IntegrityError:
            existing = await self.get_by_organization_and_user(organization_id, user_id)
            if existing is None:
                raise
            return existing, False

    async def update_membership(
        self,
        member: OrganizationMember,
        update_data: dict[str, Any],
    ) -> OrganizationMember:
        """Stage an update from a plain mapping of columns."""
        member.sqlmodel_update(update_data)
        self.db.add(member)
        await self.db.flush()
        await self.db.refresh(member)
        return member

    async def count_active_owners(self, organization_id: uuid.UUID) -> int:
        """Count an organization's active owners.

        The last-owner guard reads this: an organization with no owner left has
        nobody who can delete it or manage its members.
        """
        result = await self.db.execute(
            select(func.count())
            .select_from(OrganizationMember)
            .where(
                col(OrganizationMember.organization_id) == organization_id,
                col(OrganizationMember.role) == "owner",
                col(OrganizationMember.status) == "active",
            )
        )
        return result.scalar_one()

    async def get_active_owner(self, organization_id: uuid.UUID) -> OrganizationMember | None:
        """Return an organization's earliest active owner, if any.

        Ordered by ``created_at`` then ``id`` so the answer is stable across
        calls even when two owner rows share a timestamp.
        """
        result = await self.db.execute(
            select(OrganizationMember)
            .where(
                col(OrganizationMember.organization_id) == organization_id,
                col(OrganizationMember.role) == "owner",
                col(OrganizationMember.status) == "active",
            )
            .order_by(col(OrganizationMember.created_at), col(OrganizationMember.id))
        )
        return result.scalars().first()

    async def get_by_organization_with_users(
        self,
        organization_id: uuid.UUID,
        *,
        skip: int = 0,
        limit: int = 100,
        search: str | None = None,
    ) -> tuple[list[tuple[OrganizationMember, User]], int]:
        """Return a page of the roster as ``(membership, identity)`` pairs, plus the total.

        One join rather than a lookup per row, so a roster of N members costs
        one query instead of N+1.

        ``search`` narrows on the name and the email, case-insensitively, and the
        count narrows with it: a picker that searched the page it had fetched
        offered a subset of the roster and said nothing about it (otari#1380), so
        the match has to happen where every row is.
        """
        filters = [
            col(OrganizationMember.organization_id) == organization_id,
            col(OrganizationMember.status).in_(LISTABLE_STATUSES),
        ]
        if term := (search or "").strip():
            filters.append(user_matches_search(term))

        count_result = await self.db.execute(
            select(func.count())
            .select_from(OrganizationMember)
            .join(User, col(OrganizationMember.user_id) == col(User.id))
            .where(*filters)
        )
        count = count_result.scalar_one()

        result = await self.db.execute(
            select(OrganizationMember, User)
            .join(User, col(OrganizationMember.user_id) == col(User.id))
            .where(*filters)
            .order_by(user_alphabetical_order(), col(OrganizationMember.id))
            .offset(skip)
            .limit(limit)
        )
        return [(member, user) for member, user in result.all()], count

    async def get_by_users_with_organizations(
        self,
        user_ids: Iterable[uuid.UUID],
    ) -> dict[uuid.UUID, list[tuple[OrganizationMember, Organization]]]:
        """Group a page of identities' memberships, joined to their organizations, by user.

        One query for the whole page rather than a lookup per row, the same
        reason ``get_by_organization_with_users`` joins: the deployment-wide user
        list shows every organization each identity belongs to, so N rows would
        otherwise cost N queries.

        Every status, unlike ``LISTABLE_STATUSES`` above: an operator looking at
        a stuck account needs to see that its one membership is suspended, which
        is the state the roster deliberately hides. An identity with no
        membership is absent from the mapping rather than present with an empty
        list, so the caller's ``.get(user_id, [])`` is the only place that
        decides what "none" renders as.

        Sorted the way ``get_by_user_with_organizations`` sorts one identity's,
        ``created_at`` then ``id``, so the organizations under a row do not
        reorder between reads.
        """
        ids = list(user_ids)
        if not ids:
            return {}

        result = await self.db.execute(
            select(OrganizationMember, Organization)
            .join(Organization, col(OrganizationMember.organization_id) == col(Organization.id))
            .where(col(OrganizationMember.user_id).in_(ids))
            .order_by(col(OrganizationMember.created_at), col(OrganizationMember.id))
        )
        grouped: dict[uuid.UUID, list[tuple[OrganizationMember, Organization]]] = {}
        for membership, organization in result.all():
            grouped.setdefault(membership.user_id, []).append((membership, organization))
        return grouped

    async def get_by_user_with_organizations(
        self,
        user_id: uuid.UUID,
        *,
        status: str | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[list[tuple[OrganizationMember, Organization]], int]:
        """Return a page of a user's memberships joined to their organizations, plus the total.

        One join rather than a lookup per row, the same shape as
        ``get_by_organization_with_users``: this is what an organization
        switcher renders, so every row needs its organization's name.

        ``id`` breaks the ``created_at`` tie for the reason
        ``get_first_active_for_user`` gives: two memberships written in one
        transaction share a timestamp, and a page whose order is not total
        can return one row twice and another never.
        """
        conditions = [col(OrganizationMember.user_id) == user_id]
        if status is not None:
            conditions.append(col(OrganizationMember.status) == status)

        count_result = await self.db.execute(select(func.count()).select_from(OrganizationMember).where(*conditions))
        count = count_result.scalar_one()

        result = await self.db.execute(
            select(OrganizationMember, Organization)
            .join(Organization, col(OrganizationMember.organization_id) == col(Organization.id))
            .where(*conditions)
            .order_by(col(OrganizationMember.created_at), col(OrganizationMember.id))
            .offset(skip)
            .limit(limit)
        )
        return [(member, organization) for member, organization in result.all()], count

    async def placements_for_users(
        self,
        organization_id: uuid.UUID,
        user_ids: Iterable[uuid.UUID],
    ) -> dict[uuid.UUID, list[tuple[WorkspaceMember, Workspace]]]:
        """Which workspaces each of these identities belongs to, with the workspace.

        Two bounded reads for a page of members rather than one per workspace:
        the roster page used to fan out a roster read per workspace and join the
        results in the browser (otari#1381).
        """

        ids = list(user_ids)
        if not ids:
            return {}

        rows = (
            await self.db.execute(
                select(WorkspaceMember, Workspace)
                .join(Workspace, col(WorkspaceMember.workspace_id) == col(Workspace.id))
                .where(
                    col(Workspace.organization_id) == organization_id,
                    col(WorkspaceMember.user_id).in_(ids),
                    # Active only, the way ``get_workspaces_for_user`` answers
                    # the same question: a suspended membership is somebody who
                    # is no longer in that workspace, and listing it would put
                    # them somewhere they cannot act.
                    col(WorkspaceMember.status) == "active",
                )
                .order_by(col(Workspace.name), col(Workspace.id))
            )
        ).all()

        placements: dict[uuid.UUID, list[tuple[WorkspaceMember, Workspace]]] = {}
        for membership, workspace in rows:
            placements.setdefault(membership.user_id, []).append((membership, workspace))
        return placements

    async def ceilings_for_memberships(
        self,
        membership_ids: Iterable[uuid.UUID],
    ) -> dict[str, tuple[ScopedBudget, Budget]]:
        """The spend ceiling on each of these workspace memberships, by membership id.

        The aggregate ceiling only, the one narrowed to no provider: a ceiling
        carrying a ``provider_key_id`` caps one credential rather than the
        membership, and the roster reports what the member may spend at all.
        ``WorkspaceBudgetDefault`` is read the same way for the same reason.

        ``scoped_budgets.scope_id`` is text while a membership id is a UUID, so
        the ids are matched as strings here rather than cast in SQL, which the
        two engines spell differently.
        """

        keys = [str(membership_id) for membership_id in membership_ids]
        if not keys:
            return {}

        rows = (
            await self.db.execute(
                select(ScopedBudget, Budget)
                .join(Budget, ScopedBudget.budget_id == Budget.budget_id)
                .where(
                    ScopedBudget.scope_type == "workspace_member",
                    ScopedBudget.scope_id.in_(keys),
                    ScopedBudget.provider_key_id.is_(None),
                )
            )
        ).all()
        return {ceiling.scope_id: (ceiling, budget) for ceiling, budget in rows}


__all__ = ["LISTABLE_STATUSES", "OrganizationMemberRepository"]
