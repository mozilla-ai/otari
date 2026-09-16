import uuid
from collections.abc import Sequence

from sqlalchemy import String, and_, cast, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement
from sqlmodel import col

from gateway.models.entities import APIKey, UsageLog, User
from gateway.models.tenancy import OrganizationMember, Workspace

# The owner a key falls back to when it is created without a user_id (the API's
# convenience path, and the first-run bootstrap key). One shared, visible,
# budgetable user rather than a throwaway per key, so nothing is untracked and the
# operator can cap all such keys with a single budget on this user.
DEFAULT_USER_ID = "default"


async def get_active_user(db: AsyncSession, user_id: str, *, for_update: bool = False) -> User | None:
    """Query for a non-deleted user by user_id."""

    stmt = select(User).where(User.user_id == user_id, User.deleted_at.is_(None))
    if for_update:
        stmt = stmt.with_for_update()
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def _revive(user: User) -> User:
    """Clear a soft delete so the shared default owner is usable again."""
    if user.deleted_at is not None:
        user.deleted_at = None
    return user


async def get_or_create_default_user(db: AsyncSession) -> User:
    """Return the shared ``default`` user, creating (or reviving) it if needed.

    The caller still owns the final commit. The insert goes through a SAVEPOINT so
    that losing a race to a concurrent creator rolls back only this row, not
    whatever the caller has already staged: ``user_id`` is the primary key, so
    without this the loser's commit would raise and surface as a 500 for a request
    that should simply have reused the row the winner just created.
    """
    existing = (await db.execute(select(User).where(User.user_id == DEFAULT_USER_ID))).scalar_one_or_none()
    if existing is not None:
        return await _revive(existing)

    user = User(user_id=DEFAULT_USER_ID, alias="Default")
    try:
        async with db.begin_nested():
            db.add(user)
        return user
    except IntegrityError:
        # Someone else inserted it between our select and our flush; adopt theirs.
        winner = (await db.execute(select(User).where(User.user_id == DEFAULT_USER_ID))).scalar_one_or_none()
        if winner is None:
            raise
        return await _revive(winner)


async def get_or_create_attribution_user(db: AsyncSession, *, user_id: str, alias: str | None) -> User:
    """Return the request-plane owner a tenancy identity bills through.

    Keys, budgets, and usage rows hang off ``users.user_id`` (a string); tenancy
    members are ``user.id`` (a UUID). Nothing joins the two, so a member with no
    row here cannot own a key. This mints that row, keyed on the identity's UUID
    rendered as a string, which is what makes it idempotent: re-adding a member
    finds the existing row instead of minting a second one.

    A soft-deleted row is revived rather than skipped. Leaving it deleted would
    leave a member the roster lists but ``POST /v1/keys`` refuses, and the
    counters are deliberately untouched: ``spend`` and ``budget_id`` carry over,
    so removing and re-adding someone cannot be used to clear their spend against
    a budget.

    The caller owns the commit. The insert goes through a SAVEPOINT for the same
    reason :func:`get_or_create_default_user` does: a lost race rolls back this
    row alone, not whatever the caller has already staged.
    """
    existing = (await db.execute(select(User).where(User.user_id == user_id))).scalar_one_or_none()
    if existing is not None:
        return await _revive(existing)

    user = User(user_id=user_id, alias=alias)
    try:
        async with db.begin_nested():
            db.add(user)
        return user
    except IntegrityError:
        winner = (await db.execute(select(User).where(User.user_id == user_id))).scalar_one_or_none()
        if winner is None:
            raise
        return await _revive(winner)


async def live_attribution_user_ids(db: AsyncSession, user_ids: Sequence[str]) -> set[str]:
    """Return which of ``user_ids`` name a usable request-plane owner.

    One query for the whole roster rather than a lookup per member. A row that is
    absent or soft-deleted is left out, so the caller reports ``None`` rather than
    an id that ``POST /v1/keys`` would reject.
    """
    if not user_ids:
        return set()
    rows = await db.execute(select(User.user_id).where(User.user_id.in_(list(user_ids)), User.deleted_at.is_(None)))
    return set(rows.scalars().all())


def _keyed_in(organization_id: uuid.UUID | None) -> ColumnElement[bool]:
    """Whether this user owns an API key in ``organization_id``, or in any."""
    condition = select(APIKey.user_id).where(APIKey.user_id == User.user_id)
    if organization_id is not None:
        condition = condition.join(Workspace, col(Workspace.id) == APIKey.workspace_id).where(
            col(Workspace.organization_id) == organization_id
        )
    return condition.exists()


def _spent_in(organization_id: uuid.UUID | None) -> ColumnElement[bool]:
    """Whether this user has usage in ``organization_id``, or in any.

    Separate from :func:`_keyed_in` because a key can be revoked: an owner whose
    keys are all gone still has the spend the budgets page is there to show, and
    dropping them would hide a ledger rather than a credential.
    """
    condition = select(UsageLog.user_id).where(UsageLog.user_id == User.user_id)
    if organization_id is not None:
        condition = condition.join(Workspace, col(Workspace.id) == UsageLog.workspace_id).where(
            col(Workspace.organization_id) == organization_id
        )
    return condition.exists()


def _on_roster_of(organization_id: uuid.UUID | None) -> ColumnElement[bool]:
    """Whether this user is the attribution row of a member of ``organization_id``.

    ``organization_member`` keys on the UUID identity while ``users`` keys on a
    string, and the attribution row is that UUID rendered as one (see
    :func:`get_or_create_attribution_user`), so the join is a cast rather than a
    foreign key. This is what puts a member who has not minted a key yet in
    their own organization's scope.
    """
    condition = select(col(OrganizationMember.id)).where(
        cast(col(OrganizationMember.user_id), String) == User.user_id,
    )
    if organization_id is not None:
        condition = condition.where(col(OrganizationMember.organization_id) == organization_id)
    return condition.exists()


def in_organization(organization_id: uuid.UUID) -> ColumnElement[bool]:
    """Restrict a ``users`` query to the spend identities one organization can name.

    ``users`` is deployment-global and carries no organization of its own, which
    on a deployment serving mutually-untrusting tenants made every operator read
    here a read of every tenant's people (otari-ai#2108). The column that would
    make this a lookup is otari-ai#1727's to add, and membership is many-to-many
    besides: one identity's attribution row serves every organization that
    person belongs to, so a single column could not hold the answer anyway.

    So the scope is derived from the three joins that do exist, any of which puts
    a user in reach: a key, usage, or a roster row.

    A user reached by none of the three, in any organization at all, is shared
    rather than hidden. That is the shared ``default`` owner by construction, and
    a user an operator has just created and not yet keyed, and hiding those would
    take a freshly created user out of the page that assigns it a budget. The
    first key written for one binds it, because the key is itself a join.
    """
    return or_(
        _keyed_in(organization_id),
        _spent_in(organization_id),
        _on_roster_of(organization_id),
        and_(~_keyed_in(None), ~_spent_in(None), ~_on_roster_of(None)),
    )


async def owned_by_organization(db: AsyncSession, user_id: str, organization_id: uuid.UUID) -> bool:
    """Whether ``organization_id`` may name ``user_id`` as an owner.

    The write-side half of :func:`in_organization`: scoping the read alone stops
    a picker offering another organization's people without stopping a client
    naming one.
    """
    found = await db.execute(select(User.user_id).where(User.user_id == user_id, in_organization(organization_id)))
    return found.scalar_one_or_none() is not None
