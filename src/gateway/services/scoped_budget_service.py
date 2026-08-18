"""Enforcement for ``scoped_budgets``: the tenancy-scoped USD ceilings.

The legacy per-user path in :mod:`gateway.services.budget_service` is unchanged
and still enforced; this is a second mechanism beside it. A request resolves the
scopes it bills to (its workspace, that workspace's organization, the caller's
membership rows, and the API key itself), collects every ceiling attached to one
of them, and must pass all of them.

Each row is an independent ceiling. There is deliberately no check that the
children of a scope sum to less than their parent: the parent ceiling already
bounds the total, so the extra rule would refuse configurations that cannot
overspend.

**No row locks.** ``budget_service.reserve_budget`` is lock-free by design, and
this path stays that way: one conditional UPDATE per ceiling, each committed on
its own, so no lock is held across the next one or across the provider call. The
price is that a partial reservation is possible, and the price of that is
compensation: when a ceiling refuses, the holds already taken are released
before the request is rejected. The ceilings are always visited in one total
order so concurrent reservers converge on the same sequence instead of each
compensating the other's progress.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal, get_args

from sqlalchemy import and_, case, or_, select, update
from sqlalchemy.sql.elements import ColumnElement
from sqlmodel import col

from gateway.models.entities import APIKey, ScopedBudget
from gateway.models.tenancy import OrganizationMember, Workspace, WorkspaceMember
from gateway.services.workspace_scope import resolve_workspace_id

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

SCOPE_ORGANIZATION = "organization"
SCOPE_WORKSPACE = "workspace"
SCOPE_WORKSPACE_MEMBER = "workspace_member"
SCOPE_ORG_MEMBER = "org_member"
SCOPE_API_TOKEN = "api_token"

# The wire vocabulary, and the one place it is written. The route layer annotates
# its request and query models with this rather than restating the five strings:
# a scope the service cannot resolve must not be creatable, and two rosters would
# eventually disagree about which those are. `Literal` takes no variables, so the
# constants above and this list are the one unavoidable repetition; `get_args`
# keeps the tuple form derived rather than typed out a third time.
ScopeType = Literal["organization", "workspace", "workspace_member", "org_member", "api_token"]
SCOPE_TYPES: tuple[ScopeType, ...] = get_args(ScopeType)

# Most specific first, so the ceiling closest to the caller is the one that
# refuses when several are exhausted at once and the reported error is the
# actionable one. It is also the reservation order, which is why it must be
# total: see the module docstring.
_SCOPE_PRECEDENCE: dict[str, int] = {
    SCOPE_WORKSPACE_MEMBER: 0,
    SCOPE_ORG_MEMBER: 1,
    SCOPE_API_TOKEN: 2,
    SCOPE_WORKSPACE: 3,
    SCOPE_ORGANIZATION: 4,
}

# What each scope is called in the refusal a client sees. Both member scopes read
# as "Member"; naming the table would leak the tenancy model for no benefit.
_SCOPE_SUBJECT: dict[str, str] = {
    SCOPE_ORGANIZATION: "Organization",
    SCOPE_WORKSPACE: "Workspace",
    SCOPE_WORKSPACE_MEMBER: "Member",
    SCOPE_ORG_MEMBER: "Member",
    SCOPE_API_TOKEN: "API key",
}


@dataclass(frozen=True)
class BudgetScopeRequest:
    """What a request bills to, as the route knows it before resolution.

    ``api_key`` is the key that authenticated the request, or None for a
    master-key caller, which bills to the deployment's default workspace and has
    no API-key ceiling. ``provider_instance`` is the resolved provider the call
    is about to go to (``openai``, or a named instance), which is what a
    provider-narrowed ceiling matches on.
    """

    api_key: APIKey | None
    provider_instance: str | None = None


@dataclass(frozen=True)
class ApplicableBudget:
    """One ceiling a request must pass, and enough of it to report a refusal."""

    budget_id: str
    scope_type: str
    provider_key_id: str | None

    @property
    def subject(self) -> str:
        """The scope's name in a refusal message."""
        return _SCOPE_SUBJECT.get(self.scope_type, "Budget")


def _as_utc(value: datetime | None) -> datetime | None:
    """Read a stored timestamp as UTC.

    SQLite hands datetimes back naive, and comparing one to an aware ``now``
    raises. A stored value is always the UTC it was written as, so say so.
    """
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=UTC)


def _identity_uuid(user_id: str) -> uuid.UUID | None:
    """The tenancy identity behind a billed user, when there is one.

    ``users.user_id`` is a free-form string, and the attribution row minted for a
    tenancy member is keyed on that member's UUID rendered as a string (see
    ``repositories.users_repository.get_or_create_attribution_user``). Anything
    that does not parse is a plain gateway user with no membership rows, which
    resolves to the workspace and organization ceilings alone.
    """
    try:
        return uuid.UUID(user_id)
    except ValueError:
        return None


async def _resolve_identities(
    db: AsyncSession,
    *,
    user_id: str,
    scope: BudgetScopeRequest,
) -> list[tuple[str, str]]:
    """The ``(scope_type, scope_id)`` pairs a request bills to."""
    workspace_id = await resolve_workspace_id(db, scope.api_key)
    identity = _identity_uuid(user_id)
    workspace_member_id: uuid.UUID | None = None
    org_member_id: uuid.UUID | None = None

    if identity is None:
        organization_id = (
            await db.execute(select(col(Workspace.organization_id)).where(col(Workspace.id) == workspace_id))
        ).scalar_one_or_none()
    else:
        # One query rather than three: the workspace names its organization, and
        # both membership rows hang off those two ids for this identity.
        row = (
            await db.execute(
                select(col(Workspace.organization_id), col(WorkspaceMember.id), col(OrganizationMember.id))
                .outerjoin(
                    WorkspaceMember,
                    and_(
                        col(WorkspaceMember.workspace_id) == col(Workspace.id),
                        col(WorkspaceMember.user_id) == identity,
                    ),
                )
                .outerjoin(
                    OrganizationMember,
                    and_(
                        col(OrganizationMember.organization_id) == col(Workspace.organization_id),
                        col(OrganizationMember.user_id) == identity,
                    ),
                )
                .where(col(Workspace.id) == workspace_id)
            )
        ).first()
        organization_id = None if row is None else row[0]
        if row is not None:
            workspace_member_id = row[1]
            org_member_id = row[2]

    identities: list[tuple[str, str]] = [(SCOPE_WORKSPACE, str(workspace_id))]
    if organization_id is not None:
        identities.append((SCOPE_ORGANIZATION, str(organization_id)))
    if workspace_member_id is not None:
        identities.append((SCOPE_WORKSPACE_MEMBER, str(workspace_member_id)))
    if org_member_id is not None:
        identities.append((SCOPE_ORG_MEMBER, str(org_member_id)))
    if scope.api_key is not None:
        identities.append((SCOPE_API_TOKEN, scope.api_key.id))
    return identities


async def _roll_expired_periods(
    db: AsyncSession,
    expired: Sequence[tuple[str, int | None]],
    now: datetime,
) -> None:
    """Start a fresh window on every ceiling whose period has run out.

    Guarded on ``period_end`` so it is the same lock-free compare-and-swap the
    per-user reset uses: concurrent requests at the boundary all issue it, one
    wins, and the losers see zero rows and carry on against the rolled window.
    ``reserved_spend`` is untouched, so a hold taken before the roll is still
    released against the right counter after it.
    """
    for budget_id, duration in expired:
        await db.execute(
            update(ScopedBudget)
            .where(
                ScopedBudget.id == budget_id,
                ScopedBudget.period_end.is_not(None),
                ScopedBudget.period_end <= now,
            )
            .values(
                current_spend=0.0,
                period_start=now,
                period_end=now + timedelta(seconds=duration) if duration else None,
            )
            .execution_options(synchronize_session=False)
        )
    await db.commit()


async def applicable_budgets(
    db: AsyncSession,
    *,
    user_id: str,
    scope: BudgetScopeRequest,
) -> tuple[ApplicableBudget, ...]:
    """Every ceiling this request must pass, in reservation order.

    A ceiling applies when its identity is one the request bills to AND it is
    either aggregate (no provider) or narrowed to this request's provider. A
    window that has run out is rolled here, before the reservation reads the
    counters, so a request at the boundary is gated on the new period.
    """
    identities = await _resolve_identities(db, user_id=user_id, scope=scope)
    if not identities:
        return ()

    ident_clause = or_(
        *[
            and_(ScopedBudget.scope_type == scope_type, ScopedBudget.scope_id == scope_id)
            for scope_type, scope_id in identities
        ]
    )
    key_clause: ColumnElement[bool] = ScopedBudget.provider_key_id.is_(None)
    if scope.provider_instance is not None:
        key_clause = or_(
            ScopedBudget.provider_key_id == scope.provider_instance,
            ScopedBudget.provider_key_id.is_(None),
        )

    rows = (
        await db.execute(
            select(
                ScopedBudget.id,
                ScopedBudget.scope_type,
                ScopedBudget.provider_key_id,
                ScopedBudget.budget_duration_sec,
                ScopedBudget.period_end,
            ).where(ident_clause, key_clause)
        )
    ).all()
    if not rows:
        return ()

    now = datetime.now(UTC)
    expired = [
        (budget_id, duration)
        for budget_id, _scope_type, _provider, duration, period_end in rows
        if (parsed := _as_utc(period_end)) is not None and now >= parsed
    ]
    if expired:
        await _roll_expired_periods(db, expired, now)

    resolved = [
        ApplicableBudget(budget_id=budget_id, scope_type=scope_type, provider_key_id=provider)
        for budget_id, scope_type, provider, _duration, _period_end in rows
    ]
    # Most specific first, provider-narrowed before aggregate within a scope, then
    # the id so the order stays total when two ceilings tie on both.
    resolved.sort(
        key=lambda budget: (
            _SCOPE_PRECEDENCE.get(budget.scope_type, len(_SCOPE_PRECEDENCE)),
            0 if budget.provider_key_id is not None else 1,
            budget.budget_id,
        )
    )
    return tuple(resolved)


def _release_expression(amount: float) -> object:
    """Subtract ``amount`` from the hold, clamped at zero.

    CASE rather than GREATEST, matching the per-user release, because SQLite has
    no GREATEST.
    """
    return case(
        (ScopedBudget.reserved_spend - amount < 0, 0.0),
        else_=ScopedBudget.reserved_spend - amount,
    )


async def release(db: AsyncSession, budget_ids: Sequence[str], amount: float) -> None:
    """Give a held amount back to every ceiling that took it."""
    if not budget_ids or amount <= 0:
        return
    await db.execute(
        update(ScopedBudget)
        .where(ScopedBudget.id.in_(list(budget_ids)))
        .values(reserved_spend=_release_expression(amount))
        .execution_options(synchronize_session=False)
    )
    await db.commit()


async def reserve(
    db: AsyncSession,
    budgets: Sequence[ApplicableBudget],
    amount: float,
) -> ApplicableBudget | None:
    """Hold ``amount`` on every ceiling, or hold none and name the one that refused.

    One conditional UPDATE per ceiling, each committed before the next is
    attempted, so no lock spans two rows. A ceiling with no limit still takes the
    hold, which keeps the release arithmetic uniform and makes concurrent spend
    visible immediately. Zero rows means this request does not fit, or the
    ceiling is already at its cap (which is what the strict comparison catches,
    including for a zero-cost request): the holds already taken are released and
    the caller rejects.
    """
    taken: list[str] = []
    for budget in budgets:
        result = await db.execute(
            update(ScopedBudget)
            .where(
                ScopedBudget.id == budget.budget_id,
                or_(
                    ScopedBudget.max_budget.is_(None),
                    and_(
                        ScopedBudget.current_spend + ScopedBudget.reserved_spend < ScopedBudget.max_budget,
                        ScopedBudget.current_spend + ScopedBudget.reserved_spend + amount <= ScopedBudget.max_budget,
                    ),
                ),
            )
            .values(reserved_spend=ScopedBudget.reserved_spend + amount)
            .execution_options(synchronize_session=False)
        )
        await db.commit()
        if not getattr(result, "rowcount", 0):
            await release(db, taken, amount)
            return budget
        taken.append(budget.budget_id)
    return None


async def settle(
    db: AsyncSession,
    budget_ids: Sequence[str],
    *,
    actual_cost: float,
    held: float,
    counts_toward_budget: bool = True,
) -> None:
    """Record the real cost on every ceiling and release what the request held."""
    if not budget_ids:
        return
    values: dict[str, object] = {}
    if actual_cost > 0 and counts_toward_budget:
        values["current_spend"] = ScopedBudget.current_spend + actual_cost
    if held > 0:
        values["reserved_spend"] = _release_expression(held)
    if not values:
        return
    await db.execute(
        update(ScopedBudget)
        .where(ScopedBudget.id.in_(list(budget_ids)))
        .values(**values)
        .execution_options(synchronize_session=False)
    )
    await db.commit()


__all__ = [
    "SCOPE_API_TOKEN",
    "SCOPE_ORGANIZATION",
    "SCOPE_ORG_MEMBER",
    "SCOPE_TYPES",
    "ScopeType",
    "SCOPE_WORKSPACE",
    "SCOPE_WORKSPACE_MEMBER",
    "ApplicableBudget",
    "BudgetScopeRequest",
    "applicable_budgets",
    "release",
    "reserve",
    "settle",
]
