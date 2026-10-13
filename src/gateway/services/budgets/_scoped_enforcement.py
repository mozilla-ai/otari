"""Enforce the tenancy-scoped spending ceilings in ``scoped_budgets``.

A request must pass every ceiling attached to a scope it bills to.
The scopes are its workspace, that workspace's organization, the caller's memberships and the API key.
A reservation holds dollars, tokens and requests at once, and a ceiling admits it only when every capped axis has room.
Each ceiling is independent, so the children of a scope may sum to more than their parent.
No hold takes a row lock, so a refusal releases the holds already taken.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import and_, select
from sqlmodel import col

from gateway.log_config import logger
from gateway.models.api_keys import APIKey
from gateway.models.budgets import (
    SCOPE_API_TOKEN,
    SCOPE_ORG_MEMBER,
    SCOPE_ORGANIZATION,
    SCOPE_WORKSPACE,
    SCOPE_WORKSPACE_MEMBER,
    ScopeType,
)
from gateway.models.tenancy import OrganizationMember, Workspace, WorkspaceMember
from gateway.repositories.budgets import ApplicableCeiling, CeilingUsage, ScopedBudgetRepository, ScopeIdSets
from gateway.services.budgets._periods import period_window
from gateway.services.budgets._refusals import BlockedAxis, blocked_on
from gateway.services.workspace_scope import organization_for_workspace_id, resolve_workspace_id

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

ZERO = Decimal(0)

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
    scope_type: ScopeType
    provider_key_id: str | None
    # The non-USD caps of the budget this ceiling names, carried so a caller can
    # tell whether an axis is capped anywhere without reading every budget again.
    # The reserve path reads them from the row instead, inside its one conditional
    # UPDATE, so these are not what enforcement is decided on.
    token_limit: int | None = None
    request_limit: int | None = None

    @property
    def caps_counts(self) -> bool:
        """Whether this ceiling caps tokens or requests as well as dollars."""
        return self.token_limit is not None or self.request_limit is not None

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
) -> ScopeIdSets:
    """The scopes a request bills to.

    Membership is read off the key's owner rather than the billed user. The two
    differ only for a service key billing an end user, and that end user spends
    inside its owner's member ceilings, not around them.
    """
    workspace_id = await resolve_workspace_id(db, scope.api_key)
    owner = scope.api_key.user_id if scope.api_key is not None and scope.api_key.user_id else user_id
    identity = _identity_uuid(owner)
    workspace_member_id: uuid.UUID | None = None
    org_member_id: uuid.UUID | None = None

    if identity is None:
        organization_id = await organization_for_workspace_id(db, workspace_id)
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

    return ScopeIdSets(
        organization_ids=() if organization_id is None else (str(organization_id),),
        workspace_ids=(str(workspace_id),),
        organization_member_ids=() if org_member_id is None else (str(org_member_id),),
        workspace_member_ids=() if workspace_member_id is None else (str(workspace_member_id),),
        api_key_ids=() if scope.api_key is None else (scope.api_key.id,),
    )


async def roll_ended_windows(
    ceilings_repository: ScopedBudgetRepository,
    ceilings: Sequence[ApplicableCeiling],
    now: datetime,
) -> None:
    """Start a fresh window on every one of ``ceilings`` whose period has run out by ``now``.

    Guarded on ``period_end`` so it is the same lock-free compare-and-swap the
    per-user reset uses: concurrent requests at the boundary all issue it, one
    wins, and the losers see zero rows and carry on against the rolled window.
    Every ``current_*`` counter is zeroed and no hold is, so a hold taken before
    the roll is still released against the right counter after it, on each axis.

    No backfill either way: a ceiling untouched for two months lands in the
    current window with fresh counters, not in each window it slept through.
    """
    for ceiling in ceilings:
        if (period_end := _as_utc(ceiling.period_end)) is None or now < period_end:
            continue
        budget_id, alignment = ceiling.scoped_budget_id, ceiling.reset_alignment
        try:
            window = period_window(now, duration=ceiling.budget_duration_sec, alignment=alignment)
        except ValueError:
            # Only reachable by a write that went around the API, and the safe
            # direction is to leave the exhausted window in place: not resetting
            # refuses requests, while guessing a cadence would admit them.
            logger.warning(
                "Scoped budget %s has an unrecognized reset_alignment %r; leaving its period in place",
                budget_id,
                alignment,
            )
            continue
        period_start, next_period_end = window if window is not None else (now, None)
        await ceilings_repository.roll_period(budget_id, now=now, period_start=period_start, period_end=next_period_end)


async def roll_and_list_applicable(
    db: AsyncSession, ceilings: Sequence[ApplicableCeiling]
) -> tuple[ApplicableBudget, ...]:
    """Return ``ceilings`` as the budgets a request must pass, in order, after rolling any window that has ended."""
    now = datetime.now(UTC)
    if any((period_end := _as_utc(ceiling.period_end)) is not None and now >= period_end for ceiling in ceilings):
        await roll_ended_windows(ScopedBudgetRepository(db), ceilings, now)
        await db.commit()
    return tuple(to_applicable_budget(ceiling) for ceiling in ceilings)


def get_provider_ceilings(
    ceilings: Sequence[ApplicableCeiling], provider_instance: str | None
) -> tuple[ApplicableCeiling, ...]:
    """Return the ceilings among ``ceilings`` that cap every provider or ``provider_instance``, in reservation order.

    Most specific first, provider-narrowed before aggregate within a scope, then the ID, so the order is total.
    """
    bound = [
        ceiling
        for ceiling in ceilings
        if ceiling.provider_key_id is None or ceiling.provider_key_id == provider_instance
    ]
    bound.sort(
        key=lambda ceiling: (
            _SCOPE_PRECEDENCE.get(ceiling.scope_type, len(_SCOPE_PRECEDENCE)),
            0 if ceiling.provider_key_id is not None else 1,
            ceiling.scoped_budget_id,
        )
    )
    return tuple(bound)


def to_applicable_budget(ceiling: ApplicableCeiling) -> ApplicableBudget:
    """Return what a hold on ``ceiling`` needs to know of it."""
    return ApplicableBudget(
        budget_id=ceiling.scoped_budget_id,
        scope_type=ceiling.scope_type,
        provider_key_id=ceiling.provider_key_id,
        token_limit=ceiling.token_limit,
        request_limit=ceiling.request_limit,
    )


async def get_scope_ceilings(
    db: AsyncSession, *, user_id: str, scope: BudgetScopeRequest
) -> tuple[ApplicableCeiling, ...]:
    """Return every ceiling on the scopes a request bills to, whichever provider it caps."""
    scopes = await _resolve_identities(db, user_id=user_id, scope=scope)
    return tuple(await ScopedBudgetRepository(db).list_applicable(scopes))


async def release(
    db: AsyncSession,
    budget_ids: Sequence[str],
    amount: Decimal,
    *,
    tokens: int = 0,
    requests: int = 0,
    commit: bool = True,
) -> None:
    """Give a held amount back to every ceiling that took it, on every axis.

    Each axis is released only when the reservation held something on it, so a
    caller that took a dollar hold alone issues the same UPDATE it always did.

    ``commit=False`` folds this into a surrounding unit of work, which is what
    lets the reservation ledger land a hold's terminal status and the release it
    authorizes in one transaction instead of two.
    """
    if not budget_ids or not (amount > ZERO or tokens > 0 or requests > 0):
        return
    await ScopedBudgetRepository(db).release_holds(budget_ids, amount, tokens=tokens, requests=requests)
    if commit:
        await db.commit()


async def reserve(
    db: AsyncSession,
    budgets: Sequence[ApplicableBudget],
    amount: Decimal | None,
    *,
    tokens: int = 0,
    requests: int = 1,
    new_request: bool = True,
) -> ApplicableBudget | None:
    """Hold a request on every ceiling, or hold none and name the one that refused.

    One conditional UPDATE per ceiling, each committed before the next is
    attempted, so no lock spans two rows. A ceiling with no limit still takes the
    hold, which keeps the release arithmetic uniform and makes concurrent spend
    visible immediately. When a ceiling has no room, the holds already taken are
    released and the caller rejects.

    Every capped axis has to admit the request, and the hold is placed on all
    three at once so a later refusal gives back exactly what was taken. An axis
    the budget leaves NULL admits any hold, which is how a dollars-only budget
    keeps behaving as it did before tokens and requests could be capped.

    ``new_request=False`` grows a request this ceiling has already admitted, and
    is asked only whether the delta fits. A top-up is otherwise refused by its
    own hold: on a request cap of one, the reservation it is growing has already
    taken the only slot.

    ``amount=None`` is a request with no dollar amount, which is a free-priced
    model: the dollar axis is neither held nor gated, because the request cannot
    spend, while the token and request axes hold and gate as they do for any
    other request. It still consumes tokens and is still a request.
    """
    ceilings = ScopedBudgetRepository(db)
    taken: list[str] = []
    for budget in budgets:
        held = await ceilings.hold(budget.budget_id, amount, tokens=tokens, requests=requests, new_request=new_request)
        await db.commit()
        if not held:
            await release(db, taken, amount if amount is not None else ZERO, tokens=tokens, requests=requests)
            return budget
        taken.append(budget.budget_id)
    return None


async def blocked_axis(
    db: AsyncSession,
    budget: ApplicableBudget,
    *,
    amount: Decimal | None,
    tokens: int,
    requests: int,
    new_request: bool,
) -> BlockedAxis:
    """Return the capped axis that left this ceiling no room.

    The test must mirror both clauses of the admission test in ``ScopedBudgetRepository.hold``,
    or a refusal names the wrong axis.
    ``amount=None`` skips the dollar axis, which is named "budget".
    """
    usage = await ScopedBudgetRepository(db).get_usage(budget.budget_id)
    return refused_axis(usage, amount=amount, tokens=tokens, requests=requests, new_request=new_request)


def refused_axis(
    usage: CeilingUsage | None,
    *,
    amount: Decimal | None,
    tokens: int,
    requests: int,
    new_request: bool,
) -> BlockedAxis:
    """Return the capped axis that left a ceiling with ``usage`` no room, as :func:`blocked_axis` does."""
    if usage is None:
        # The ceiling went away between the refusal and here, so there is no axis
        # to name and guessing one would be worse than saying nothing.
        return BlockedAxis("budget")
    axes = (
        # Skipped when the request has no dollar amount, which is a free-priced
        # model: :func:`reserve` does not gate that axis, so it cannot be what
        # refused, and naming it would point at the one cap with room.
        (
            "budget",
            usage.max_budget if amount is not None else None,
            usage.current_spend,
            usage.reserved_spend,
            amount or ZERO,
        ),
        ("token", usage.token_limit, usage.current_tokens, usage.reserved_tokens, tokens),
        ("request", usage.request_limit, usage.current_requests, usage.reserved_requests, requests),
    )
    for name, cap, spent, reserved, wanted in axes:
        if cap is None:
            continue
        # Compared as Decimals whatever the driver handed back: a NUMERIC column
        # arrives as Decimal on PostgreSQL and can arrive as a float on SQLite,
        # and adding one of each raises rather than comparing.
        blocked = blocked_on(
            name,
            committed=Decimal(str(spent)) + Decimal(str(reserved)),
            cap=Decimal(str(cap)),
            wanted=Decimal(str(wanted)),
            new_request=new_request,
        )
        if blocked is not None:
            return replace(blocked, resets_at=_as_utc(usage.period_end))
    return BlockedAxis("budget")


async def settle(
    db: AsyncSession,
    budget_ids: Sequence[str],
    *,
    actual_cost: Decimal,
    held: Decimal,
    actual_tokens: int = 0,
    held_tokens: int = 0,
    requests: int = 0,
    held_requests: int = 0,
    counts_toward_budget: bool = True,
    commit: bool = True,
) -> None:
    """Record what the request really used on every ceiling and release its holds.

    What is recorded and what is released are separate arguments on every axis,
    because they diverge in two ways. A dollar or token estimate is released while
    the measured figure is recorded; and a reservation the TTL sweep already
    reclaimed has a figure to record and nothing left to release, which is what
    the ``held_*`` arguments express when they are zero.

    ``commit=False`` folds this into a surrounding unit of work; see
    :func:`release`.
    """
    spend = actual_cost if counts_toward_budget else ZERO
    tokens = actual_tokens if counts_toward_budget else 0
    counted_requests = requests if counts_toward_budget else 0
    if not budget_ids or not (
        spend > ZERO or held > ZERO or tokens > 0 or held_tokens > 0 or counted_requests > 0 or held_requests > 0
    ):
        return
    await ScopedBudgetRepository(db).settle(
        budget_ids,
        spend=spend,
        held=held,
        tokens=tokens,
        held_tokens=held_tokens,
        requests=counted_requests,
        held_requests=held_requests,
    )
    if commit:
        await db.commit()


__all__ = [
    "ApplicableBudget",
    "BudgetScopeRequest",
    "blocked_axis",
    "get_provider_ceilings",
    "get_scope_ceilings",
    "refused_axis",
    "release",
    "reserve",
    "roll_and_list_applicable",
    "roll_ended_windows",
    "settle",
    "to_applicable_budget",
]
