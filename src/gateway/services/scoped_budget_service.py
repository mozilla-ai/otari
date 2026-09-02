"""Enforcement for ``scoped_budgets``: the tenancy-scoped spending ceilings.

The legacy per-user path in :mod:`gateway.services.budget_service` is unchanged
and still enforced; this is a second mechanism beside it. A request resolves the
scopes it bills to (its workspace, that workspace's organization, the caller's
membership rows, and the API key itself), collects every ceiling attached to one
of them, and must pass all of them.

Each row is an independent ceiling. There is deliberately no check that the
children of a scope sum to less than their parent: the parent ceiling already
bounds the total, so the extra rule would refuse configurations that cannot
overspend.

**Three axes, one hold.** The budget a ceiling names can cap dollars, tokens and
requests independently (``max_budget``, ``token_limit``, ``request_limit``), so a
reservation holds on all three at once and a ceiling admits it only when every
capped axis has room. The dollar and token amounts are estimates reconciled at
settlement; the request count is exact at admission, so its hold is what
settlement records.

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
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal, get_args

from sqlalchemy import and_, case, or_, select, update
from sqlalchemy.orm import Mapped
from sqlalchemy.sql.elements import ColumnElement
from sqlmodel import col

from gateway.log_config import logger
from gateway.models.entities import APIKey, Budget, ScopedBudget
from gateway.models.tenancy import OrganizationMember, Workspace, WorkspaceMember
from gateway.services.budget_periods import (
    ALIGN_DAY,
    ALIGN_MONTH,
    ALIGN_WEEK,
    ResetAlignment,
    period_window,
)
from gateway.services.workspace_scope import resolve_workspace_id

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

# The counters on this table are ``NUMERIC(18, 6)`` (mozilla-ai/otari#691), so
# the constants the SQL is built from are ``Decimal``: a bare ``0.0`` in a CASE
# arm or a ``.values()`` would make PostgreSQL resolve the expression as double
# precision and hand a binary-rounded amount back to an exact column.
ZERO = Decimal(0)

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
    expired: Sequence[tuple[str, int | None, str | None]],
    now: datetime,
) -> None:
    """Start a fresh window on every ceiling whose period has run out.

    Guarded on ``period_end`` so it is the same lock-free compare-and-swap the
    per-user reset uses: concurrent requests at the boundary all issue it, one
    wins, and the losers see zero rows and carry on against the rolled window.
    Every ``current_*`` counter is zeroed and no hold is, so a hold taken before
    the roll is still released against the right counter after it, on each axis.

    No backfill either way: a ceiling untouched for two months lands in the
    current window with fresh counters, not in each window it slept through.
    """
    for budget_id, duration, alignment in expired:
        try:
            window = period_window(now, duration=duration, alignment=alignment)
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
        period_start, period_end = window if window is not None else (now, None)
        await db.execute(
            update(ScopedBudget)
            .where(
                ScopedBudget.id == budget_id,
                ScopedBudget.period_end.is_not(None),
                ScopedBudget.period_end <= now,
            )
            .values(
                current_spend=ZERO,
                current_tokens=0,
                current_requests=0,
                period_start=period_start,
                period_end=period_end,
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
                Budget.budget_duration_sec,
                Budget.reset_alignment,
                ScopedBudget.period_end,
                Budget.token_limit,
                Budget.request_limit,
            )
            .join(Budget, Budget.budget_id == ScopedBudget.budget_id)
            .where(ident_clause, key_clause)
        )
    ).all()
    if not rows:
        return ()

    now = datetime.now(UTC)
    expired = [
        (budget_id, duration, alignment)
        for budget_id, _scope_type, _provider, duration, alignment, period_end, _tokens, _requests in rows
        if (parsed := _as_utc(period_end)) is not None and now >= parsed
    ]
    if expired:
        await _roll_expired_periods(db, expired, now)

    resolved = [
        ApplicableBudget(
            budget_id=budget_id,
            scope_type=scope_type,
            provider_key_id=provider,
            token_limit=token_limit,
            request_limit=request_limit,
        )
        for budget_id, scope_type, provider, _duration, _alignment, _period_end, token_limit, request_limit in rows
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


def _release_expression(amount: Decimal) -> object:
    """Subtract ``amount`` from the money hold, clamped at zero.

    CASE rather than GREATEST, matching the per-user release, because SQLite has
    no GREATEST. Both arms are ``Decimal`` so the CASE resolves as ``numeric``.
    """
    return case(
        (ScopedBudget.reserved_spend - amount < ZERO, ZERO),
        else_=ScopedBudget.reserved_spend - amount,
    )


def _release_count_expression(column: Mapped[int], amount: int) -> object:
    """The same clamp for a token or request hold, whose arms are integers."""
    return case((column - amount < 0, 0), else_=column - amount)


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
    values: dict[str, object] = {}
    if amount > ZERO:
        values["reserved_spend"] = _release_expression(amount)
    if tokens > 0:
        values["reserved_tokens"] = _release_count_expression(ScopedBudget.reserved_tokens, tokens)
    if requests > 0:
        values["reserved_requests"] = _release_count_expression(ScopedBudget.reserved_requests, requests)
    if not budget_ids or not values:
        return
    await db.execute(
        update(ScopedBudget)
        .where(ScopedBudget.id.in_(list(budget_ids)))
        .values(**values)
        .execution_options(synchronize_session=False)
    )
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
    visible immediately. Zero rows means the request does not fit, per
    :func:`admits`: the holds already taken are released and the caller rejects.

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
    taken: list[str] = []
    usd = amount if amount is not None else ZERO

    # Each cap is a column on the budget this ceiling names, read as a correlated
    # subquery so the whole check stays inside the one conditional UPDATE. Reading
    # them first and comparing in Python would reintroduce the read-then-write
    # race this service exists to close. Three subqueries over one row rather than
    # a join: the row is the same buffered page for all three, and a join here
    # would have to be an UPDATE ... FROM, which SQLite does not take.
    def cap_of(column: Mapped[Decimal | None] | Mapped[int | None]) -> Any:
        return select(column).where(Budget.budget_id == ScopedBudget.budget_id).scalar_subquery()

    def admits(committed: Any, held: Any, cap: Any) -> ColumnElement[bool]:
        """Whether one axis has room, or has no cap at all.

        Two clauses on an arrival: it is refused when the axis is already at or
        over its cap (which is what catches a request estimating nothing against
        an exhausted ceiling), and refused when this hold would push it past.
        Only the second on a top-up, per ``new_request``.
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
            (ScopedBudget.current_requests + ScopedBudget.reserved_requests, requests, cap_of(Budget.request_limit)),
        )
        if held is not None
    ]
    for budget in budgets:
        result = await db.execute(
            update(ScopedBudget)
            .where(
                ScopedBudget.id == budget.budget_id,
                *gates,
            )
            .values(
                reserved_spend=ScopedBudget.reserved_spend + usd,
                reserved_tokens=ScopedBudget.reserved_tokens + tokens,
                reserved_requests=ScopedBudget.reserved_requests + requests,
            )
            .execution_options(synchronize_session=False)
        )
        await db.commit()
        if not getattr(result, "rowcount", 0):
            await release(db, taken, usd, tokens=tokens, requests=requests)
            return budget
        taken.append(budget.budget_id)
    return None


async def blocked_axis(db: AsyncSession, budget: ApplicableBudget) -> str:
    """Which capped axis left this ceiling no room, for the refusal message.

    :func:`reserve` refuses by matching no row, so nothing in its result says
    which of three caps bound. Read here instead, on the refusal path only, and
    named in the 403: "has exceeded budget limit" alone cannot tell an operator a
    spent allowance from a spent token allowance, which is the signal a cutover
    onto a token or request cap needs. Returns the first exhausted axis, in the
    order :func:`reserve` builds its clauses; where two are exhausted at once,
    either answer is true.
    """
    row = (
        await db.execute(
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
            )
            .join(Budget, Budget.budget_id == ScopedBudget.budget_id)
            .where(ScopedBudget.id == budget.budget_id)
        )
    ).first()
    if row is None:
        # The ceiling went away between the refusal and here, so there is no axis
        # to name and guessing one would be worse than saying nothing.
        return "budget"
    for name, cap, spent, held in (
        # "budget" rather than "spend" for the dollar axis: the message a dollar
        # refusal sends is unchanged, and the two new axes are the ones that had
        # no signal. See ``budget_service._blocked_axis``.
        ("budget", row[0], row[1], row[2]),
        ("token", row[3], row[4], row[5]),
        ("request", row[6], row[7], row[8]),
    ):
        # Compared as Decimals whatever the driver handed back: a NUMERIC column
        # arrives as Decimal on PostgreSQL and can arrive as float on SQLite, and
        # adding one of each raises rather than comparing.
        if cap is not None and Decimal(str(spent)) + Decimal(str(held)) >= Decimal(str(cap)):
            return name
    return "budget"


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
    if not budget_ids:
        return
    values: dict[str, object] = {}
    if actual_cost > ZERO and counts_toward_budget:
        values["current_spend"] = ScopedBudget.current_spend + actual_cost
    if held > ZERO:
        values["reserved_spend"] = _release_expression(held)
    if actual_tokens > 0 and counts_toward_budget:
        values["current_tokens"] = ScopedBudget.current_tokens + actual_tokens
    if held_tokens > 0:
        values["reserved_tokens"] = _release_count_expression(ScopedBudget.reserved_tokens, held_tokens)
    if requests > 0 and counts_toward_budget:
        values["current_requests"] = ScopedBudget.current_requests + requests
    if held_requests > 0:
        values["reserved_requests"] = _release_count_expression(ScopedBudget.reserved_requests, held_requests)
    if not values:
        return
    await db.execute(
        update(ScopedBudget)
        .where(ScopedBudget.id.in_(list(budget_ids)))
        .values(**values)
        .execution_options(synchronize_session=False)
    )
    if commit:
        await db.commit()


__all__ = [
    "ALIGN_DAY",
    "ALIGN_MONTH",
    "ALIGN_WEEK",
    "ResetAlignment",
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
    "blocked_axis",
    "period_window",
    "release",
    "reserve",
    "settle",
]
