"""Enforcement and management of the tenancy-scoped budget ceilings.

The legacy per-user path (``budgets`` + ``users.spend``/``users.reserved``) is
covered by ``test_budget_race_condition.py``; the last test here asserts that it
is untouched by a request that also passes scoped ceilings.
"""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from gateway.models.entities import APIKey, Budget, ScopedBudget, User
from gateway.models.tenancy import Organization, OrganizationMember, Workspace, WorkspaceMember
from gateway.models.tenancy import User as TenancyUser
from gateway.services.budget_service import (
    ReservationHandle,
    reconcile_reservation,
    refund_reservation,
    reserve_budget,
)
from gateway.services.scoped_budget_service import BudgetScopeRequest

from .conftest import _to_async_url


class Fixture:
    """The tenancy graph one test bills against."""

    def __init__(
        self,
        *,
        organization_id: uuid.UUID,
        workspace_id: uuid.UUID,
        org_member_id: uuid.UUID,
        workspace_member_id: uuid.UUID,
        user_id: str,
        api_key: APIKey,
    ) -> None:
        self.organization_id = organization_id
        self.workspace_id = workspace_id
        self.org_member_id = org_member_id
        self.workspace_member_id = workspace_member_id
        self.user_id = user_id
        self.api_key = api_key

    def scope(self, provider: str | None = "openai") -> BudgetScopeRequest:
        return BudgetScopeRequest(api_key=self.api_key, provider_instance=provider)


async def _build_tenancy(db: AsyncSession, slug: str) -> Fixture:
    """Create an organization, a workspace, a member of both, and a key."""
    organization = Organization(name=f"Org {slug}", slug=slug)
    db.add(organization)
    await db.flush()

    identity = TenancyUser(email=f"{slug}@example.test", active_organization_id=organization.id)
    db.add(identity)
    await db.flush()

    workspace = Workspace(name=f"Workspace {slug}", organization_id=organization.id)
    db.add(workspace)
    await db.flush()

    org_member = OrganizationMember(organization_id=organization.id, user_id=identity.id)
    workspace_member = WorkspaceMember(workspace_id=workspace.id, user_id=identity.id)
    db.add(org_member)
    db.add(workspace_member)
    await db.flush()

    # The request plane bills to a string-keyed user whose id is the tenancy
    # identity's UUID, which is the bridge the resolver walks back.
    attribution_id = str(identity.id)
    db.add(User(user_id=attribution_id))
    api_key = APIKey(
        id=f"key-{slug}",
        key_hash=f"hash-{slug}",
        workspace_id=workspace.id,
        user_id=attribution_id,
    )
    db.add(api_key)
    await db.commit()

    return Fixture(
        organization_id=organization.id,
        workspace_id=workspace.id,
        org_member_id=org_member.id,
        workspace_member_id=workspace_member.id,
        user_id=attribution_id,
        api_key=api_key,
    )


async def _counters(db: AsyncSession, budget_id: str) -> tuple[float, float]:
    row = (
        await db.execute(
            select(ScopedBudget.current_spend, ScopedBudget.reserved_spend).where(ScopedBudget.id == budget_id)
        )
    ).one()
    return float(row[0]), float(row[1])


async def _token_counters(db: AsyncSession, budget_id: str) -> tuple[int, int]:
    row = (
        await db.execute(
            select(ScopedBudget.current_tokens, ScopedBudget.reserved_tokens).where(ScopedBudget.id == budget_id)
        )
    ).one()
    return int(row[0]), int(row[1])


async def _request_counters(db: AsyncSession, budget_id: str) -> tuple[int, int]:
    row = (
        await db.execute(
            select(ScopedBudget.current_requests, ScopedBudget.reserved_requests).where(ScopedBudget.id == budget_id)
        )
    ).one()
    return int(row[0]), int(row[1])


async def _scoped(
    db: AsyncSession,
    *,
    scope_type: str,
    scope_id: str,
    max_budget: float | None,
    token_limit: int | None = None,
    request_limit: int | None = None,
    provider_key_id: str | None = None,
    budget_duration_sec: int | None = None,
    reset_alignment: str | None = None,
    period_start: datetime | None = None,
    period_end: datetime | None = None,
) -> ScopedBudget:
    """A ceiling, and the budget it enforces.

    A ceiling carries no limit of its own any more: it names a budget, which is
    the only row that maps a cap to an amount. Tests still say "a $10 ceiling",
    so this mints the budget behind it rather than making every case do so.
    """
    budget = Budget(
        max_budget=max_budget,
        token_limit=token_limit,
        request_limit=request_limit,
        budget_duration_sec=budget_duration_sec,
        reset_alignment=reset_alignment,
    )
    db.add(budget)
    await db.flush()
    return ScopedBudget(
        scope_type=scope_type,
        scope_id=scope_id,
        provider_key_id=provider_key_id,
        budget_id=budget.budget_id,
        period_start=period_start,
        period_end=period_end,
    )


@pytest_asyncio.fixture
async def tenancy(async_db: AsyncSession) -> Fixture:
    return await _build_tenancy(async_db, "acme")


@pytest.mark.asyncio
async def test_request_passing_two_ceilings_holds_on_both(async_db: AsyncSession, tenancy: Fixture) -> None:
    """An organization cap and a member cap both admit the estimate and both hold it."""
    org_cap = await _scoped(
        async_db, scope_type="organization", scope_id=str(tenancy.organization_id), max_budget=100.0
    )
    member_cap = await _scoped(
        async_db, scope_type="workspace_member", scope_id=str(tenancy.workspace_member_id), max_budget=10.0
    )
    async_db.add(org_cap)
    async_db.add(member_cap)
    await async_db.commit()

    handle = await reserve_budget(async_db, tenancy.user_id, 2.0, scope=tenancy.scope())

    assert set(handle.scoped_budget_ids) == {org_cap.id, member_cap.id}
    assert await _counters(async_db, org_cap.id) == (0.0, 2.0)
    assert await _counters(async_db, member_cap.id) == (0.0, 2.0)

    await reconcile_reservation(async_db, handle, 1.5)

    assert await _counters(async_db, org_cap.id) == (1.5, 0.0)
    assert await _counters(async_db, member_cap.id) == (1.5, 0.0)


@pytest.mark.asyncio
async def test_tighter_ceiling_rejects_and_compensates_the_looser_one(async_db: AsyncSession, tenancy: Fixture) -> None:
    """The organization cap admits the estimate, the member cap refuses, and the
    hold already taken on the organization is given back."""
    org_cap = await _scoped(
        async_db, scope_type="organization", scope_id=str(tenancy.organization_id), max_budget=100.0
    )
    member_cap = await _scoped(
        async_db, scope_type="workspace_member", scope_id=str(tenancy.workspace_member_id), max_budget=1.0
    )
    async_db.add(org_cap)
    async_db.add(member_cap)
    await async_db.commit()

    with pytest.raises(HTTPException) as exc_info:
        await reserve_budget(async_db, tenancy.user_id, 5.0, scope=tenancy.scope())
    assert exc_info.value.status_code == 403
    assert "Member" in str(exc_info.value.detail)

    # No counter leaks: the member cap never took a hold, and the organization's
    # was compensated on the way out.
    assert await _counters(async_db, org_cap.id) == (0.0, 0.0)
    assert await _counters(async_db, member_cap.id) == (0.0, 0.0)


@pytest.mark.asyncio
async def test_per_user_refusal_compensates_the_scoped_holds(async_db: AsyncSession, tenancy: Fixture) -> None:
    """The scoped ceilings admit the request, the legacy per-user cap refuses,
    and the scoped holds are released before the 403."""
    org_cap = await _scoped(
        async_db, scope_type="organization", scope_id=str(tenancy.organization_id), max_budget=100.0
    )
    async_db.add(org_cap)
    async_db.add(Budget(budget_id="tiny", max_budget=1.0))
    await async_db.commit()

    user = (await async_db.execute(select(User).where(User.user_id == tenancy.user_id))).scalar_one()
    user.budget_id = "tiny"
    user.spend = Decimal("1.0")
    await async_db.commit()

    with pytest.raises(HTTPException) as exc_info:
        await reserve_budget(async_db, tenancy.user_id, 0.5, scope=tenancy.scope())
    assert exc_info.value.status_code == 403

    assert await _counters(async_db, org_cap.id) == (0.0, 0.0)


@pytest.mark.asyncio
async def test_provider_narrowed_cap_binds_only_that_provider(async_db: AsyncSession, tenancy: Fixture) -> None:
    """A cap narrowed to one provider is invisible to a request on another."""
    narrowed = await _scoped(
        async_db,
        scope_type="workspace",
        scope_id=str(tenancy.workspace_id),
        max_budget=1.0,
        provider_key_id="openai",
    )
    async_db.add(narrowed)
    await async_db.commit()

    # A request on a different provider does not resolve the cap at all.
    other = await reserve_budget(async_db, tenancy.user_id, 5.0, scope=tenancy.scope("anthropic"))
    assert other.scoped_budget_ids == ()
    assert await _counters(async_db, narrowed.id) == (0.0, 0.0)

    # The same estimate on the narrowed provider is refused by it.
    with pytest.raises(HTTPException) as exc_info:
        await reserve_budget(async_db, tenancy.user_id, 5.0, scope=tenancy.scope("openai"))
    assert exc_info.value.status_code == 403
    assert "Workspace" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_narrowed_and_aggregate_caps_both_apply(async_db: AsyncSession, tenancy: Fixture) -> None:
    """A provider-narrowed cap and the workspace's aggregate cap are independent
    ceilings, and a request on that provider holds against both."""
    aggregate = await _scoped(async_db, scope_type="workspace", scope_id=str(tenancy.workspace_id), max_budget=50.0)
    narrowed = await _scoped(
        async_db,
        scope_type="workspace",
        scope_id=str(tenancy.workspace_id),
        max_budget=5.0,
        provider_key_id="openai",
    )
    async_db.add(aggregate)
    async_db.add(narrowed)
    await async_db.commit()

    handle = await reserve_budget(async_db, tenancy.user_id, 3.0, scope=tenancy.scope("openai"))
    assert set(handle.scoped_budget_ids) == {aggregate.id, narrowed.id}
    # Provider-narrowed first: the most specific ceiling is the one that reports.
    assert handle.scoped_budget_ids[0] == narrowed.id

    await refund_reservation(async_db, handle)
    assert await _counters(async_db, aggregate.id) == (0.0, 0.0)
    assert await _counters(async_db, narrowed.id) == (0.0, 0.0)


@pytest.mark.asyncio
async def test_expired_period_rolls_before_the_gate(async_db: AsyncSession, tenancy: Fixture) -> None:
    """A ceiling whose window has run out starts a fresh one, so a request that
    would not fit the old period's spend is admitted."""
    now = datetime.now(UTC)
    cap = await _scoped(
        async_db,
        scope_type="workspace",
        scope_id=str(tenancy.workspace_id),
        max_budget=10.0,
        budget_duration_sec=3600,
        period_start=now - timedelta(seconds=7200),
        period_end=now - timedelta(seconds=3600),
    )
    cap.current_spend = Decimal("9.5")
    async_db.add(cap)
    await async_db.commit()

    handle = await reserve_budget(async_db, tenancy.user_id, 5.0, scope=tenancy.scope())
    assert handle.scoped_budget_ids == (cap.id,)
    assert await _counters(async_db, cap.id) == (0.0, 5.0)

    refreshed = (await async_db.execute(select(ScopedBudget).where(ScopedBudget.id == cap.id))).scalar_one()
    await async_db.refresh(refreshed)
    assert refreshed.period_end is not None
    assert refreshed.period_end > now


def _midnight(moment: datetime) -> datetime:
    return moment.replace(hour=0, minute=0, second=0, microsecond=0)


def _first_of_month(moment: datetime) -> datetime:
    return _midnight(moment).replace(day=1)


def _first_of_next_month(moment: datetime) -> datetime:
    first = _first_of_month(moment)
    return first.replace(year=first.year + 1, month=1) if first.month == 12 else first.replace(month=first.month + 1)


# The code under test reads the clock itself, so a test that reads its own cannot
# assert one exact boundary: the two reads straddle a boundary if the call spans
# one. Bracketing the call and accepting either candidate keeps the assertion
# race-free while still proving the window came from a boundary and not from the
# request instant. The exact month arithmetic is pinned in
# ``tests/unit/test_scoped_budget_periods.py``, where the clock is an argument.


@pytest.mark.asyncio
async def test_an_aligned_period_rolls_onto_the_boundary_not_onto_the_request(
    async_db: AsyncSession, tenancy: Fixture
) -> None:
    """A daily ceiling rolled three days late still lands on today's window.

    This is the difference the column buys. A rolling window is measured from the
    first request after expiry, so on a quiet deployment the reset time walks
    forward and each delay is permanent; an aligned one is derived from the
    boundary, so when the roll happened does not leak into the data.
    """
    now = datetime.now(UTC)
    midnight = _midnight(now)
    cap = await _scoped(
        async_db,
        scope_type="workspace",
        scope_id=str(tenancy.workspace_id),
        max_budget=10.0,
        reset_alignment="calendar_day",
        period_start=midnight - timedelta(days=4),
        period_end=midnight - timedelta(days=3),
    )
    cap.current_spend = Decimal("9.5")
    async_db.add(cap)
    await async_db.commit()

    before = datetime.now(UTC)
    handle = await reserve_budget(async_db, tenancy.user_id, 5.0, scope=tenancy.scope())
    after = datetime.now(UTC)
    assert handle.scoped_budget_ids == (cap.id,)
    assert await _counters(async_db, cap.id) == (0.0, 5.0)

    refreshed = (await async_db.execute(select(ScopedBudget).where(ScopedBudget.id == cap.id))).scalar_one()
    await async_db.refresh(refreshed)
    assert refreshed.period_start in {_midnight(before), _midnight(after)}
    assert refreshed.period_end == refreshed.period_start + timedelta(days=1)
    # And the window contains the roll, so the next request is gated on it rather
    # than rolling again.
    assert refreshed.period_start <= after < refreshed.period_end


@pytest.mark.asyncio
async def test_a_monthly_ceiling_asleep_for_two_months_lands_in_the_current_one(
    async_db: AsyncSession, tenancy: Fixture
) -> None:
    """No backfill: the window containing ``now``, with fresh counters, not one
    window per month it slept through."""
    first_of_month = _first_of_month(datetime.now(UTC))
    cap = await _scoped(
        async_db,
        scope_type="workspace",
        scope_id=str(tenancy.workspace_id),
        max_budget=10.0,
        reset_alignment="calendar_month",
        period_start=first_of_month - timedelta(days=90),
        period_end=first_of_month - timedelta(days=60),
    )
    cap.current_spend = Decimal("10.0")
    async_db.add(cap)
    await async_db.commit()

    before = datetime.now(UTC)
    await reserve_budget(async_db, tenancy.user_id, 1.0, scope=tenancy.scope())
    after = datetime.now(UTC)

    refreshed = (await async_db.execute(select(ScopedBudget).where(ScopedBudget.id == cap.id))).scalar_one()
    await async_db.refresh(refreshed)
    assert refreshed.period_start in {_first_of_month(before), _first_of_month(after)}
    assert refreshed.period_end == _first_of_next_month(refreshed.period_start)
    assert refreshed.period_start <= after < refreshed.period_end
    assert refreshed.current_spend == 0.0


@pytest.mark.asyncio
async def test_an_unrecognized_alignment_leaves_the_exhausted_window_in_place(
    async_db: AsyncSession, tenancy: Fixture
) -> None:
    """A value the API cannot create, so only a write that went around it. The
    safe direction is refusing requests, not guessing a cadence and admitting
    them, so the window stays where it is and the cap stays exhausted."""
    now = datetime.now(UTC)
    cap = await _scoped(
        async_db,
        scope_type="workspace",
        scope_id=str(tenancy.workspace_id),
        max_budget=10.0,
        reset_alignment="calendar_quarter",
        period_start=now - timedelta(days=2),
        period_end=now - timedelta(days=1),
    )
    cap.current_spend = Decimal("10.0")
    async_db.add(cap)
    await async_db.commit()

    with pytest.raises(HTTPException) as exc_info:
        await reserve_budget(async_db, tenancy.user_id, 1.0, scope=tenancy.scope())
    assert exc_info.value.status_code == 403

    refreshed = (await async_db.execute(select(ScopedBudget).where(ScopedBudget.id == cap.id))).scalar_one()
    await async_db.refresh(refreshed)
    assert refreshed.period_end == now - timedelta(days=1)
    assert refreshed.current_spend == 10.0


@pytest.mark.asyncio
async def test_a_budget_cannot_carry_both_kinds_of_period(async_db: AsyncSession, tenancy: Fixture) -> None:
    """The fourth state is not storable, so the pair never needs an "ignored
    when" rule to be read.

    The constraint moved onto ``budgets`` with the cadence itself: a ceiling has
    no period of its own to contradict any more.
    """
    async_db.add(Budget(max_budget=10.0, budget_duration_sec=86400, reset_alignment="calendar_month"))
    with pytest.raises(IntegrityError):
        await async_db.commit()
    await async_db.rollback()


@pytest.mark.asyncio
async def test_concurrent_reservers_cannot_overspend_a_shared_ceiling(
    async_db: AsyncSession, postgres_url: str, tenancy: Fixture
) -> None:
    """Eight simultaneous reservations against a ceiling with room for four all
    contend on the same conditional UPDATE; exactly four are admitted."""
    cap = await _scoped(async_db, scope_type="organization", scope_id=str(tenancy.organization_id), max_budget=4.0)
    async_db.add(cap)
    await async_db.commit()

    engine = create_async_engine(_to_async_url(postgres_url), pool_pre_ping=True)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def attempt() -> bool:
        async with factory() as session:
            try:
                await reserve_budget(session, tenancy.user_id, 1.0, scope=tenancy.scope())
            except HTTPException:
                return False
            return True

    try:
        results = await asyncio.gather(*(attempt() for _ in range(8)))
    finally:
        await engine.dispose()

    assert sum(results) == 4
    current, reserved = await _counters(async_db, cap.id)
    assert current == 0.0
    assert reserved == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_legacy_per_user_path_is_unchanged(async_db: AsyncSession, tenancy: Fixture) -> None:
    """The per-user counters behave exactly as before beside a scoped ceiling:
    the estimate lands on ``users.reserved`` and settles into ``users.spend``,
    and nothing on the scoped row changes either of them."""
    async_db.add(Budget(budget_id="legacy", max_budget=20.0))
    cap = await _scoped(async_db, scope_type="organization", scope_id=str(tenancy.organization_id), max_budget=100.0)
    async_db.add(cap)
    await async_db.commit()

    user = (await async_db.execute(select(User).where(User.user_id == tenancy.user_id))).scalar_one()
    user.budget_id = "legacy"
    await async_db.commit()

    handle = await reserve_budget(async_db, tenancy.user_id, 4.0, scope=tenancy.scope())
    assert handle.reserved is True
    assert handle.estimate == pytest.approx(4.0)

    async_db.expire_all()
    user = (await async_db.execute(select(User).where(User.user_id == tenancy.user_id))).scalar_one()
    assert user.reserved == pytest.approx(4.0)
    assert user.spend == pytest.approx(0.0)

    await reconcile_reservation(async_db, handle, 2.5)

    async_db.expire_all()
    user = (await async_db.execute(select(User).where(User.user_id == tenancy.user_id))).scalar_one()
    assert user.reserved == pytest.approx(0.0)
    assert user.spend == pytest.approx(2.5)


@pytest.mark.asyncio
async def test_no_scope_leaves_scoped_ceilings_untouched(async_db: AsyncSession, tenancy: Fixture) -> None:
    """A caller that passes no scope is on the pre-existing path exactly: the
    scoped rows are neither resolved nor charged."""
    cap = await _scoped(async_db, scope_type="organization", scope_id=str(tenancy.organization_id), max_budget=1.0)
    async_db.add(cap)
    await async_db.commit()

    handle = await reserve_budget(async_db, tenancy.user_id, 100.0)
    assert handle.scoped_budgets == ()
    assert await _counters(async_db, cap.id) == (0.0, 0.0)


@pytest.mark.asyncio
async def test_budget_exempt_key_skips_scoped_ceilings(async_db: AsyncSession, tenancy: Fixture) -> None:
    """A key flagged ``exclude_from_budget`` is exempt from the scoped ceilings
    too, matching what the flag already means for the per-user cap."""
    cap = await _scoped(async_db, scope_type="organization", scope_id=str(tenancy.organization_id), max_budget=1.0)
    async_db.add(cap)
    await async_db.commit()

    handle = await reserve_budget(
        async_db,
        tenancy.user_id,
        100.0,
        scope=tenancy.scope(),
        counts_toward_budget=False,
    )
    assert handle.scoped_budgets == ()
    assert await _counters(async_db, cap.id) == (0.0, 0.0)


@pytest.mark.asyncio
async def test_settle_is_inert_for_a_handle_that_held_nothing(async_db: AsyncSession) -> None:
    """Every pre-existing construction site builds a handle with no scoped rows;
    reconciling one must not error or write anything."""
    async_db.add(User(user_id="plain-user"))
    await async_db.commit()

    handle = ReservationHandle(user_id="plain-user", estimate=Decimal(0), reserved=False, strategy="disabled")
    await reconcile_reservation(async_db, handle, 3.0)

    async_db.expire_all()
    user = (await async_db.execute(select(User).where(User.user_id == "plain-user"))).scalar_one()
    assert user.spend == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_aggregate_uniqueness_survives_postgres_null_semantics(async_db: AsyncSession) -> None:
    """Two aggregate ceilings on one scope are refused.

    A single UNIQUE over ``(scope_type, scope_id, provider_key_id)`` would let
    both in on PostgreSQL, where NULLs never compare equal. The partial index
    over the identity alone is what makes this fail.
    """
    scope_id = str(uuid.uuid4())
    async_db.add(await _scoped(async_db, scope_type="workspace", scope_id=scope_id, max_budget=1.0))
    await async_db.commit()

    async_db.add(await _scoped(async_db, scope_type="workspace", scope_id=scope_id, max_budget=2.0))
    with pytest.raises(IntegrityError):
        await async_db.commit()
    await async_db.rollback()


def _a_workspace_id(client: Any, headers: dict[str, str]) -> str:
    """A real workspace to hang a ceiling on.

    The scope ids in these API tests have to name rows that exist, because
    create refuses one that does not. The deployment provisions a default
    workspace on the first tenancy request, which is what this reads.
    """
    listed = client.get("/v1/workspaces", headers=headers)
    assert listed.status_code == 200, listed.text
    return str(listed.json()["data"][0]["id"])


def _a_budget_id(
    client: Any,
    headers: dict[str, str],
    *,
    max_budget: float | None = None,
    budget_duration_sec: int | None = None,
    reset_alignment: str | None = None,
) -> str:
    """A budget for a ceiling to enforce, returned by id.

    A ceiling carries no limit of its own, so every case that used to say
    ``max_budget`` in a scoped-budget body now names one of these.
    """
    made = client.post(
        "/v1/budgets",
        json={
            "max_budget": max_budget,
            "budget_duration_sec": budget_duration_sec,
            "reset_alignment": reset_alignment,
        },
        headers=headers,
    )
    assert made.status_code == 200, made.text
    return str(made.json()["budget_id"])


def test_management_surface_round_trip(client: Any, master_key_header: dict[str, str]) -> None:
    """Create, read, list, update and delete one scoped budget over the API."""
    workspace_id = _a_workspace_id(client, master_key_header)
    daily = _a_budget_id(client, master_key_header, max_budget=25.0, budget_duration_sec=86400)
    created = client.post(
        "/v1/scoped-budgets",
        json={
            "scope_type": "workspace",
            "scope_id": workspace_id,
            "name": "Workspace cap",
            "budget_id": daily,
        },
        headers=master_key_header,
    )
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["scope_type"] == "workspace"
    assert body["current_spend"] == 0.0
    assert body["period_end"] is not None
    budget_id = body["id"]

    duplicate = client.post(
        "/v1/scoped-budgets",
        json={"scope_type": "workspace", "scope_id": workspace_id, "budget_id": daily},
        headers=master_key_header,
    )
    assert duplicate.status_code == 409

    narrowed = client.post(
        "/v1/scoped-budgets",
        json={
            "scope_type": "workspace",
            "scope_id": workspace_id,
            "provider_key_id": "openai",
            "budget_id": _a_budget_id(client, master_key_header, max_budget=5.0),
        },
        headers=master_key_header,
    )
    assert narrowed.status_code == 200

    listed = client.get(f"/v1/scoped-budgets?scope_type=workspace&scope_id={workspace_id}", headers=master_key_header)
    assert listed.status_code == 200
    assert len(listed.json()) == 2

    # Changing what a ceiling allows is naming a different budget, since the
    # figure is the budget's and not the ceiling's.
    bigger = _a_budget_id(client, master_key_header, max_budget=40.0, budget_duration_sec=86400)
    updated = client.patch(
        f"/v1/scoped-budgets/{budget_id}",
        json={"budget_id": bigger, "name": None},
        headers=master_key_header,
    )
    assert updated.status_code == 200
    assert updated.json()["max_budget"] == 40.0
    assert updated.json()["budget_id"] == bigger
    assert updated.json()["name"] is None

    assert client.delete(f"/v1/scoped-budgets/{budget_id}", headers=master_key_header).status_code == 204
    assert client.get(f"/v1/scoped-budgets/{budget_id}", headers=master_key_header).status_code == 404


def test_a_budget_can_be_relaxed_back_to_the_states_creation_allows(
    client: Any,
    master_key_header: dict[str, str],
) -> None:
    """Null is a state ``POST`` can write, so ``PATCH`` has to be able to reach it.

    A null ``max_budget`` is a budget that meters and admits everything; a null
    ``budget_duration_sec`` is one that never resets. Both are creatable, so
    testing the value rather than whether the field was sent would make a cadence
    addable and never removable. On ``/v1/budgets`` now, with the cadence.

    ``max_budget`` is the exception and stays value-tested, which is pre-existing
    behavior this change does not touch: clearing a limit is still done by
    creating a budget without one.
    """
    created = client.post(
        "/v1/budgets",
        json={"max_budget": 10.0, "budget_duration_sec": 86400},
        headers=master_key_header,
    ).json()

    cleared = client.patch(
        f"/v1/budgets/{created['budget_id']}",
        json={"budget_duration_sec": None},
        headers=master_key_header,
    )
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["budget_duration_sec"] is None
    assert cleared.json()["reset_alignment"] is None

    # Naming only the alignment is refused rather than silently clearing a
    # duration the caller did not mention.
    half_switched = client.patch(
        f"/v1/budgets/{created['budget_id']}",
        json={"budget_duration_sec": 3600},
        headers=master_key_header,
    )
    assert half_switched.status_code == 200
    conflicting = client.patch(
        f"/v1/budgets/{created['budget_id']}",
        json={"reset_alignment": "calendar_day"},
        headers=master_key_header,
    )
    assert conflicting.status_code == 400, conflicting.text

    # An omitted field is still "leave it alone", which is the half that already
    # worked and must keep working.
    renamed = client.patch(
        f"/v1/budgets/{created['budget_id']}",
        json={"name": "Metering only"},
        headers=master_key_header,
    )
    assert renamed.status_code == 200
    assert renamed.json()["budget_duration_sec"] == 3600


def test_a_ceiling_on_a_scope_that_does_not_exist_is_refused(
    client: Any,
    master_key_header: dict[str, str],
) -> None:
    """A typo must not answer 200 and then quietly enforce nothing.

    Resolution matches a ceiling by id, so one naming a workspace that does not
    exist is created, listed, and never applied, with nothing anywhere to say
    so. That is what a mis-mapped id in a bulk import produces, and it fails in
    the permissive direction. ``POST /v1/keys`` already refuses an unknown
    workspace; this matches it.
    """
    missing = client.post(
        "/v1/scoped-budgets",
        json={
            "scope_type": "workspace",
            "scope_id": str(uuid.uuid4()),
            "budget_id": _a_budget_id(client, master_key_header, max_budget=5.0),
        },
        headers=master_key_header,
    )

    assert missing.status_code == 404, missing.text
    assert "not found" in missing.json()["detail"]

    # Not a UUID at all is the same answer, not a 500 and not a stored row.
    malformed = client.post(
        "/v1/scoped-budgets",
        json={
            "scope_type": "organization",
            "scope_id": "not-a-uuid-at-all",
            "budget_id": _a_budget_id(client, master_key_header, max_budget=5.0),
        },
        headers=master_key_header,
    )

    assert malformed.status_code == 404, malformed.text
    assert client.get("/v1/scoped-budgets", headers=master_key_header).json() == []


def test_a_calendar_aligned_ceiling_opens_on_its_boundary(
    client: Any,
    master_key_header: dict[str, str],
) -> None:
    """Create writes the window rather than waiting for first spend, and for an
    aligned ceiling that window is the calendar one it was created in."""
    workspace_id = _a_workspace_id(client, master_key_header)
    monthly = _a_budget_id(client, master_key_header, max_budget=500.0, reset_alignment="calendar_month")
    before = datetime.now(UTC)
    created = client.post(
        "/v1/scoped-budgets",
        json={"scope_type": "workspace", "scope_id": workspace_id, "budget_id": monthly},
        headers=master_key_header,
    )
    after = datetime.now(UTC)

    assert created.status_code == 200, created.text
    body = created.json()
    assert body["reset_alignment"] == "calendar_month"
    assert body["budget_duration_sec"] is None
    period_start = datetime.fromisoformat(body["period_start"])
    assert period_start in {_first_of_month(before), _first_of_month(after)}
    assert datetime.fromisoformat(body["period_end"]) == _first_of_next_month(period_start)


def test_pointing_a_ceiling_at_another_budget_retimes_it(
    client: Any,
    master_key_header: dict[str, str],
) -> None:
    """Changing what a ceiling allows is naming a different budget.

    The window restarts from now rather than re-deriving an end from a
    ``period_start`` belonging to the old budget's cadence, which is what the
    ceiling used to do when it carried the cadence itself.
    """
    workspace_id = _a_workspace_id(client, master_key_header)
    rolling = _a_budget_id(client, master_key_header, max_budget=10.0, budget_duration_sec=86400)
    created = client.post(
        "/v1/scoped-budgets",
        json={"scope_type": "workspace", "scope_id": workspace_id, "budget_id": rolling},
        headers=master_key_header,
    ).json()
    assert created["budget_duration_sec"] == 86400

    aligned = _a_budget_id(client, master_key_header, max_budget=10.0, reset_alignment="calendar_day")
    before = datetime.now(UTC)
    switched = client.patch(
        f"/v1/scoped-budgets/{created['id']}",
        json={"budget_id": aligned},
        headers=master_key_header,
    )
    after = datetime.now(UTC)
    assert switched.status_code == 200, switched.text
    assert switched.json()["budget_duration_sec"] is None
    assert switched.json()["reset_alignment"] == "calendar_day"
    period_start = datetime.fromisoformat(switched.json()["period_start"])
    assert period_start in {_midnight(before), _midnight(after)}
    assert datetime.fromisoformat(switched.json()["period_end"]) == period_start + timedelta(days=1)

    # A budget that does not exist is refused rather than leaving the ceiling
    # naming nothing.
    missing = client.patch(
        f"/v1/scoped-budgets/{created['id']}",
        json={"budget_id": str(uuid.uuid4())},
        headers=master_key_header,
    )
    assert missing.status_code == 404, missing.text


def test_a_budget_cannot_be_created_with_both_kinds_of_period(
    client: Any,
    master_key_header: dict[str, str],
) -> None:
    """The state the table's CHECK refuses is answered as a request error, not a
    database error.

    On ``/v1/budgets`` now, because that is where a period lives.
    """
    response = client.post(
        "/v1/budgets",
        json={"max_budget": 10.0, "budget_duration_sec": 86400, "reset_alignment": "calendar_month"},
        headers=master_key_header,
    )

    assert response.status_code == 400, response.text
    assert "not both" in response.json()["detail"]


def test_an_unknown_reset_alignment_is_refused(client: Any, master_key_header: dict[str, str]) -> None:
    """The alignment vocabulary is published in the schema, so an unknown one is
    a 422 and never reaches a row nothing can roll."""
    workspace_id = _a_workspace_id(client, master_key_header)
    response = client.post(
        "/v1/scoped-budgets",
        json={"scope_type": "workspace", "scope_id": workspace_id, "reset_alignment": "calendar_quarter"},
        headers=master_key_header,
    )
    assert response.status_code == 422


def test_management_surface_requires_the_master_key(client: Any, api_key_header: dict[str, str]) -> None:
    """A plain API key may not read or write the ceilings that bind it."""
    assert client.get("/v1/scoped-budgets", headers=api_key_header).status_code == 401
    assert (
        client.post(
            "/v1/scoped-budgets",
            json={"scope_type": "workspace", "scope_id": "ws-1", "max_budget": 1.0},
            headers=api_key_header,
        ).status_code
        == 401
    )


def test_unknown_scope_type_is_refused(client: Any, master_key_header: dict[str, str]) -> None:
    """The scope vocabulary is published in the schema, so an unknown one is a 422."""
    response = client.post(
        "/v1/scoped-budgets",
        json={
            "scope_type": "team",
            "scope_id": "ws-1",
            "budget_id": _a_budget_id(client, master_key_header, max_budget=1.0),
        },
        headers=master_key_header,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_token_ceiling_holds_the_estimate_and_records_the_measured_total(
    async_db: AsyncSession, tenancy: Fixture
) -> None:
    """A token cap holds the request's upper bound, then settles at what it really used."""
    cap = await _scoped(
        async_db,
        scope_type="workspace",
        scope_id=str(tenancy.workspace_id),
        max_budget=None,
        token_limit=10_000,
    )
    async_db.add(cap)
    await async_db.commit()

    handle = await reserve_budget(
        async_db, tenancy.user_id, 0.0, estimated_tokens=4_000, scope=tenancy.scope()
    )

    assert handle.token_estimate == 4_000
    assert await _token_counters(async_db, cap.id) == (0, 4_000)

    await reconcile_reservation(async_db, handle, 0.0, actual_tokens=1_200)

    # The estimate goes back and the measured figure is what the next request is
    # gated against, so an over-estimate costs the ceiling nothing.
    assert await _token_counters(async_db, cap.id) == (1_200, 0)


@pytest.mark.asyncio
async def test_token_ceiling_refuses_a_request_that_would_exceed_it(
    async_db: AsyncSession, tenancy: Fixture
) -> None:
    """A hold larger than the remaining token headroom is refused, and nothing is held."""
    cap = await _scoped(
        async_db,
        scope_type="workspace",
        scope_id=str(tenancy.workspace_id),
        max_budget=None,
        token_limit=1_000,
    )
    async_db.add(cap)
    await async_db.commit()

    with pytest.raises(HTTPException) as refusal:
        await reserve_budget(async_db, tenancy.user_id, 0.0, estimated_tokens=1_001, scope=tenancy.scope())

    assert refusal.value.status_code == 403
    assert await _token_counters(async_db, cap.id) == (0, 0)


@pytest.mark.asyncio
async def test_an_exhausted_token_ceiling_refuses_even_a_request_that_estimates_nothing(
    async_db: AsyncSession, tenancy: Fixture
) -> None:
    """The strict comparison is what stops a path with no token estimate slipping through.

    Embeddings and the other pass-through endpoints hold no token estimate, so
    without it a cap already at its limit would keep admitting them.
    """
    cap = await _scoped(
        async_db,
        scope_type="workspace",
        scope_id=str(tenancy.workspace_id),
        max_budget=None,
        token_limit=500,
    )
    cap.current_tokens = 500
    async_db.add(cap)
    await async_db.commit()

    with pytest.raises(HTTPException) as refusal:
        await reserve_budget(async_db, tenancy.user_id, 0.0, estimated_tokens=0, scope=tenancy.scope())

    assert refusal.value.status_code == 403


@pytest.mark.asyncio
async def test_request_ceiling_counts_one_per_reservation(async_db: AsyncSession, tenancy: Fixture) -> None:
    """Two requests fit a cap of two; the third is refused."""
    cap = await _scoped(
        async_db,
        scope_type="workspace",
        scope_id=str(tenancy.workspace_id),
        max_budget=None,
        request_limit=2,
    )
    async_db.add(cap)
    await async_db.commit()

    for _ in range(2):
        handle = await reserve_budget(async_db, tenancy.user_id, 0.0, scope=tenancy.scope())
        await reconcile_reservation(async_db, handle, 0.0)

    assert await _request_counters(async_db, cap.id) == (2, 0)

    with pytest.raises(HTTPException) as refusal:
        await reserve_budget(async_db, tenancy.user_id, 0.0, scope=tenancy.scope())

    assert refusal.value.status_code == 403


@pytest.mark.asyncio
async def test_a_refunded_request_gives_back_every_axis(async_db: AsyncSession, tenancy: Fixture) -> None:
    """A provider failure releases the dollar, token and request holds together."""
    cap = await _scoped(
        async_db,
        scope_type="workspace",
        scope_id=str(tenancy.workspace_id),
        max_budget=10.0,
        token_limit=10_000,
        request_limit=5,
    )
    async_db.add(cap)
    await async_db.commit()

    handle = await reserve_budget(
        async_db, tenancy.user_id, 1.0, estimated_tokens=2_000, scope=tenancy.scope()
    )
    assert await _counters(async_db, cap.id) == (0.0, 1.0)
    assert await _token_counters(async_db, cap.id) == (0, 2_000)
    assert await _request_counters(async_db, cap.id) == (0, 1)

    await refund_reservation(async_db, handle)

    assert await _counters(async_db, cap.id) == (0.0, 0.0)
    assert await _token_counters(async_db, cap.id) == (0, 0)
    assert await _request_counters(async_db, cap.id) == (0, 0)


@pytest.mark.asyncio
async def test_a_rolled_period_zeroes_every_axis_and_leaves_the_holds(
    async_db: AsyncSession, tenancy: Fixture
) -> None:
    """An expired window starts fresh on all three counters, so a spent cap admits again."""
    now = datetime.now(UTC)
    cap = await _scoped(
        async_db,
        scope_type="workspace",
        scope_id=str(tenancy.workspace_id),
        max_budget=None,
        token_limit=1_000,
        request_limit=1,
        budget_duration_sec=3600,
        period_start=now - timedelta(hours=2),
        period_end=now - timedelta(hours=1),
    )
    cap.current_tokens = 1_000
    cap.current_requests = 1
    cap.reserved_tokens = 7
    async_db.add(cap)
    await async_db.commit()

    handle = await reserve_budget(async_db, tenancy.user_id, 0.0, estimated_tokens=10, scope=tenancy.scope())

    assert handle.scoped_budget_ids == (cap.id,)
    current_tokens, reserved_tokens = await _token_counters(async_db, cap.id)
    assert current_tokens == 0
    # The roll left the pre-existing hold alone and this request added its own.
    assert reserved_tokens == 17
    assert await _request_counters(async_db, cap.id) == (0, 1)
