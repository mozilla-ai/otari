"""Enforcement and management of the tenancy-scoped budget ceilings.

The legacy per-user path (``budgets`` + ``users.spend``/``users.reserved``) is
covered by ``test_budget_race_condition.py``; the last test here asserts that it
is untouched by a request that also passes scoped ceilings.
"""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import select
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


def _scoped(
    *,
    scope_type: str,
    scope_id: str,
    max_budget: float | None,
    provider_key_id: str | None = None,
) -> ScopedBudget:
    return ScopedBudget(
        scope_type=scope_type,
        scope_id=scope_id,
        max_budget=max_budget,
        provider_key_id=provider_key_id,
    )


@pytest_asyncio.fixture
async def tenancy(async_db: AsyncSession) -> Fixture:
    return await _build_tenancy(async_db, "acme")


@pytest.mark.asyncio
async def test_request_passing_two_ceilings_holds_on_both(async_db: AsyncSession, tenancy: Fixture) -> None:
    """An organization cap and a member cap both admit the estimate and both hold it."""
    org_cap = _scoped(scope_type="organization", scope_id=str(tenancy.organization_id), max_budget=100.0)
    member_cap = _scoped(
        scope_type="workspace_member", scope_id=str(tenancy.workspace_member_id), max_budget=10.0
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
async def test_tighter_ceiling_rejects_and_compensates_the_looser_one(
    async_db: AsyncSession, tenancy: Fixture
) -> None:
    """The organization cap admits the estimate, the member cap refuses, and the
    hold already taken on the organization is given back."""
    org_cap = _scoped(scope_type="organization", scope_id=str(tenancy.organization_id), max_budget=100.0)
    member_cap = _scoped(scope_type="workspace_member", scope_id=str(tenancy.workspace_member_id), max_budget=1.0)
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
    org_cap = _scoped(scope_type="organization", scope_id=str(tenancy.organization_id), max_budget=100.0)
    async_db.add(org_cap)
    async_db.add(Budget(budget_id="tiny", max_budget=1.0))
    await async_db.commit()

    user = (await async_db.execute(select(User).where(User.user_id == tenancy.user_id))).scalar_one()
    user.budget_id = "tiny"
    user.spend = 1.0
    await async_db.commit()

    with pytest.raises(HTTPException) as exc_info:
        await reserve_budget(async_db, tenancy.user_id, 0.5, scope=tenancy.scope())
    assert exc_info.value.status_code == 403

    assert await _counters(async_db, org_cap.id) == (0.0, 0.0)


@pytest.mark.asyncio
async def test_provider_narrowed_cap_binds_only_that_provider(async_db: AsyncSession, tenancy: Fixture) -> None:
    """A cap narrowed to one provider is invisible to a request on another."""
    narrowed = _scoped(
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
    aggregate = _scoped(scope_type="workspace", scope_id=str(tenancy.workspace_id), max_budget=50.0)
    narrowed = _scoped(
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
    cap = ScopedBudget(
        scope_type="workspace",
        scope_id=str(tenancy.workspace_id),
        max_budget=10.0,
        current_spend=9.5,
        budget_duration_sec=3600,
        period_start=now - timedelta(seconds=7200),
        period_end=now - timedelta(seconds=3600),
    )
    async_db.add(cap)
    await async_db.commit()

    handle = await reserve_budget(async_db, tenancy.user_id, 5.0, scope=tenancy.scope())
    assert handle.scoped_budget_ids == (cap.id,)
    assert await _counters(async_db, cap.id) == (0.0, 5.0)

    refreshed = (await async_db.execute(select(ScopedBudget).where(ScopedBudget.id == cap.id))).scalar_one()
    await async_db.refresh(refreshed)
    assert refreshed.period_end is not None
    assert refreshed.period_end > now


@pytest.mark.asyncio
async def test_concurrent_reservers_cannot_overspend_a_shared_ceiling(
    async_db: AsyncSession, postgres_url: str, tenancy: Fixture
) -> None:
    """Eight simultaneous reservations against a ceiling with room for four all
    contend on the same conditional UPDATE; exactly four are admitted."""
    cap = _scoped(scope_type="organization", scope_id=str(tenancy.organization_id), max_budget=4.0)
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
    cap = _scoped(scope_type="organization", scope_id=str(tenancy.organization_id), max_budget=100.0)
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
    cap = _scoped(scope_type="organization", scope_id=str(tenancy.organization_id), max_budget=1.0)
    async_db.add(cap)
    await async_db.commit()

    handle = await reserve_budget(async_db, tenancy.user_id, 100.0)
    assert handle.scoped_budgets == ()
    assert await _counters(async_db, cap.id) == (0.0, 0.0)


@pytest.mark.asyncio
async def test_budget_exempt_key_skips_scoped_ceilings(async_db: AsyncSession, tenancy: Fixture) -> None:
    """A key flagged ``exclude_from_budget`` is exempt from the scoped ceilings
    too, matching what the flag already means for the per-user cap."""
    cap = _scoped(scope_type="organization", scope_id=str(tenancy.organization_id), max_budget=1.0)
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

    handle = ReservationHandle(user_id="plain-user", estimate=0.0, reserved=False, strategy="disabled")
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
    from sqlalchemy.exc import IntegrityError

    scope_id = str(uuid.uuid4())
    async_db.add(_scoped(scope_type="workspace", scope_id=scope_id, max_budget=1.0))
    await async_db.commit()

    async_db.add(_scoped(scope_type="workspace", scope_id=scope_id, max_budget=2.0))
    with pytest.raises(IntegrityError):
        await async_db.commit()
    await async_db.rollback()


def test_management_surface_round_trip(client: Any, master_key_header: dict[str, str]) -> None:
    """Create, read, list, update and delete one scoped budget over the API."""
    created = client.post(
        "/v1/scoped-budgets",
        json={
            "scope_type": "workspace",
            "scope_id": "ws-1",
            "name": "Workspace cap",
            "max_budget": 25.0,
            "budget_duration_sec": 86400,
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
        json={"scope_type": "workspace", "scope_id": "ws-1", "max_budget": 5.0},
        headers=master_key_header,
    )
    assert duplicate.status_code == 409

    narrowed = client.post(
        "/v1/scoped-budgets",
        json={"scope_type": "workspace", "scope_id": "ws-1", "provider_key_id": "openai", "max_budget": 5.0},
        headers=master_key_header,
    )
    assert narrowed.status_code == 200

    listed = client.get("/v1/scoped-budgets?scope_type=workspace&scope_id=ws-1", headers=master_key_header)
    assert listed.status_code == 200
    assert len(listed.json()) == 2

    updated = client.patch(
        f"/v1/scoped-budgets/{budget_id}",
        json={"max_budget": 40.0, "name": None},
        headers=master_key_header,
    )
    assert updated.status_code == 200
    assert updated.json()["max_budget"] == 40.0
    assert updated.json()["name"] is None

    assert client.delete(f"/v1/scoped-budgets/{budget_id}", headers=master_key_header).status_code == 204
    assert client.get(f"/v1/scoped-budgets/{budget_id}", headers=master_key_header).status_code == 404


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
        json={"scope_type": "team", "scope_id": "ws-1", "max_budget": 1.0},
        headers=master_key_header,
    )
    assert response.status_code == 422
