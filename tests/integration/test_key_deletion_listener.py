"""The budgets domain reacts to an API key's deletion through a listener (#1529).

The direct test drives the listener the way the api-keys domain calls it, so the
contract is exercised without the service that calls it. The route test proves the
whole path: a key's ``api_token`` ceiling is swept when the key is deleted, rather
than stranded where it would block its budget's deletion for good.
"""

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_api_key_service
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.api_keys import APIKey
from gateway.models.budgets import SCOPE_API_TOKEN, ScopedBudget
from gateway.repositories.budgets import BudgetRepositories
from gateway.services.api_keys import ApiKeyDeletionListener
from gateway.services.budgets import BudgetKeyDeletionListener

from .tenancy_helpers import create_budget, create_member, create_organization, create_workspace

pytestmark = pytest.mark.asyncio


def _as_listener(listener: BudgetKeyDeletionListener) -> ApiKeyDeletionListener:
    """Checked by mypy only: the budget listener satisfies the api-keys contract structurally."""
    return listener


async def _api_token_ceilings(db: AsyncSession, key_id: str) -> list[ScopedBudget]:
    stmt = select(ScopedBudget).where(
        ScopedBudget.scope_type == SCOPE_API_TOKEN, ScopedBudget.scope_id == key_id
    )
    return list((await db.execute(stmt)).scalars().all())


async def test_key_deleted_deletes_the_api_token_ceilings(async_db: AsyncSession) -> None:
    key_id = f"key-{uuid.uuid4().hex}"
    budget_id = await create_budget(async_db, max_budget=25.0)
    async_db.add(ScopedBudget(scope_type=SCOPE_API_TOKEN, scope_id=key_id, budget_id=budget_id))
    await async_db.flush()
    assert len(await _api_token_ceilings(async_db, key_id)) == 1

    uow = UnitOfWork(async_db)
    listener = BudgetKeyDeletionListener(BudgetRepositories.on(uow))
    async with uow:
        await listener.key_deleted(APIKey(id=key_id, key_hash=f"hash-{key_id}", workspace_id=uuid.uuid4()))

    assert await _api_token_ceilings(async_db, key_id) == []


async def test_a_key_without_a_ceiling_is_a_no_op(async_db: AsyncSession) -> None:
    """A key that was never capped leaves nothing behind and raises nothing."""
    uow = UnitOfWork(async_db)
    listener = BudgetKeyDeletionListener(BudgetRepositories.on(uow))
    async with uow:
        await listener.key_deleted(APIKey(id="key-uncapped", key_hash="hash-uncapped", workspace_id=uuid.uuid4()))


async def test_deleting_a_key_through_the_service_sweeps_its_ceiling(async_db: AsyncSession) -> None:
    """The whole path: the key goes, its ceiling goes with it, and the budget it named can be deleted.

    Built the way the composition root builds it, so the budgets listener the request path wires is the one
    exercised, rather than one the test assembles.
    """
    organization = await create_organization(async_db, slug=f"org-{uuid.uuid4().hex[:8]}")
    owner = await create_member(async_db, organization, role="owner", full_name="Owner")
    workspace = await create_workspace(async_db, organization, name="Engineering", owner=owner)
    budget_id = await create_budget(async_db, max_budget=25.0)
    key = APIKey(id=f"key-{uuid.uuid4().hex}", key_hash=f"hash-{uuid.uuid4().hex}", workspace_id=workspace.id)
    async_db.add(key)
    async_db.add(ScopedBudget(scope_type=SCOPE_API_TOKEN, scope_id=key.id, budget_id=budget_id))
    await async_db.commit()

    await get_api_key_service(UnitOfWork(async_db)).delete_key(key)

    assert await _api_token_ceilings(async_db, key.id) == []
    # The budget's RESTRICT reference is gone with the ceiling, so the budget is deletable again.
    repositories = BudgetRepositories.on(UnitOfWork(async_db))
    assert await repositories.ceilings.has_ceiling(SCOPE_API_TOKEN, key.id, None) is False
