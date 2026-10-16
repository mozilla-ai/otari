"""How much invitation mail an organization may send and an address may receive."""

import uuid

import pytest

from gateway.adapters.rate_limit_store_adapter import InMemoryRateLimitStore
from gateway.services.organizations import InvitationMailLimit

pytestmark = pytest.mark.asyncio


async def test_an_address_is_limited_across_organizations() -> None:
    limit = InvitationMailLimit(InMemoryRateLimitStore(), per_organization=None, per_address=2)

    assert await limit.admit(uuid.uuid4(), "ada@example.com")
    assert await limit.admit(uuid.uuid4(), "ada@example.com")
    assert not await limit.admit(uuid.uuid4(), "ada@example.com")
    assert await limit.admit(uuid.uuid4(), "bob@example.com")


async def test_an_organization_is_limited_across_addresses() -> None:
    organization = uuid.uuid4()
    limit = InvitationMailLimit(InMemoryRateLimitStore(), per_organization=2, per_address=None)

    assert await limit.admit(organization, "a@example.com")
    assert await limit.admit(organization, "b@example.com")
    assert not await limit.admit(organization, "c@example.com")
    assert await limit.admit(uuid.uuid4(), "c@example.com")


async def test_a_refusal_by_the_organization_limit_does_not_spend_the_address() -> None:
    full = uuid.uuid4()
    limit = InvitationMailLimit(InMemoryRateLimitStore(), per_organization=1, per_address=1)
    assert await limit.admit(full, "a@example.com")

    assert not await limit.admit(full, "b@example.com")
    # Refused above, so b@ still has its one email for another organization.
    assert await limit.admit(uuid.uuid4(), "b@example.com")


async def test_no_store_or_no_limits_admits_everything() -> None:
    organization = uuid.uuid4()
    unbound = InvitationMailLimit(None, per_organization=1, per_address=1)
    disabled = InvitationMailLimit(InMemoryRateLimitStore(), per_organization=None, per_address=None)

    for _ in range(3):
        assert await unbound.admit(organization, "a@example.com")
        assert await disabled.admit(organization, "a@example.com")
