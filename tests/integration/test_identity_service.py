"""OAuth sign-in through the identity service and the core identity provider policy."""

import functools

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.adapters.identity_provider_adapter import DeploymentIdentityProviderAdapter
from gateway.models.identity import OAuthIdentityLink
from gateway.ports.identity_provider_port import EmailAssertion
from gateway.repositories.identity import OAuthIdentityLinkRepository
from gateway.repositories.tenancy import UserRepository
from gateway.services.identity import IdentityService
from gateway.services.oauth_service import OAuthIdentity
from gateway.services.tenancy import OrganizationService, forget_unproven_access

from .tenancy_helpers import membership_writes

pytestmark = pytest.mark.asyncio


def _service(db: AsyncSession, *, open_signup: bool) -> IdentityService:
    writes = membership_writes(db)
    adapter = DeploymentIdentityProviderAdapter(
        UserRepository(db),
        OrganizationService(db, **writes),
        functools.partial(forget_unproven_access, db),
        open_signup=open_signup,
    )
    return IdentityService(
        writes["uow"],
        identity_provider=adapter,
        links=OAuthIdentityLinkRepository(writes["uow"]),
        users=UserRepository(writes["uow"]),
    )


def _verified(provider: str, email: str) -> OAuthIdentity:
    return OAuthIdentity(
        provider=provider,
        subject="provider-subject",
        email=email,
        full_name=None,
        email_verified=EmailAssertion.VERIFIED,
    )


async def test_open_signup_links_the_account_it_registers(async_db: AsyncSession) -> None:
    resolution = await _service(async_db, open_signup=True).sign_in_with_oauth(
        _verified("google", "nova@example.com")
    )

    link = await async_db.get(OAuthIdentityLink, ("google", "provider-subject"))
    assert link is not None
    assert link.user_id == resolution.identity.id
