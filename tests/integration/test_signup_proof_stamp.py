"""What a signup records about how its address was proved."""

import pytest
from fastapi import BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tenancy import User
from gateway.ports.signup_invite_port import AddressProof
from gateway.repositories.tenancy import OrganizationMemberRepository, UserRepository
from gateway.services.tenancy import OrganizationService
from gateway.services.tenancy.user_service import create_user_for_signup

from .tenancy_helpers import NO_BLOCKED_ADDRESSES, create_organization, membership_listener_on

pytestmark = pytest.mark.asyncio

_EMAIL = "ada@example.com"
_PASSWORD = "a-str0ng-Passphrase!"


def _config() -> GatewayConfig:
    return GatewayConfig(
        mail_from_email="otari@example.com",
        mail_transport="smtp",
        public_base_url="https://otari.example.com",
        smtp_host="smtp.example.com",
    )


async def _signup(session: AsyncSession, *, proof: AddressProof | None) -> User | None:
    organization = await create_organization(session, slug=f"signup-{proof or 'none'}")
    identity = await UserRepository(session).create_local_identity(
        full_name="Ada", active_organization_id=organization.id, email=_EMAIL
    )
    # An unverified identity is claimable on a closed deployment only while somebody still wants it.
    await OrganizationMemberRepository(session).create_membership(
        organization_id=organization.id, user_id=identity.id, role="member"
    )
    await session.flush()
    uow = UnitOfWork(session)
    return await create_user_for_signup(
        session,
        _config(),
        background_tasks=BackgroundTasks(),
        email=_EMAIL,
        password=_PASSWORD,
        uow=uow,
        organizations=OrganizationService(session, membership_listener=membership_listener_on(uow), uow=uow),
        blocked_addresses=NO_BLOCKED_ADDRESSES,
        proof=proof,
    )


async def test_a_mailbox_proof_verifies_the_address_and_vouches_for_nothing(async_db: AsyncSession) -> None:
    claimed = await _signup(async_db, proof=AddressProof.MAILBOX)

    assert claimed is not None
    assert claimed.email_verified_at is not None
    assert claimed.email_vouched_at is None


async def test_a_vouched_proof_is_recorded_as_vouched_for(async_db: AsyncSession) -> None:
    """Domain auto-join refuses a vouched address, so the stamp is what keeps that rule working."""
    claimed = await _signup(async_db, proof=AddressProof.VOUCHED)

    assert claimed is not None
    assert claimed.email_verified_at is not None
    assert claimed.email_vouched_at is not None
