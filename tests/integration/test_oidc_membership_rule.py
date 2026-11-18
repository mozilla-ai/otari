"""The rule deciding which organization an SSO identity is admitted to."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.identity import EmailAssertion, OIDCConnection, ProviderIdentity, oidc_identity_provider
from gateway.models.tenancy import DOMAIN_PROOF_TTL, OrganizationDomain
from gateway.repositories.identity import OIDCConnectionRepository
from gateway.repositories.tenancy import OrganizationDomainRepository
from gateway.services.identity._membership import organization_for_sso_identity

from .tenancy_helpers import create_organization

pytestmark = pytest.mark.asyncio

_ISSUER = "https://login.example.com"


async def _add_connection(
    db: AsyncSession,
    organization_id: uuid.UUID,
    *,
    issuer: str = _ISSUER,
    enabled: bool = True,
) -> OIDCConnection:
    uow = UnitOfWork(db)
    async with uow:
        return await OIDCConnectionRepository(uow).add(
            organization_id=organization_id,
            issuer=issuer,
            resolved_issuer=issuer,
            client_id="a-client-id",
            encrypted_client_secret="encrypted-secret",
            client_secret_last4="cret",
            display_name="Example IdP",
            discovery_document={"issuer": issuer},
            discovered_at=datetime.now(UTC),
            enabled=enabled,
        )


async def _add_claim(
    db: AsyncSession,
    organization_id: uuid.UUID,
    *,
    domain: str,
    enabled: bool = True,
    verified_at: datetime | None,
) -> OrganizationDomain:
    row = await OrganizationDomainRepository(db).create_domain(
        organization_id=organization_id,
        domain=domain,
        default_role="member",
        enabled=enabled,
        verification_token="tok" + uuid.uuid4().hex[:16],
    )
    row.verified_at = verified_at
    db.add(row)
    await db.flush()
    return row


def _identity(*, issuer: str = _ISSUER, email: str | None) -> ProviderIdentity:
    return ProviderIdentity(
        provider=oidc_identity_provider(issuer),
        subject="subject-1",
        email=email,
        full_name="Some Person",
        email_verified=EmailAssertion.VERIFIED,
    )


async def _admit(
    db: AsyncSession,
    external: ProviderIdentity,
    *,
    now: datetime | None = None,
) -> OIDCConnection | None:
    uow = UnitOfWork(db)
    async with uow:
        return await organization_for_sso_identity(
            external,
            connections=OIDCConnectionRepository(uow),
            domains=OrganizationDomainRepository(uow),
            now=now if now is not None else datetime.now(UTC),
        )


async def test_a_connection_alone_admits_nobody(async_db: AsyncSession) -> None:
    """The organization has a connection and no verified domain."""
    organization = await create_organization(async_db, slug="acme")
    await _add_connection(async_db, organization.id)

    identity = _identity(email="person@acme.example")

    assert await _admit(async_db, identity) is None


async def test_a_verified_domain_alone_admits_nobody(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    await _add_claim(async_db, organization.id, domain="acme.example", verified_at=datetime.now(UTC))

    identity = _identity(email="person@acme.example")

    assert await _admit(async_db, identity) is None


async def test_two_organizations_on_one_issuer_are_told_apart_by_their_domains(async_db: AsyncSession) -> None:
    acme = await create_organization(async_db, slug="acme")
    globex = await create_organization(async_db, slug="globex")
    acme_connection = await _add_connection(async_db, acme.id)
    globex_connection = await _add_connection(async_db, globex.id)
    await _add_claim(async_db, acme.id, domain="acme.example", verified_at=datetime.now(UTC))
    await _add_claim(async_db, globex.id, domain="globex.example", verified_at=datetime.now(UTC))

    acme_admitted = await _admit(async_db, _identity(email="person@acme.example"))
    globex_admitted = await _admit(async_db, _identity(email="person@globex.example"))

    assert acme_admitted is not None
    assert acme_admitted.id == acme_connection.id
    assert globex_admitted is not None
    assert globex_admitted.id == globex_connection.id


async def test_an_expired_proof_admits_nobody(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    await _add_connection(async_db, organization.id)
    verified_at = datetime.now(UTC) - timedelta(days=1)
    await _add_claim(async_db, organization.id, domain="acme.example", verified_at=verified_at)

    identity = _identity(email="person@acme.example")

    just_before_ttl = verified_at + DOMAIN_PROOF_TTL - timedelta(seconds=1)
    assert await _admit(async_db, identity, now=just_before_ttl) is not None

    just_after_ttl = verified_at + DOMAIN_PROOF_TTL + timedelta(seconds=1)
    assert await _admit(async_db, identity, now=just_after_ttl) is None


async def test_a_disabled_claim_admits_nobody(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    await _add_connection(async_db, organization.id)
    await _add_claim(
        async_db,
        organization.id,
        domain="acme.example",
        enabled=False,
        verified_at=datetime.now(UTC),
    )

    identity = _identity(email="person@acme.example")

    assert await _admit(async_db, identity) is None


async def test_an_identity_with_no_email_claim_admits_nobody(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    await _add_connection(async_db, organization.id)
    await _add_claim(async_db, organization.id, domain="acme.example", verified_at=datetime.now(UTC))

    identity = _identity(email=None)

    assert await _admit(async_db, identity) is None


async def test_an_email_with_no_at_sign_admits_nobody(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    await _add_connection(async_db, organization.id)
    await _add_claim(async_db, organization.id, domain="acme.example", verified_at=datetime.now(UTC))

    identity = _identity(email="not-an-email")

    assert await _admit(async_db, identity) is None


async def test_a_mixed_case_domain_with_a_trailing_dot_matches(async_db: AsyncSession) -> None:
    """A domain differing only in letter case or a trailing dot matches the same claim."""
    organization = await create_organization(async_db, slug="acme")
    connection = await _add_connection(async_db, organization.id)
    await _add_claim(async_db, organization.id, domain="acme.example", verified_at=datetime.now(UTC))

    identity = _identity(email="Person@ACME.example.")

    admitted = await _admit(async_db, identity)

    assert admitted is not None
    assert admitted.id == connection.id
