"""An organization's SSO enforcement: its state, the break-glass preconditions, and who it blocks.

``sso_blocks_membership`` is pure and needs no database; its cases are kept here rather than in
``tests/unit`` so the whole layer this task adds reads as one file.
"""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.identity_exceptions import (
    OIDCConnectionNotFoundError,
    OIDCConnectionUnprovenError,
    SsoBreakGlassRequiredError,
)
from gateway.exceptions.organizations_exceptions import NotAuthorizedError
from gateway.models.identity import OIDCConnection
from gateway.models.tenancy import Organization, User
from gateway.repositories.identity import IdentityRepositories
from gateway.repositories.tenancy import OrganizationMemberRepository, UserRepository
from gateway.repositories.tenancy.organization_domain_repository import OrganizationDomainRepository
from gateway.services.identity import OIDCConnectionService
from gateway.services.secret_box import encrypt_secret, generate_secret_key
from gateway.services.tenancy.organization_service import OrganizationService, sso_blocks_membership

from .tenancy_helpers import create_member, create_organization

pytestmark = pytest.mark.asyncio

_ISSUER = "https://sso.acme.example"


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Storing a connection's client secret needs a key configured."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())


def _connection_service(db: AsyncSession) -> OIDCConnectionService:
    uow = UnitOfWork(db)
    return OIDCConnectionService(
        uow,
        IdentityRepositories.on(uow),
        organizations=OrganizationService(db, membership_listener=None),
        domains=OrganizationDomainRepository(uow),
        allow_private_hosts=False,
    )


async def _add_connection(
    db: AsyncSession, organization_id: uuid.UUID, *, first_sign_in_at: datetime | None
) -> OIDCConnection:
    connection = OIDCConnection(
        organization_id=organization_id,
        issuer=_ISSUER,
        resolved_issuer=_ISSUER,
        client_id="a-client-id",
        encrypted_client_secret=encrypt_secret("a-client-secret"),
        client_secret_last4="cret",
        display_name="Acme SSO",
        discovery_document={"issuer": _ISSUER},
        discovered_at=datetime.now(UTC),
        first_sign_in_at=first_sign_in_at,
        enabled=True,
    )
    db.add(connection)
    await db.flush()
    await db.refresh(connection)
    return connection


async def _member_with_email(
    db: AsyncSession, organization: Organization, *, role: str, email: str, full_name: str
) -> User:
    user = await UserRepository(db).create_local_identity(
        email=email, full_name=full_name, active_organization_id=organization.id
    )
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id, user_id=user.id, role=role
    )
    return user


async def _reloaded(db: AsyncSession, organization: Organization) -> Organization:
    """Reload every column of ``organization`` now, inside this ``await``.

    NOTE: A write under test may have rolled the session back, which expires every attribute on
    ``organization``. A plain attribute access after that reloads lazily outside any ``await``,
    which asyncio SQLAlchemy cannot do.
    """
    await db.refresh(organization)
    return organization


# =============================================================================
# The predicate
# =============================================================================


async def test_enforcement_blocks_a_member_with_no_link_for_this_connection(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    organization.sso_required = True
    organization.sso_break_glass_waived = True
    async_db.add(organization)
    await async_db.flush()
    member = await _member_with_email(
        async_db, organization, role="member", email="mo@acme.example", full_name="Mo Member"
    )

    assert sso_blocks_membership(organization=organization, user=member, linked_providers=frozenset()) is True


async def test_enforcement_does_not_block_the_break_glass_owner(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    organization.sso_required = True
    organization.sso_break_glass_user_id = owner.id
    async_db.add(organization)
    await async_db.flush()

    assert sso_blocks_membership(organization=organization, user=owner, linked_providers=frozenset()) is False


async def test_enforcement_does_not_block_the_master_key_operator(async_db: AsyncSession) -> None:
    """The deployment's operator reaches an enforcing organization; an ordinary member with no link does not."""
    organization = await create_organization(async_db, slug="acme")
    organization.sso_required = True
    organization.sso_break_glass_waived = True
    async_db.add(organization)
    await async_db.flush()
    operator = await create_member(async_db, organization, role="member", full_name="The Operator")
    member = await _member_with_email(
        async_db, organization, role="member", email="mo@acme.example", full_name="Mo Member"
    )

    assert sso_blocks_membership(organization=organization, user=operator, linked_providers=frozenset()) is False
    assert sso_blocks_membership(organization=organization, user=member, linked_providers=frozenset()) is True


async def test_enforcement_does_not_block_a_member_linked_to_this_connection(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    organization.sso_required = True
    organization.sso_break_glass_waived = True
    async_db.add(organization)
    await async_db.flush()
    member = await _member_with_email(
        async_db, organization, role="member", email="mo@acme.example", full_name="Mo Member"
    )

    assert (
        sso_blocks_membership(organization=organization, user=member, linked_providers=frozenset({"oidc:acme"}))
        is False
    )


async def test_enforcement_blocks_nobody_while_turned_off(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    member = await _member_with_email(
        async_db, organization, role="member", email="mo@acme.example", full_name="Mo Member"
    )

    assert sso_blocks_membership(organization=organization, user=member, linked_providers=frozenset()) is False


# =============================================================================
# Setting enforcement
# =============================================================================


async def test_enforcement_cannot_be_set_before_one_sign_in_has_succeeded(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=None)
    await async_db.commit()

    with pytest.raises(OIDCConnectionUnprovenError):
        await _connection_service(async_db).set_sso_enforcement_for_user(
            user=owner, required=True, break_glass_user_id=None, break_glass_waived=True
        )
    assert (await _reloaded(async_db, organization)).sso_required is False


async def test_enforcement_is_permitted_once_the_connection_has_signed_somebody_in(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))

    await _connection_service(async_db).set_sso_enforcement_for_user(
        user=owner, required=True, break_glass_user_id=None, break_glass_waived=True
    )

    assert (await _reloaded(async_db, organization)).sso_required is True


async def test_enforcement_cannot_be_set_without_an_owner_exception_or_an_explicit_waiver(
    async_db: AsyncSession,
) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await async_db.commit()

    with pytest.raises(SsoBreakGlassRequiredError):
        await _connection_service(async_db).set_sso_enforcement_for_user(
            user=owner, required=True, break_glass_user_id=None, break_glass_waived=False
        )
    assert (await _reloaded(async_db, organization)).sso_required is False


async def test_enforcement_is_permitted_with_an_explicit_waiver(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))

    await _connection_service(async_db).set_sso_enforcement_for_user(
        user=owner, required=True, break_glass_user_id=None, break_glass_waived=True
    )

    assert (await _reloaded(async_db, organization)).sso_required is True


async def test_enforcement_is_permitted_with_a_break_glass_owner_exception(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))

    await _connection_service(async_db).set_sso_enforcement_for_user(
        user=owner, required=True, break_glass_user_id=owner.id, break_glass_waived=False
    )

    stored = await _reloaded(async_db, organization)
    assert stored.sso_required is True
    assert stored.sso_break_glass_user_id == owner.id


async def test_enforcement_refuses_a_break_glass_account_that_is_not_an_owner(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    member = await _member_with_email(
        async_db, organization, role="member", email="mo@acme.example", full_name="Mo Member"
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await async_db.commit()

    with pytest.raises(SsoBreakGlassRequiredError):
        await _connection_service(async_db).set_sso_enforcement_for_user(
            user=owner, required=True, break_glass_user_id=member.id, break_glass_waived=False
        )
    assert (await _reloaded(async_db, organization)).sso_required is False


async def test_enforcement_cannot_be_set_for_an_organization_with_no_connection(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )

    with pytest.raises(OIDCConnectionNotFoundError):
        await _connection_service(async_db).set_sso_enforcement_for_user(
            user=owner, required=True, break_glass_user_id=None, break_glass_waived=True
        )


async def test_disabling_enforcement_needs_no_precondition(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=None)

    await _connection_service(async_db).set_sso_enforcement_for_user(
        user=owner, required=False, break_glass_user_id=None, break_glass_waived=False
    )

    assert (await _reloaded(async_db, organization)).sso_required is False


@pytest.mark.parametrize("role", ["member", "viewer"])
async def test_enforcement_cannot_be_set_by_a_caller_who_cannot_manage_the_organization(
    async_db: AsyncSession, role: str
) -> None:
    organization = await create_organization(async_db, slug="acme")
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    caller = await _member_with_email(async_db, organization, role=role, email="mo@acme.example", full_name="Mo Member")

    with pytest.raises(NotAuthorizedError):
        await _connection_service(async_db).set_sso_enforcement_for_user(
            user=caller, required=True, break_glass_user_id=None, break_glass_waived=True
        )
    assert (await _reloaded(async_db, organization)).sso_required is False
