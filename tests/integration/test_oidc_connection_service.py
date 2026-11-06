"""An organization's own OIDC connection: discovery, encryption, and the sign-in refresh.

`discover_connection_metadata` is stubbed throughout. It is the single call that leaves the
process, and a test that reached a real issuer would pass or fail on somebody else's server.
One test also patches `release_session`, to record when the connection goes back to the pool.
"""

import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from gateway.core.database import release_session
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.identity_exceptions import (
    OIDCConnectionNotFoundError,
    OIDCConnectionSecretRequiredError,
    OIDCDiscoveryError,
)
from gateway.exceptions.organizations_exceptions import NotAuthorizedError
from gateway.exceptions.shared_exceptions import SecretBoxUnavailableTenancyError
from gateway.models.identity import OIDCConnection
from gateway.models.tenancy import DOMAIN_PROOF_TTL, Organization, User
from gateway.repositories.identity import IdentityRepositories, OIDCConnectionRepository
from gateway.repositories.tenancy.organization_domain_repository import OrganizationDomainRepository
from gateway.schemas.identity import OIDCConnectionRequest
from gateway.services.identity import OIDCConnectionService
from gateway.services.identity._discovery import DiscoveredProvider
from gateway.services.secret_box import decrypt_secret, encrypt_secret, generate_secret_key
from gateway.services.tenancy.organization_service import OrganizationService

from .tenancy_helpers import create_member, create_organization

pytestmark = pytest.mark.asyncio

_SERVICE_MODULE = "gateway.services.identity._connection_service"
_ISSUER = "https://login.example.com"


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Encrypting a client secret on the way in needs a secret key to be set."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())


def _document(issuer: str = _ISSUER) -> dict[str, Any]:
    return {"issuer": issuer, "authorize_url": f"{issuer}/authorize", "token_url": f"{issuer}/token"}


def _stub_discovery(
    monkeypatch: pytest.MonkeyPatch,
    *,
    resolved_issuer: str = _ISSUER,
    document: dict[str, Any] | None = None,
    discovered_at: datetime | None = None,
    calls: list[str] | None = None,
) -> None:
    """Stand in for a successful discovery, recording the issuer each call named."""
    resolved_document = document if document is not None else _document(resolved_issuer)
    resolved_at = discovered_at or datetime.now(UTC)

    async def _discover(issuer: str, *, allow_private_hosts: bool) -> DiscoveredProvider:
        if calls is not None:
            calls.append(issuer)
        return DiscoveredProvider(
            resolved_issuer=resolved_issuer, document=resolved_document, discovered_at=resolved_at
        )

    monkeypatch.setattr(f"{_SERVICE_MODULE}.discover_connection_metadata", _discover)


def _stub_refused_discovery(monkeypatch: pytest.MonkeyPatch, *, calls: list[str] | None = None) -> None:
    """Stand in for a discovery that never resolves to a usable configuration."""

    async def _discover(issuer: str, *, allow_private_hosts: bool) -> DiscoveredProvider:
        if calls is not None:
            calls.append(issuer)
        raise OIDCDiscoveryError

    monkeypatch.setattr(f"{_SERVICE_MODULE}.discover_connection_metadata", _discover)


async def _stored(db: AsyncSession, organization_id: uuid.UUID) -> OIDCConnection | None:
    """Read the organization's connection straight from the repository, bypassing the service."""
    uow = UnitOfWork(db)
    async with uow:
        return await OIDCConnectionRepository(uow).get_for_organization(organization_id)


def _connection_service(db: AsyncSession) -> OIDCConnectionService:
    uow = UnitOfWork(db)
    return OIDCConnectionService(
        uow,
        IdentityRepositories.on(uow),
        organizations=OrganizationService(db, membership_listener=None),
        domains=OrganizationDomainRepository(uow),
        allow_private_hosts=False,
    )


async def _delete_in_another_session(db: AsyncSession, admin: User) -> None:
    """Delete the organization's connection the way a concurrent request would.

    NOTE: A delete issued on *db* expunges the row from the identity map the service under test
    reads through. A concurrent request cannot do that, so this opens an engine of its own.
    """
    other_engine = create_async_engine(db.get_bind().engine.url)
    try:
        async with AsyncSession(other_engine, expire_on_commit=False) as other:
            await _connection_service(other).delete_for_user(user=admin)
    finally:
        await other_engine.dispose()


async def _insert_in_another_session(db: AsyncSession, organization_id: uuid.UUID) -> None:
    """Store the organization's first connection the way a concurrent save would.

    NOTE: Another transaction has to commit the row, or the unique constraint the race turns on
    never sees two of them.
    """
    other_engine = create_async_engine(db.get_bind().engine.url)
    try:
        async with AsyncSession(other_engine, expire_on_commit=False) as other:
            other.add(
                OIDCConnection(
                    organization_id=organization_id,
                    issuer=_ISSUER,
                    resolved_issuer=_ISSUER,
                    client_id="the-client-id-that-won",
                    encrypted_client_secret=encrypt_secret("the-secret-that-won"),
                    client_secret_last4="swon",
                    display_name="The IdP that won",
                    discovery_document=_document(),
                    discovered_at=datetime.now(UTC),
                    default_role="member",
                    enabled=True,
                )
            )
            await other.commit()
    finally:
        await other_engine.dispose()


def _request(**overrides: Any) -> OIDCConnectionRequest:
    fields: dict[str, Any] = {
        "issuer": _ISSUER,
        "client_id": "a-client-id",
        "client_secret": "s3cr3t-client-value",
        "display_name": "Example IdP",
        "default_role": "member",
        "enabled": True,
    }
    fields.update(overrides)
    return OIDCConnectionRequest(**fields)


async def _admin(db: AsyncSession, organization: Organization, *, name: str = "Ada Admin") -> User:
    return await create_member(db, organization, role="admin", full_name=name)


# =============================================================================
# Saving
# =============================================================================


async def test_save_stores_the_discovered_issuer_and_endpoints(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    organization = await create_organization(async_db, slug="acme")
    admin = await _admin(async_db, organization)
    resolved_issuer = "https://sso.acme.example/"
    document = _document(resolved_issuer)
    _stub_discovery(monkeypatch, resolved_issuer=resolved_issuer, document=document)

    public = await _connection_service(async_db).save_for_user(user=admin, request=_request())

    assert public.resolved_issuer == resolved_issuer

    stored = await _stored(async_db, organization.id)
    assert stored is not None
    assert stored.resolved_issuer == resolved_issuer
    assert stored.discovery_document == document


async def test_save_encrypts_the_client_secret_and_returns_only_last4(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    organization = await create_organization(async_db, slug="acme")
    admin = await _admin(async_db, organization)
    _stub_discovery(monkeypatch)

    public = await _connection_service(async_db).save_for_user(
        user=admin, request=_request(client_secret="s3cr3t-client-value")
    )

    assert public.client_secret_last4 == "alue"
    assert "client_secret" not in public.model_dump()
    assert "s3cr3t-client-value" not in public.model_dump_json()

    stored = await _stored(async_db, organization.id)
    assert stored is not None
    assert stored.encrypted_client_secret != "s3cr3t-client-value"
    assert decrypt_secret(stored.encrypted_client_secret) == "s3cr3t-client-value"


async def test_save_stores_nothing_when_discovery_is_refused(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    organization = await create_organization(async_db, slug="acme")
    admin = await _admin(async_db, organization)

    # A first save that never discovers an issuer leaves no row at all.
    _stub_refused_discovery(monkeypatch)
    with pytest.raises(OIDCDiscoveryError):
        await _connection_service(async_db).save_for_user(user=admin, request=_request())
    assert await _stored(async_db, organization.id) is None

    # A connection that exists is left exactly as it was by a refused edit.
    _stub_discovery(monkeypatch, document=_document())
    await _connection_service(async_db).save_for_user(
        user=admin, request=_request(display_name="Original IdP", client_secret="original-secret-value")
    )
    before = await _stored(async_db, organization.id)
    assert before is not None

    _stub_refused_discovery(monkeypatch)
    with pytest.raises(OIDCDiscoveryError):
        await _connection_service(async_db).save_for_user(
            user=admin, request=_request(display_name="Renamed IdP", client_secret="replacement-secret")
        )

    after = await _stored(async_db, organization.id)
    assert after is not None
    assert after.display_name == "Original IdP"
    assert after.discovery_document == before.discovery_document
    assert after.encrypted_client_secret == before.encrypted_client_secret
    assert decrypt_secret(after.encrypted_client_secret) == "original-secret-value"


async def test_save_over_an_existing_connection_moves_every_submitted_field(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    organization = await create_organization(async_db, slug="acme")
    admin = await _admin(async_db, organization)
    _stub_discovery(monkeypatch)
    first = await _connection_service(async_db).save_for_user(user=admin, request=_request())

    replacement_issuer = "https://idp.acme.example"
    replacement_document = _document(replacement_issuer)
    replacement_discovered_at = datetime.now(UTC) - timedelta(minutes=7)
    _stub_discovery(
        monkeypatch,
        resolved_issuer=replacement_issuer,
        document=replacement_document,
        discovered_at=replacement_discovered_at,
    )

    public = await _connection_service(async_db).save_for_user(
        user=admin,
        request=_request(
            issuer=replacement_issuer,
            client_id="a-replacement-client-id",
            client_secret="replacement-secret-9999",
            display_name="Renamed IdP",
            default_role="viewer",
            enabled=False,
        ),
    )

    assert public.id == first.id
    stored = await _stored(async_db, organization.id)
    assert stored is not None
    assert stored.issuer == replacement_issuer
    assert stored.resolved_issuer == replacement_issuer
    assert stored.client_id == "a-replacement-client-id"
    assert stored.client_secret_last4 == "9999"
    assert stored.display_name == "Renamed IdP"
    assert stored.discovery_document == replacement_document
    assert stored.discovered_at == replacement_discovered_at
    assert stored.default_role == "viewer"
    assert stored.enabled is False
    assert decrypt_secret(stored.encrypted_client_secret) == "replacement-secret-9999"


async def test_an_edit_that_leaves_the_secret_out_keeps_the_stored_one(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    organization = await create_organization(async_db, slug="keeps-the-secret")
    admin = await _admin(async_db, organization)
    _stub_discovery(monkeypatch)
    await _connection_service(async_db).save_for_user(user=admin, request=_request(client_secret="first-secret-value"))

    public = await _connection_service(async_db).save_for_user(
        user=admin, request=_request(client_secret=None, display_name="Renamed IdP")
    )

    assert public.display_name == "Renamed IdP"
    assert public.client_secret_last4 == "alue"
    stored = await _stored(async_db, organization.id)
    assert stored is not None
    assert decrypt_secret(stored.encrypted_client_secret) == "first-secret-value"


async def test_a_first_save_that_leaves_the_secret_out_is_refused(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    organization = await create_organization(async_db, slug="nothing-to-keep")
    admin = await _admin(async_db, organization)
    _stub_discovery(monkeypatch)

    with pytest.raises(OIDCConnectionSecretRequiredError):
        await _connection_service(async_db).save_for_user(user=admin, request=_request(client_secret=None))

    assert await _stored(async_db, organization.id) is None


async def test_a_moved_issuer_cannot_carry_the_stored_secret(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The secret proves this deployment is that client, so it must not follow the issuer somewhere new."""
    organization = await create_organization(async_db, slug="moved-issuer")
    admin = await _admin(async_db, organization)
    _stub_discovery(monkeypatch)
    await _connection_service(async_db).save_for_user(user=admin, request=_request(client_secret="first-secret-value"))

    _stub_discovery(monkeypatch, resolved_issuer="https://elsewhere.example")
    with pytest.raises(OIDCConnectionSecretRequiredError):
        await _connection_service(async_db).save_for_user(
            user=admin, request=_request(issuer="https://elsewhere.example", client_secret=None)
        )

    stored = await _stored(async_db, organization.id)
    assert stored is not None
    assert stored.resolved_issuer == _ISSUER
    assert decrypt_secret(stored.encrypted_client_secret) == "first-secret-value"


async def test_a_changed_client_id_cannot_carry_the_stored_secret(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    organization = await create_organization(async_db, slug="changed-client")
    admin = await _admin(async_db, organization)
    _stub_discovery(monkeypatch)
    await _connection_service(async_db).save_for_user(user=admin, request=_request(client_secret="first-secret-value"))

    with pytest.raises(OIDCConnectionSecretRequiredError):
        await _connection_service(async_db).save_for_user(
            user=admin, request=_request(client_id="another-client-id", client_secret=None)
        )

    stored = await _stored(async_db, organization.id)
    assert stored is not None
    assert stored.client_id == "a-client-id"


async def test_save_is_refused_when_this_deployment_has_no_secret_key(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    organization = await create_organization(async_db, slug="acme")
    admin = await _admin(async_db, organization)
    monkeypatch.delenv("OTARI_SECRET_KEY")
    calls: list[str] = []
    _stub_discovery(monkeypatch, calls=calls)

    with pytest.raises(SecretBoxUnavailableTenancyError):
        await _connection_service(async_db).save_for_user(user=admin, request=_request())

    assert calls == []
    assert await _stored(async_db, organization.id) is None


async def test_a_first_save_that_loses_the_race_takes_over_the_row_that_won(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two first saves both read no connection, so the slower one's insert meets the unique constraint."""
    organization = await create_organization(async_db, slug="acme")
    admin = await _admin(async_db, organization)
    _stub_discovery(monkeypatch)
    original_add = OIDCConnectionRepository.add

    # Wrapping the insert is the one seam between the read that found nothing and the write the
    # unique constraint refuses.
    async def _add_after_a_concurrent_save(repository: OIDCConnectionRepository, **fields: Any) -> OIDCConnection:
        await _insert_in_another_session(async_db, organization.id)
        return await original_add(repository, **fields)

    monkeypatch.setattr(OIDCConnectionRepository, "add", _add_after_a_concurrent_save)

    public = await _connection_service(async_db).save_for_user(
        user=admin, request=_request(display_name="The IdP that asked second")
    )

    assert public.display_name == "The IdP that asked second"
    stored = await _stored(async_db, organization.id)
    assert stored is not None
    assert stored.display_name == "The IdP that asked second"
    assert decrypt_secret(stored.encrypted_client_secret) == "s3cr3t-client-value"


# =============================================================================
# The authority gate
# =============================================================================


_GATED_CALLS: dict[str, Callable[[OIDCConnectionService, User], Awaitable[object]]] = {
    "delete_for_user": lambda service, user: service.delete_for_user(user=user),
    "get_for_user": lambda service, user: service.get_for_user(user=user),
    "save_for_user": lambda service, user: service.save_for_user(user=user, request=_request()),
    "test_for_user": lambda service, user: service.test_for_user(user=user),
}


@pytest.mark.parametrize("role", ["member", "viewer"])
@pytest.mark.parametrize("method", sorted(_GATED_CALLS))
async def test_a_member_who_cannot_manage_the_organization_is_refused(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch, role: str, method: str
) -> None:
    """The organization's connection is neither served, re-discovered nor removed for such a caller."""
    organization = await create_organization(async_db, slug="acme")
    admin = await _admin(async_db, organization)
    _stub_discovery(monkeypatch)
    await _connection_service(async_db).save_for_user(user=admin, request=_request())
    caller = await create_member(async_db, organization, role=role, full_name="Mo Member")
    calls: list[str] = []
    _stub_discovery(monkeypatch, calls=calls)

    with pytest.raises(NotAuthorizedError):
        await _GATED_CALLS[method](_connection_service(async_db), caller)

    assert calls == []
    stored = await _stored(async_db, organization.id)
    assert stored is not None
    assert stored.display_name == "Example IdP"


# =============================================================================
# Reading
# =============================================================================


async def test_get_for_user_returns_none_when_no_connection_exists(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    admin = await _admin(async_db, organization)

    assert await _connection_service(async_db).get_for_user(user=admin) is None


async def test_get_for_user_lists_every_domain_with_its_proof_expiry_and_enabled_flag(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins the domains shape: every domain, proof_expires_at null only when unproven, plus enabled."""
    organization = await create_organization(async_db, slug="acme")
    admin = await _admin(async_db, organization)
    _stub_discovery(monkeypatch)
    await _connection_service(async_db).save_for_user(user=admin, request=_request())

    domains = OrganizationDomainRepository(async_db)
    verified_at = datetime.now(UTC) - timedelta(days=1)
    verified_and_enabled = await domains.create_domain(
        organization_id=organization.id,
        domain="acme.example",
        default_role="member",
        enabled=True,
        verification_token="tok-verified-enabled",
    )
    verified_and_enabled.verified_at = verified_at
    async_db.add(verified_and_enabled)
    verified_but_disabled = await domains.create_domain(
        organization_id=organization.id,
        domain="disabled.example",
        default_role="member",
        enabled=False,
        verification_token="tok-verified-disabled",
    )
    verified_but_disabled.verified_at = verified_at
    async_db.add(verified_but_disabled)
    await domains.create_domain(
        organization_id=organization.id,
        domain="unproven.example",
        default_role="member",
        enabled=True,
        verification_token="tok-unproven",
    )
    await async_db.flush()

    public = await _connection_service(async_db).get_for_user(user=admin)

    assert public is not None
    by_domain = {entry.domain: entry for entry in public.domains}
    assert set(by_domain) == {"acme.example", "disabled.example", "unproven.example"}
    assert by_domain["acme.example"].proof_expires_at == verified_at + DOMAIN_PROOF_TTL
    assert by_domain["acme.example"].enabled is True
    assert by_domain["disabled.example"].proof_expires_at == verified_at + DOMAIN_PROOF_TTL
    assert by_domain["disabled.example"].enabled is False
    assert by_domain["unproven.example"].proof_expires_at is None
    assert by_domain["unproven.example"].enabled is True


# =============================================================================
# Testing an existing connection
# =============================================================================


async def test_test_for_user_rediscovers_and_stores(async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    organization = await create_organization(async_db, slug="acme")
    admin = await _admin(async_db, organization)
    _stub_discovery(monkeypatch, document=_document())
    await _connection_service(async_db).save_for_user(user=admin, request=_request())

    refreshed_document = _document()
    refreshed_document["token_url"] = f"{_ISSUER}/token-v2"
    _stub_discovery(monkeypatch, document=refreshed_document)

    public = await _connection_service(async_db).test_for_user(user=admin)

    assert public.resolved_issuer == _ISSUER
    stored = await _stored(async_db, organization.id)
    assert stored is not None
    assert stored.discovery_document == refreshed_document


async def test_test_for_user_refuses_an_organization_with_no_connection(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    admin = await _admin(async_db, organization)

    with pytest.raises(OIDCConnectionNotFoundError):
        await _connection_service(async_db).test_for_user(user=admin)


async def test_test_for_user_refuses_a_connection_deleted_while_discovery_is_out(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    organization = await create_organization(async_db, slug="acme")
    admin = await _admin(async_db, organization)
    _stub_discovery(monkeypatch)
    await _connection_service(async_db).save_for_user(user=admin, request=_request())

    async def _discover_after_delete(issuer: str, *, allow_private_hosts: bool) -> DiscoveredProvider:
        # Stand in for another admin deleting the connection while this call is out.
        await _delete_in_another_session(async_db, admin)
        return DiscoveredProvider(resolved_issuer=_ISSUER, document=_document(), discovered_at=datetime.now(UTC))

    monkeypatch.setattr(f"{_SERVICE_MODULE}.discover_connection_metadata", _discover_after_delete)

    with pytest.raises(OIDCConnectionNotFoundError):
        await _connection_service(async_db).test_for_user(user=admin)


# =============================================================================
# Deleting
# =============================================================================


async def test_delete_for_user_refuses_an_organization_with_no_connection(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    admin = await _admin(async_db, organization)

    with pytest.raises(OIDCConnectionNotFoundError):
        await _connection_service(async_db).delete_for_user(user=admin)


async def test_delete_for_user_removes_the_connection(async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    organization = await create_organization(async_db, slug="acme")
    admin = await _admin(async_db, organization)
    _stub_discovery(monkeypatch)
    await _connection_service(async_db).save_for_user(user=admin, request=_request())

    await _connection_service(async_db).delete_for_user(user=admin)

    assert await _stored(async_db, organization.id) is None


# =============================================================================
# The sign-in refresh
# =============================================================================


async def test_metadata_for_sign_in_keeps_the_stored_document_when_a_refresh_fails(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    organization = await create_organization(async_db, slug="acme")
    admin = await _admin(async_db, organization)
    stored_document = _document()
    _stub_discovery(monkeypatch, document=stored_document, discovered_at=datetime.now(UTC) - timedelta(hours=13))
    await _connection_service(async_db).save_for_user(user=admin, request=_request())
    connection = await _stored(async_db, organization.id)
    assert connection is not None

    calls: list[str] = []
    _stub_refused_discovery(monkeypatch, calls=calls)

    result = await _connection_service(async_db).metadata_for_sign_in(connection)

    assert calls == [connection.issuer]
    assert result.document == stored_document
    assert result.resolved_issuer == connection.resolved_issuer

    unchanged = await _stored(async_db, organization.id)
    assert unchanged is not None
    assert unchanged.discovery_document == stored_document


async def test_metadata_for_sign_in_refreshes_past_the_ttl(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    organization = await create_organization(async_db, slug="acme")
    admin = await _admin(async_db, organization)
    stale_document = _document()
    _stub_discovery(monkeypatch, document=stale_document, discovered_at=datetime.now(UTC) - timedelta(hours=13))
    await _connection_service(async_db).save_for_user(user=admin, request=_request())
    connection = await _stored(async_db, organization.id)
    assert connection is not None

    fresh_document = _document()
    fresh_document["token_url"] = f"{_ISSUER}/token-refreshed"
    fresh_resolved_issuer = f"{_ISSUER}/"
    calls: list[str] = []
    _stub_discovery(monkeypatch, resolved_issuer=fresh_resolved_issuer, document=fresh_document, calls=calls)

    result = await _connection_service(async_db).metadata_for_sign_in(connection)

    assert calls == [connection.issuer]
    assert result.document == fresh_document
    assert result.resolved_issuer == fresh_resolved_issuer

    refreshed = await _stored(async_db, organization.id)
    assert refreshed is not None
    assert refreshed.discovery_document == fresh_document
    assert refreshed.resolved_issuer == fresh_resolved_issuer


async def test_metadata_for_sign_in_does_not_refresh_within_the_ttl(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    organization = await create_organization(async_db, slug="acme")
    admin = await _admin(async_db, organization)
    stored_document = _document()
    _stub_discovery(monkeypatch, document=stored_document, discovered_at=datetime.now(UTC) - timedelta(hours=1))
    await _connection_service(async_db).save_for_user(user=admin, request=_request())
    connection = await _stored(async_db, organization.id)
    assert connection is not None

    calls: list[str] = []
    _stub_discovery(monkeypatch, document=_document(), calls=calls)

    result = await _connection_service(async_db).metadata_for_sign_in(connection)

    assert calls == []
    assert result.document == stored_document


async def test_metadata_for_sign_in_releases_the_session_before_the_outbound_call(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    organization = await create_organization(async_db, slug="acme")
    admin = await _admin(async_db, organization)
    _stub_discovery(monkeypatch, discovered_at=datetime.now(UTC) - timedelta(hours=13))
    await _connection_service(async_db).save_for_user(user=admin, request=_request())
    connection = await _stored(async_db, organization.id)
    assert connection is not None

    order: list[str] = []

    async def _recording_release(session: AsyncSession) -> bool:
        order.append("release")
        return await release_session(session)

    async def _recording_discover(issuer: str, *, allow_private_hosts: bool) -> DiscoveredProvider:
        order.append("discover")
        return DiscoveredProvider(resolved_issuer=_ISSUER, document=_document(), discovered_at=datetime.now(UTC))

    monkeypatch.setattr("gateway.core.unit_of_work.release_session", _recording_release)
    monkeypatch.setattr(f"{_SERVICE_MODULE}.discover_connection_metadata", _recording_discover)

    await _connection_service(async_db).metadata_for_sign_in(connection)

    assert order == ["release", "discover"]


async def test_metadata_for_sign_in_tolerates_the_connection_being_deleted_while_discovery_is_out(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    organization = await create_organization(async_db, slug="acme")
    admin = await _admin(async_db, organization)
    _stub_discovery(monkeypatch, discovered_at=datetime.now(UTC) - timedelta(hours=13))
    await _connection_service(async_db).save_for_user(user=admin, request=_request())
    connection = await _stored(async_db, organization.id)
    assert connection is not None

    async def _discover_after_delete(issuer: str, *, allow_private_hosts: bool) -> DiscoveredProvider:
        # Stand in for the admin deleting the connection while this discovery call is out.
        await _delete_in_another_session(async_db, admin)
        return DiscoveredProvider(resolved_issuer=_ISSUER, document=_document(), discovered_at=datetime.now(UTC))

    monkeypatch.setattr(f"{_SERVICE_MODULE}.discover_connection_metadata", _discover_after_delete)

    result = await _connection_service(async_db).metadata_for_sign_in(connection)

    assert result.resolved_issuer == _ISSUER
    assert await _stored(async_db, organization.id) is None
