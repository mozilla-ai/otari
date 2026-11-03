"""An organization's OIDC connection, read and written in a Unit of Work."""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.identity import OIDCConnection
from gateway.repositories.identity import OIDCConnectionRepository

from .tenancy_helpers import create_organization

pytestmark = pytest.mark.asyncio

_ISSUER = "https://login.example.com"


async def _add_connection(
    connections: OIDCConnectionRepository,
    organization_id: uuid.UUID,
    *,
    issuer: str = _ISSUER,
    resolved_issuer: str = _ISSUER,
    enabled: bool = True,
) -> OIDCConnection:
    return await connections.add(
        organization_id=organization_id,
        issuer=issuer,
        resolved_issuer=resolved_issuer,
        client_id="a-client-id",
        encrypted_client_secret="encrypted-secret",
        client_secret_last4="cret",
        display_name="Example IdP",
        discovery_document={"issuer": issuer},
        discovered_at=datetime.now(UTC),
        enabled=enabled,
    )


async def test_one_organization_holds_at_most_one_connection(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    uow = UnitOfWork(async_db)
    connections = OIDCConnectionRepository(uow)
    async with uow:
        await _add_connection(connections, organization.id)

    with pytest.raises(IntegrityError):
        async with uow:
            await _add_connection(connections, organization.id, issuer="https://other.example.com")


async def test_two_organizations_may_share_one_issuer(async_db: AsyncSession) -> None:
    first = await create_organization(async_db, slug="acme")
    second = await create_organization(async_db, slug="globex")
    uow = UnitOfWork(async_db)
    connections = OIDCConnectionRepository(uow)
    async with uow:
        await _add_connection(connections, first.id)
        await _add_connection(connections, second.id)

    async with uow:
        shared = await connections.list_enabled_by_resolved_issuer(_ISSUER)

    assert {connection.organization_id for connection in shared} == {first.id, second.id}


async def test_a_disabled_connection_is_not_listed(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    uow = UnitOfWork(async_db)
    connections = OIDCConnectionRepository(uow)
    async with uow:
        await _add_connection(connections, organization.id, enabled=False)

    async with uow:
        listed = await connections.list_enabled_by_resolved_issuer(_ISSUER)

    assert listed == []


async def test_a_connection_is_found_by_organization_and_by_id(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    uow = UnitOfWork(async_db)
    connections = OIDCConnectionRepository(uow)
    async with uow:
        added = await _add_connection(connections, organization.id)

    async with uow:
        by_organization = await connections.get_for_organization(organization.id)
        by_id = await connections.get_by_id(added.id)

    assert by_organization is not None
    assert by_organization.id == added.id
    assert by_id is not None
    assert by_id.id == added.id


async def test_stage_and_delete_write_through_the_unit_of_work(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    uow = UnitOfWork(async_db)
    connections = OIDCConnectionRepository(uow)
    async with uow:
        added = await _add_connection(connections, organization.id)

    async with uow:
        added.display_name = "Renamed IdP"
        connections.stage(added)

    async with uow:
        renamed = await connections.get_by_id(added.id)
        assert renamed is not None
        assert renamed.display_name == "Renamed IdP"
        await connections.delete(renamed)

    async with uow:
        assert await connections.get_by_id(added.id) is None
