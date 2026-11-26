"""What the invitation flow's builders wire, and that each one is reachable at all.

A builder is the one place a service's collaborators are named, so a missing or
crossed wire shows up only where the service is used. These assemble each one
against a real Unit of Work and read back what it was given.
"""

from datetime import timedelta
from typing import Any, get_args, get_type_hints

import pytest
import pytest_asyncio
from fastapi import params
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the metadata)
from gateway.api.deps.invitations import (
    get_invitation_acceptance,
    get_invitation_link,
    get_invitations,
)
from gateway.api.deps.organizations import get_membership_organization_service
from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.services.organizations import InvitationAcceptance, InvitationLink, Invitations

pytestmark = pytest.mark.asyncio


def _config() -> GatewayConfig:
    return GatewayConfig(
        invitation_expiry_hours=48,
        invitation_per_address_per_day=2,
        invitation_per_day=7,
        invitation_resend_limit=3,
        invitation_resend_wait_minutes=11,
        mail_from_email="otari@example.com",
        mail_transport="console",
        public_base_url="https://otari.example.com",
    )


class _Container:
    """Answers the one port the invitation service resolves."""

    def __init__(self, store: object) -> None:
        self._store = store

    def resolve(self, port: type, _session: object) -> object:
        return self._store


@pytest_asyncio.fixture
async def uow() -> Any:
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield UnitOfWork(session)
    await engine.dispose()


async def test_the_link_reader_is_assembled(uow: UnitOfWork) -> None:
    """Every collaborator it needs is reachable from a Unit of Work and a config alone."""
    link = get_invitation_link(uow, _config())

    assert isinstance(link, InvitationLink)


async def test_the_invitation_service_carries_the_configured_limits(uow: UnitOfWork) -> None:
    """The limits reach the service through the builder, not only the policy function."""
    store = object()
    service = get_invitations(uow, _config(), _Container(store))  # type: ignore[arg-type]

    assert isinstance(service, Invitations)
    assert service._policy.expires_after == timedelta(hours=48)
    assert service._policy.per_address_per_day == 2
    assert service._policy.per_day == 7
    assert service._policy.resend_limit == 3
    assert service._policy.resend_wait == timedelta(minutes=11)


async def test_acceptance_is_given_the_admission_and_the_connections_it_is_handed(uow: UnitOfWork) -> None:
    """Both reach the service rather than one being dropped or built twice."""
    link = get_invitation_link(uow, _config())
    admission, connections = object(), object()

    acceptance = get_invitation_acceptance(uow, link, admission, connections)  # type: ignore[arg-type]

    assert isinstance(acceptance, InvitationAcceptance)
    assert acceptance._admission is admission
    assert acceptance._sso is connections
    assert acceptance._link is link


async def test_acceptance_asks_for_the_organization_service_that_can_write() -> None:
    """Admitting writes a membership, and the read-only pairing refuses at the first write.

    Both pairings are the same class, so each satisfies the protocol and nothing
    but this says which one the route is handed.
    """
    annotation = get_type_hints(get_invitation_acceptance, include_extras=True)["organizations"]
    depends = next(item for item in get_args(annotation)[1:] if isinstance(item, params.Depends))

    assert depends.dependency is get_membership_organization_service
