"""Data access for an organization's OIDC connection."""

import uuid
from datetime import datetime
from typing import Any, Never

from sqlalchemy import select
from sqlmodel import col

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.identity import OIDCConnection
from gateway.repositories.base_repository import BaseRepository


class OIDCConnectionRepository(BaseRepository[OIDCConnection, Never, Never]):
    """Query and stage an organization's OIDC connection in the open block of a Unit of Work."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, OIDCConnection)

    async def get_for_organization(self, organization_id: uuid.UUID) -> OIDCConnection | None:
        """Return the organization's connection, or None."""
        result = await self.db.execute(
            select(OIDCConnection).where(col(OIDCConnection.organization_id) == organization_id)
        )
        return result.scalar_one_or_none()

    async def get_by_id(self, connection_id: uuid.UUID) -> OIDCConnection | None:
        """Return the connection by its ID, or None."""
        return await self.get(connection_id)

    async def list_enabled_by_resolved_issuer(self, issuer: str) -> list[OIDCConnection]:
        """Return every enabled connection whose discovery resolved to this issuer.

        NOTE: More than one connection, from different organizations, can be returned.
        """
        result = await self.db.execute(
            select(OIDCConnection).where(
                col(OIDCConnection.resolved_issuer) == issuer, col(OIDCConnection.enabled).is_(True)
            )
        )
        return list(result.scalars().all())

    async def add(
        self,
        *,
        organization_id: uuid.UUID,
        issuer: str,
        resolved_issuer: str,
        client_id: str,
        encrypted_client_secret: str,
        client_secret_last4: str,
        display_name: str,
        discovery_document: dict[str, Any],
        discovered_at: datetime,
        default_role: str = "member",
        enabled: bool = True,
    ) -> OIDCConnection:
        """Stage a new connection for an organization.

        Raises:
            IntegrityError: The organization already holds a connection.

        """
        connection = OIDCConnection(
            organization_id=organization_id,
            issuer=issuer,
            resolved_issuer=resolved_issuer,
            client_id=client_id,
            encrypted_client_secret=encrypted_client_secret,
            client_secret_last4=client_secret_last4,
            display_name=display_name,
            discovery_document=discovery_document,
            discovered_at=discovered_at,
            default_role=default_role,
            enabled=enabled,
        )
        self.db.add(connection)
        await self.db.flush()
        return connection

    def stage(self, connection: OIDCConnection) -> None:
        """Stage the connection's changes for the caller's transaction to write."""
        self.db.add(connection)


__all__ = ["OIDCConnectionRepository"]
