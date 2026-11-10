"""Data access for the authorizations in flight, one row per pending sign-in."""

from dataclasses import dataclass
from datetime import datetime
from typing import Never

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tenancy import OAuthPendingState
from gateway.repositories.base_repository import BaseRepository


@dataclass(frozen=True)
class ClaimedAuthorization:
    """One pending authorization, read back by the claim that removed it."""

    provider: str
    code_verifier: str | None
    nonce: str | None
    openid_requested: bool
    redirect_uri: str
    created_at: datetime
    expires_at: datetime


class OAuthPendingStateRepository(BaseRepository[OAuthPendingState, Never, Never]):
    """Query and stage pending authorizations. Writes flush and never commit."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, OAuthPendingState)

    async def get_provider(self, state_hash: str) -> str | None:
        """The provider this pending authorization names, leaving it pending."""
        result = await self.db.execute(
            select(col(OAuthPendingState.provider)).where(col(OAuthPendingState.state_hash) == state_hash)
        )
        return result.scalar_one_or_none()

    async def stage(
        self,
        *,
        state_hash: str,
        provider: str,
        code_verifier: str | None,
        nonce: str | None,
        openid_requested: bool,
        redirect_uri: str,
        expires_at: datetime,
    ) -> None:
        """Stage one pending authorization."""
        self.db.add(
            OAuthPendingState(
                state_hash=state_hash,
                provider=provider,
                code_verifier=code_verifier,
                nonce=nonce,
                openid_requested=openid_requested,
                redirect_uri=redirect_uri,
                expires_at=expires_at,
            )
        )
        await self.db.flush()

    async def delete_expired(self, now: datetime) -> None:
        """Stage the removal of every authorization that is no longer in flight."""
        await self.db.execute(delete(OAuthPendingState).where(col(OAuthPendingState.expires_at) <= now))

    async def claim(self, state_hash: str) -> ClaimedAuthorization | None:
        """Remove one pending authorization and return what it held, or None.

        One conditional DELETE rather than a read and then a delete, because single use is the
        property the row carries: two callbacks replaying one state would both find it, while
        deleting by primary key and reading what came back admits exactly one.
        """
        row = (
            await self.db.execute(
                delete(OAuthPendingState)
                .where(col(OAuthPendingState.state_hash) == state_hash)
                .returning(
                    col(OAuthPendingState.provider),
                    col(OAuthPendingState.code_verifier),
                    col(OAuthPendingState.nonce),
                    col(OAuthPendingState.openid_requested),
                    col(OAuthPendingState.redirect_uri),
                    col(OAuthPendingState.created_at),
                    col(OAuthPendingState.expires_at),
                )
            )
        ).first()
        if row is None:
            return None
        provider, code_verifier, nonce, openid_requested, redirect_uri, created_at, expires_at = row
        return ClaimedAuthorization(
            provider=provider,
            code_verifier=code_verifier,
            nonce=nonce,
            openid_requested=openid_requested,
            redirect_uri=redirect_uri,
            created_at=created_at,
            expires_at=expires_at,
        )
