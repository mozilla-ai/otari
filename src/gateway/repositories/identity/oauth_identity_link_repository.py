import uuid
from typing import Never

from sqlalchemy import select

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.identity import OAuthIdentityLink
from gateway.repositories.base_repository import BaseRepository


class OAuthIdentityLinkRepository(BaseRepository[OAuthIdentityLink, Never, Never]):
    """Query provider identity links in the open block of a Unit of Work."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, OAuthIdentityLink)

    async def get_by_identity(self, provider: str, subject: str) -> OAuthIdentityLink | None:
        """Return the link for one provider identity, or None."""
        return await self.get((provider, subject))

    async def get_for_user(self, user_id: uuid.UUID, provider: str) -> OAuthIdentityLink | None:
        """Return an account's link at one provider, or None."""
        result = await self.db.execute(
            select(OAuthIdentityLink).where(
                OAuthIdentityLink.user_id == user_id, OAuthIdentityLink.provider == provider
            )
        )
        return result.scalar_one_or_none()

    async def add(
        self, *, provider: str, subject: str, user_id: uuid.UUID, email: str | None, email_verified: bool
    ) -> OAuthIdentityLink:
        """Stage a link from a provider identity to an account."""
        link = OAuthIdentityLink(
            provider=provider,
            subject=subject,
            user_id=user_id,
            email_at_link=email,
            email_verified_at_link=email_verified,
        )
        self.db.add(link)
        await self.db.flush()
        return link
