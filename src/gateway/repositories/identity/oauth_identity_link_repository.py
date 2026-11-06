import uuid
from typing import Never

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.identity_exceptions import OAuthIdentityLinkConflictError
from gateway.models.identity import OAuthIdentityLink, ProviderIdentity
from gateway.repositories.base_repository import BaseRepository


class OAuthIdentityLinkRepository(BaseRepository[OAuthIdentityLink, Never, Never]):
    """Query provider identity links in the open block of a Unit of Work."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, OAuthIdentityLink)

    async def get_by_identity(self, *, provider: str, subject: str) -> OAuthIdentityLink | None:
        """Return the link for one provider identity, or None."""
        return await self.get((provider, subject))

    async def get_for_user(self, *, provider: str, user_id: uuid.UUID) -> OAuthIdentityLink | None:
        """Return an account's link at one provider, or None."""
        result = await self.db.execute(
            select(OAuthIdentityLink).where(
                OAuthIdentityLink.user_id == user_id, OAuthIdentityLink.provider == provider
            )
        )
        return result.scalar_one_or_none()

    async def add(self, external: ProviderIdentity, *, user_id: uuid.UUID) -> OAuthIdentityLink:
        """Stage a link from a provider identity to an account.

        Raises:
            OAuthIdentityLinkConflictError: If the identity, or the account at that provider, is already linked.

        """
        link = OAuthIdentityLink(
            provider=external.provider,
            subject=external.subject,
            user_id=user_id,
            email_at_link=external.email,
            email_assertion_at_link=external.email_verified,
        )
        self.db.add(link)
        try:
            await self.db.flush()
        except IntegrityError as error:
            raise OAuthIdentityLinkConflictError(external.provider) from error
        return link
