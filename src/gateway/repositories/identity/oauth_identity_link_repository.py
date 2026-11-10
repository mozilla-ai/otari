import uuid
from collections.abc import Collection
from typing import Never

from sqlalchemy import func, or_, select
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

    async def get_other_providers_for_user(self, *, provider: str, user_id: uuid.UUID) -> set[str]:
        """Every provider this identity holds a sign-in link at, other than ``provider``.

        The providers rather than whether any exists, because whether one is a way in depends on
        which it is, and that is the caller's rule to apply.
        """
        result = await self.db.execute(
            select(OAuthIdentityLink.provider).where(
                OAuthIdentityLink.user_id == user_id, OAuthIdentityLink.provider != provider
            )
        )
        return set(result.scalars().all())

    async def get_for_user(self, *, provider: str, user_id: uuid.UUID) -> OAuthIdentityLink | None:
        """Return an account's link at one provider, or None."""
        result = await self.db.execute(
            select(OAuthIdentityLink).where(
                OAuthIdentityLink.user_id == user_id, OAuthIdentityLink.provider == provider
            )
        )
        return result.scalar_one_or_none()

    async def linked_user_ids(self, *, provider: str, user_ids: Collection[uuid.UUID]) -> set[uuid.UUID]:
        """Return which of ``user_ids`` already hold a link at ``provider``."""
        if not user_ids:
            return set()
        result = await self.db.execute(
            select(OAuthIdentityLink.user_id).where(
                OAuthIdentityLink.user_id.in_(user_ids), OAuthIdentityLink.provider == provider
            )
        )
        return set(result.scalars().all())

    async def list_with_email_at(
        self, domains: Collection[str], *, include_subdomains: bool
    ) -> list[tuple[uuid.UUID, str]]:
        """Return ``(user_id, email)`` for every link whose asserted address is at one of ``domains``, or under
        one where ``include_subdomains``.

        Matched case-insensitively, as the identity table's own lookup is. ``domains`` are plain host names, so
        they hold no ``LIKE`` wildcard to escape.
        """
        lowered = func.lower(OAuthIdentityLink.email_at_link)
        conditions = [lowered.like(f"%@{domain.lower()}") for domain in domains]
        if include_subdomains:
            conditions.extend(lowered.like(f"%@%.{domain.lower()}") for domain in domains)
        if not conditions:
            return []
        result = await self.db.execute(
            select(OAuthIdentityLink.user_id, OAuthIdentityLink.email_at_link).where(or_(*conditions))
        )
        return [(user_id, email) for user_id, email in result.tuples().all() if email is not None]

    def record_asserted_address(self, link: OAuthIdentityLink, external: ProviderIdentity) -> None:
        """Stage what ``external``'s provider asserts about its address on this sign-in."""
        link.email_at_link = external.email
        link.email_assertion_at_link = external.email_verified
        self.db.add(link)

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
