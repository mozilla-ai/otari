from collections.abc import Sequence
from typing import Never

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.api_keys import APIKeyFederationRule
from gateway.repositories.base_repository import BaseRepository


class FederationRuleRepository(BaseRepository[APIKeyFederationRule, Never, Never]):
    """Query the federation rules of API keys.

    Built on a Unit of Work for the management routes, and on the request's
    session for authentication, which only reads.
    """

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, APIKeyFederationRule)

    async def list_for_key(self, api_key_id: str) -> list[APIKeyFederationRule]:
        """Return a key's rules, oldest first."""
        result = await self.db.execute(
            select(APIKeyFederationRule)
            .where(APIKeyFederationRule.api_key_id == api_key_id)
            .order_by(APIKeyFederationRule.created_at, APIKeyFederationRule.id)
        )
        return list(result.scalars().all())

    async def count_for_key(self, api_key_id: str) -> int:
        """Return how many rules a key holds."""
        result = await self.db.execute(
            select(func.count()).select_from(APIKeyFederationRule).where(APIKeyFederationRule.api_key_id == api_key_id)
        )
        return int(result.scalar_one())

    async def get_for_key(self, api_key_id: str, rule_id: str) -> APIKeyFederationRule | None:
        """Return one of a key's rules, or None when the key holds no rule with that id."""
        result = await self.db.execute(
            select(APIKeyFederationRule).where(
                APIKeyFederationRule.id == rule_id, APIKeyFederationRule.api_key_id == api_key_id
            )
        )
        return result.scalar_one_or_none()

    async def find_identical(
        self, *, issuer: str, audience: str, subject: str | None, email: str | None
    ) -> APIKeyFederationRule | None:
        """Return a rule on any key that names exactly this identity, or None."""
        statement = select(APIKeyFederationRule).where(
            APIKeyFederationRule.issuer == issuer,
            APIKeyFederationRule.audience == audience,
            APIKeyFederationRule.subject.is_(None) if subject is None else APIKeyFederationRule.subject == subject,
            APIKeyFederationRule.email.is_(None) if email is None else APIKeyFederationRule.email == email,
        )
        result = await self.db.execute(statement.limit(1))
        return result.scalar_one_or_none()

    async def add(
        self, *, rule_id: str, api_key_id: str, issuer: str, audience: str, subject: str | None, email: str | None
    ) -> APIKeyFederationRule:
        """Stage a rule, flushed so its generated columns can be read back."""
        row = APIKeyFederationRule(
            id=rule_id, api_key_id=api_key_id, issuer=issuer, audience=audience, subject=subject, email=email
        )
        self.db.add(row)
        await self.db.flush()
        await self.db.refresh(row)
        return row

    async def candidates(self, issuer: str, audiences: Sequence[str]) -> list[APIKeyFederationRule]:
        """Return every rule for this issuer whose audience is one of ``audiences``."""
        if not audiences:
            return []
        result = await self.db.execute(
            select(APIKeyFederationRule).where(
                APIKeyFederationRule.issuer == issuer, APIKeyFederationRule.audience.in_(list(audiences))
            )
        )
        return list(result.scalars().all())
