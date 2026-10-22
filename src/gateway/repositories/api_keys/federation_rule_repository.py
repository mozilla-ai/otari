from collections.abc import Sequence
from typing import Never

from sqlalchemy import and_, false, func, or_, select, true
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

    async def find_conflicting(
        self,
        *,
        api_key_id: str,
        issuer: str,
        audience: str,
        subject: str | None,
        email: str | None,
        member_managed: bool = False,
    ) -> APIKeyFederationRule | None:
        """Return a rule that names this identity, or None.

        For an operator's rule, that is an operator's rule on any key with the same issuer, audience, subject and
        email, or one on another key with the same issuer and audience that shares a subject or an email with this
        one and contradicts it in neither. The search spans every organization, which suits rules only the
        deployment operator manages. Members' rules take no part in it: they never match a bare identity token, so
        they cannot make one ambiguous, and leaving them out keeps a member from blocking the operator's rule.

        For a member's rule, it is only the same rule on the same key. A member's rule matches only in the exchange
        that names it, so another key's rule for the same identity takes nothing from it, and searching further
        would make a 409 reveal another tenant's identity.
        """
        rule = APIKeyFederationRule
        same_subject = rule.subject.is_(None) if subject is None else rule.subject == subject
        same_email = rule.email.is_(None) if email is None else rule.email == email
        same_identity = and_(rule.issuer == issuer, rule.audience == audience, same_subject, same_email)
        if member_managed:
            statement = select(rule).where(rule.api_key_id == api_key_id, rule.member_managed.is_(True), same_identity)
        else:
            compatible_subject = true() if subject is None else or_(rule.subject.is_(None), rule.subject == subject)
            compatible_email = true() if email is None else or_(rule.email.is_(None), rule.email == email)
            shared = or_(
                false() if subject is None else rule.subject == subject,
                false() if email is None else rule.email == email,
            )
            statement = select(rule).where(
                rule.member_managed.is_(False),
                or_(
                    same_identity,
                    and_(
                        rule.issuer == issuer,
                        rule.audience == audience,
                        rule.api_key_id != api_key_id,
                        compatible_subject,
                        compatible_email,
                        shared,
                    ),
                ),
            )
        result = await self.db.execute(statement.limit(1))
        return result.scalar_one_or_none()

    async def add(
        self,
        *,
        rule_id: str,
        api_key_id: str,
        issuer: str,
        audience: str,
        subject: str | None,
        email: str | None,
        member_managed: bool = False,
    ) -> APIKeyFederationRule:
        """Stage a rule, flushed so its generated columns can be read back."""
        row = APIKeyFederationRule(
            id=rule_id,
            api_key_id=api_key_id,
            issuer=issuer,
            audience=audience,
            subject=subject,
            email=email,
            member_managed=member_managed,
        )
        self.db.add(row)
        await self.db.flush()
        await self.db.refresh(row)
        return row

    async def candidates(self, issuer: str, audiences: Sequence[str]) -> list[APIKeyFederationRule]:
        """Return every operator's rule for this issuer whose audience is one of ``audiences``.

        What a bare identity token is matched against, across every key. A member's rule is left out: it matches only
        in the token exchange that names it, so it can never claim a token another tenant's workload presents.
        """
        if not audiences:
            return []
        result = await self.db.execute(
            select(APIKeyFederationRule).where(
                APIKeyFederationRule.issuer == issuer,
                APIKeyFederationRule.audience.in_(list(audiences)),
                APIKeyFederationRule.member_managed.is_(False),
            )
        )
        return list(result.scalars().all())
