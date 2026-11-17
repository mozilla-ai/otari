"""Which email domains an organization has claimed."""

import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from gateway.models.tenancy import DOMAIN_PROOF_TTL, OrganizationDomain
from gateway.repositories.tenancy import OrganizationDomainRepository


@dataclass(frozen=True)
class ClaimedDomain:
    """One organization's claim on an email domain, as another domain reads it.

    ``verified_at`` is when the DNS proof was last seen, and None where it never was. Whether that
    proof is still young enough to admit anyone is the reader's rule, not this claim's.
    """

    domain: str
    verified_at: datetime | None
    enabled: bool


class ClaimedDomains(Protocol):
    """Answers which email domains an organization has claimed.

    The organizations domain implements it, so a reader needs neither the rows nor the repository
    that holds them. NOTE: an implementation reads through the request's Unit of Work, so call it
    before staging any write, or inside a block already open on that Unit of Work.
    """

    async def for_organization(self, organization_id: uuid.UUID) -> list[ClaimedDomain]:
        """Return the organization's claims, in the order the claim surface lists them."""
        ...

    async def get_verified_by_domain(self, domain: str) -> OrganizationDomain | None:
        """Return the one proven claim on ``domain``, or None where nobody holds one."""
        ...


class OrganizationClaimedDomains:
    """Answers the question above from this organization's own rows."""

    def __init__(self, domains: OrganizationDomainRepository) -> None:
        self._domains = domains

    async def for_organization(self, organization_id: uuid.UUID) -> list[ClaimedDomain]:
        """Return the organization's claims, oldest first, as the claim surface lists them."""
        claimed = await self._domains.list_by_organization(organization_id)
        return [ClaimedDomain(domain=row.domain, verified_at=row.verified_at, enabled=row.enabled) for row in claimed]

    async def get_verified_by_domain(self, domain: str) -> OrganizationDomain | None:
        """Return the one proven claim on ``domain``, or None where nobody holds one."""
        return await self._domains.get_verified_by_domain(domain)


def proven_domains(claims: Iterable[ClaimedDomain], *, now: datetime) -> tuple[str, ...]:
    """Return the domain of each claim in ``claims`` whose proof is still within ``DOMAIN_PROOF_TTL`` of ``now``.

    A claim that has never been verified is left out.
    A disabled claim is returned the same as an enabled one.
    The claims keep the order they were given in.
    """
    return tuple(
        claim.domain
        for claim in claims
        if claim.verified_at is not None and now - claim.verified_at <= DOMAIN_PROOF_TTL
    )
