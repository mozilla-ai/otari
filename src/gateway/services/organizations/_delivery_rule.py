"""The rule for delivering an invitation email to an address."""

import uuid
from datetime import datetime

from gateway.core.addresses import normalized_address
from gateway.core.email_domains import email_domain
from gateway.services.organizations._blocked_addresses import BlockedAddresses
from gateway.services.organizations._claimed_domains import ClaimedDomains


class DeliveryRule:
    """Answers whether Otari delivers an invitation email to an address."""

    def __init__(self, domains: ClaimedDomains, blocked: BlockedAddresses) -> None:
        self._domains = domains
        self._blocked = blocked

    async def allows(self, email: str, organization_id: uuid.UUID, *, now: datetime) -> bool:
        """Return whether an invitation from ``organization_id`` is delivered to ``email`` at ``now``.

        The answer is False when another organization holds a proof of the address's domain that has not expired.
        It is False when the deployment's email blocklist matches the address.
        It is False when ``email`` is not an email address.

        NOTE: Callers must call this inside an open Unit of Work block.
        """
        address = normalized_address(email)
        if address is None:
            return False
        domain = email_domain(address)
        if domain is None or await self._blocked.blocked_among([address]):
            return False
        proof = await self._domains.get_verified_by_domain(domain)
        return proof is None or proof.organization_id == organization_id or proof.proof_expired(now=now)
