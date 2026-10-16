"""How much invitation mail an organization may send, and an address may receive."""

import hashlib
import uuid

from gateway.core.settings.organizations import INVITATION_MAIL_WINDOW_HOURS
from gateway.ports.rate_limit_store_port import RateLimitStorePort

_WINDOW_SEC = INVITATION_MAIL_WINDOW_HOURS * 3600.0


class InvitationMailLimit:
    """Counts invitation emails per organization and per recipient address, in the rate-limit store.

    An invitation email carries text an organization chose (its name, the
    inviter's name) from the deployment's own sender, so on a deployment where
    anyone can create an organization it would otherwise be an open relay. The
    address limit holds across organizations, so neither one organization
    re-inviting an address nor several organizations together can flood it.
    """

    def __init__(
        self, store: RateLimitStorePort | None, *, per_organization: int | None, per_address: int | None
    ) -> None:
        self._store = store
        self._per_organization = per_organization
        self._per_address = per_address

    async def admit(self, organization_id: uuid.UUID, address: str) -> bool:
        """Count one email to ``address`` from ``organization_id``, or return False when either limit is spent.

        A refusal counts nothing against either limit.
        """
        if self._store is None:
            return True
        # Hashed, so the store never holds an address in the clear.
        address_key = f"invitation-mail:to:{hashlib.sha256(address.encode()).hexdigest()}"
        address_hit = None
        if self._per_address is not None:
            address_hit = await self._store.hit(address_key, self._per_address, _WINDOW_SEC)
            if not address_hit.allowed:
                return False
        if self._per_organization is not None:
            organization_hit = await self._store.hit(
                f"invitation-mail:org:{organization_id}", self._per_organization, _WINDOW_SEC
            )
            if not organization_hit.allowed:
                if address_hit is not None and address_hit.handle is not None:
                    await self._store.settle(address_key, address_hit.handle, 0)
                return False
        return True
