"""Whether some organization a person belongs to admits them only through its own connection."""

import uuid
from collections.abc import Collection
from typing import Protocol

from gateway.models.tenancy import Organization, User


class SsoEnforcement(Protocol):
    """Answers whether some organization a person actively belongs to refuses them a way in outside its own connection.

    The identity domain implements it. NOTE: an implementation reads through the request's Unit of Work, so call
    it before staging any write, or inside a block already open on that Unit of Work.
    """

    async def blocks_any_membership(self, *, user: User) -> bool:
        """Whether some organization ``user`` actively belongs to requires sign-in through its own connection."""
        ...

    async def blocks_membership(self, *, user: User, organization: Organization) -> bool:
        """Whether ``organization``'s own enforcement refuses ``user`` from acting in it."""
        ...

    async def revoke_refused_sessions(self, *, organization_id: uuid.UUID, user_ids: Collection[uuid.UUID]) -> None:
        """End the live session of every listed account the organization refuses.

        NOTE: A caller that points an account at an organization owes this call, because nothing on
        the read path refuses a live session.
        """
        ...
