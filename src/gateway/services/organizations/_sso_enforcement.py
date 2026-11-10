"""Whether some organization a person belongs to admits them only through its own connection.

Also ends the sessions such an organization has come to refuse, because the sessions belong to the
identity domain and the changes that newly refuse somebody do not.
"""

from typing import Protocol

from gateway.models.tenancy import Organization, User


class SsoEnforcement(Protocol):
    """Answers whether some organization a person actively belongs to refuses them a way in outside its own connection.

    The identity domain implements it. NOTE: an implementation reads through the request's Unit of Work, so call
    it before staging any write, or inside a block already open on that Unit of Work. ``revoke_refused_sessions``
    is the exception and is called after the write it follows, because it acts on what that write left behind.
    """

    async def refuses_every_membership(self, *, user: User) -> bool:
        """Whether every organization ``user`` actively belongs to refuses them a way in outside its connection."""
        ...

    async def blocks_membership(self, *, user: User, organization: Organization) -> bool:
        """Whether ``organization``'s own enforcement refuses ``user`` from acting in it."""
        ...

    async def revoke_refused_sessions(self, *, organization: Organization) -> None:
        """End the live session of every active member ``organization``'s enforcement now refuses.

        Callers hand this the organization after a write that may have changed who it refuses. An
        organization requiring nothing, or holding no connection, refuses nobody and loses no
        session.
        """
        ...
