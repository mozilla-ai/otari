"""Which addresses the deployment refuses to hold an account for."""

from collections.abc import Collection
from typing import Protocol


class BlockedAddresses(Protocol):
    """Answers which of some addresses the deployment's email blocklist matches.

    The identity domain implements it. NOTE: an implementation reads through the request's Unit of Work, so call
    it before staging any write, or inside a block already open on that Unit of Work.
    """

    async def blocked_among(self, emails: Collection[str]) -> set[str]:
        """Return the members of ``emails`` the blocklist matches, spelled as they were given."""
        ...
