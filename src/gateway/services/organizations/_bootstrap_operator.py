"""Which account is the deployment's bootstrap operator, and handing that role on."""

import uuid

from gateway.repositories.organizations import BootstrapMarkerRepository


class BootstrapOperatorMarker:
    """Read and move the bootstrap operator marker in the caller's transaction."""

    def __init__(self, marker: BootstrapMarkerRepository) -> None:
        self._marker = marker

    async def user_id(self) -> uuid.UUID | None:
        """The account the marker names now, read past the session's identity map."""
        return await self._marker.user_id()

    async def move(self, *, from_user_id: uuid.UUID, to_user_id: uuid.UUID) -> bool:
        """Name ``to_user_id`` if the marker still names ``from_user_id``; False if it no longer does."""
        return await self._marker.move(from_user_id=from_user_id, to_user_id=to_user_id)
