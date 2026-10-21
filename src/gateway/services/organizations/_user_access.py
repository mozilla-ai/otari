"""Users' model and search tool access lists, which a key's own lists may only narrow."""

from gateway.repositories.users_repository import AccessLists, UserAccessRepository


class UserAccess:
    """Read users' access lists. The reads run in the caller's Unit of Work block."""

    def __init__(self, users: UserAccessRepository) -> None:
        self._users = users

    async def lists(self, user_id: str) -> AccessLists | None:
        """A user's access lists, or ``None`` when there is no such user."""
        return await self._users.access_lists(user_id)

    async def live_lists(self) -> dict[str, AccessLists]:
        """Every live user's access lists, by user id."""
        return await self._users.live_access_lists()
