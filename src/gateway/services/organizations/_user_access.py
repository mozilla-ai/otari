"""Users' model and search tool access lists, which a key's own lists may only narrow, their names and budget use."""

from collections.abc import Collection

from gateway.repositories.organizations import OrganizationRepositories
from gateway.repositories.users_repository import AccessLists, BudgetUse


class UserAccess:
    """Read users' access lists, aliases and budget use. The reads run in the caller's Unit of Work block."""

    def __init__(self, repositories: OrganizationRepositories) -> None:
        self._users = repositories.users

    async def ensure_app_key_owner(self, user_id: str, alias: str) -> bool:
        """Ensure a live attribution owner exists without reviving a blocked or deleted one."""
        return await self._users.ensure_app_key_owner(user_id, alias)

    async def lists(self, user_id: str) -> AccessLists | None:
        """A user's access lists, or ``None`` when there is no such user."""
        return await self._users.access_lists(user_id)

    async def live_lists(self) -> dict[str, AccessLists]:
        """Every live user's access lists, by user id."""
        return await self._users.live_access_lists()

    async def budget_use(self, user_id: str) -> BudgetUse | None:
        """The budget a live user is on and the user's use of it, or None when it is on none."""
        return await self._users.budget_use(user_id)

    async def aliases(self, user_ids: Collection[str]) -> dict[str, str | None]:
        """Each of these users that has a row, deleted or not, to its alias."""
        return await self._users.aliases(user_ids)
