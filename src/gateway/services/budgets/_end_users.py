"""The end users a service key bills, each under a budget of its own."""

import uuid
from datetime import UTC, datetime

from gateway.exceptions.budget_exceptions import (
    EndUserIdInvalidError,
    EndUserOwnerUnavailableError,
)
from gateway.models.api_keys import APIKey
from gateway.models.users import User
from gateway.repositories.budgets import BudgetRepositories
from gateway.services.budgets._periods import budget_window

# Long enough for an email address or an opaque id, short enough that a caller
# cannot use the column as storage.
MAX_EXTERNAL_ID_LENGTH = 256

# Generated ids carry a prefix that does not parse as a UUID, so an end user is
# never mistaken for a tenancy member's attribution row (see
# ``_scoped_enforcement._identity_uuid``).
END_USER_ID_PREFIX = "eu_"


class _EndUsers:
    """Find or create the end user a service key named."""

    def __init__(self, repositories: BudgetRepositories) -> None:
        self._repositories = repositories

    async def resolve(self, api_key: APIKey, external_id: str) -> str:
        """The ``users.user_id`` that ``external_id`` names under this key's owner.

        Created on first use under the key's end-user budget, and revived when it
        was soft-deleted, with its counters kept so that deleting an end user
        cannot clear what they spent.
        """
        if len(external_id) > MAX_EXTERNAL_ID_LENGTH:
            raise EndUserIdInvalidError(MAX_EXTERNAL_ID_LENGTH)
        owner_user_id = str(api_key.user_id)
        owner = await self._repositories.end_users.get(owner_user_id)
        if owner is None or owner.blocked or owner.deleted_at is not None:
            raise EndUserOwnerUnavailableError()

        existing = await self._repositories.end_users.find(owner_user_id, external_id)
        if existing is not None:
            if existing.deleted_at is not None:
                existing.deleted_at = None
            return existing.user_id

        user = User(
            user_id=f"{END_USER_ID_PREFIX}{uuid.uuid4().hex}",
            alias=external_id,
            parent_user_id=owner_user_id,
            external_id=external_id,
        )
        budget = (
            await self._repositories.budgets.get(api_key.end_user_budget_id) if api_key.end_user_budget_id else None
        )
        if budget is not None:
            now = datetime.now(UTC)
            window = budget_window(now, budget)
            user.budget_id = budget.budget_id
            user.budget_started_at, user.next_budget_reset_at = window if window is not None else (now, None)

        if await self._repositories.end_users.add(user):
            return user.user_id
        winner = await self._repositories.end_users.find(owner_user_id, external_id)
        if winner is None:
            raise RuntimeError("An end user's insert conflicted with a row that is not there")
        return winner.user_id
