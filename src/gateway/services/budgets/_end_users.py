"""The end users a service key bills, each under a budget of its own."""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from gateway.exceptions.budget_exceptions import (
    EndUserBudgetNotAllowedError,
    EndUserBudgetNotFoundError,
    EndUserIdInvalidError,
    EndUserNotFoundError,
    EndUserOwnerUnavailableError,
    NotAServiceKeyError,
)
from gateway.models.api_keys import APIKey
from gateway.models.budgets import Budget
from gateway.models.users import User
from gateway.repositories.budgets import BudgetRepositories
from gateway.schemas.budgets import EndUserPublic
from gateway.services.budgets._periods import budget_window

# Long enough for an email address or an opaque id, short enough that a caller
# cannot use the column as storage.
MAX_EXTERNAL_ID_LENGTH = 256

# Generated ids carry a prefix that does not parse as a UUID, so an end user is
# never mistaken for a tenancy member's attribution row (see
# ``_scoped_enforcement._identity_uuid``).
END_USER_ID_PREFIX = "eu_"


@dataclass(frozen=True)
class ResolvedEndUser:
    """The end user a request bills, and the budget it is on."""

    user_id: str
    budget_id: str | None


class _EndUsers:
    """Find, create and manage the end users of a service key's owner."""

    def __init__(self, repositories: BudgetRepositories) -> None:
        self._repositories = repositories

    async def require_assignable_budget(self, budget_id: str) -> None:
        """Refuse an end-user budget that does not exist or that a tenant owns."""
        await self._assignable_budget(budget_id)

    async def resolve(self, api_key: APIKey, external_id: str, requested_budget_id: str | None) -> ResolvedEndUser:
        """The end user that ``external_id`` names under this key's owner.

        Created on first use on ``requested_budget_id``, or on the key's default
        when the request named none, and revived when it was soft-deleted, with
        its counters kept so that deleting an end user cannot clear what they
        spent. An existing end user keeps its budget whatever a request names, so
        a request cannot undo an operator's move; a budget the key may not assign
        is refused either way.
        """
        _check_external_id(external_id)
        requested = await self._listed_budget(api_key, requested_budget_id) if requested_budget_id else None
        owner_user_id = str(api_key.user_id)
        owner = await self._repositories.end_users.get(owner_user_id)
        if owner is None or owner.blocked or owner.deleted_at is not None:
            raise EndUserOwnerUnavailableError()

        existing = await self._repositories.end_users.find(owner_user_id, external_id)
        if existing is not None:
            if existing.deleted_at is not None:
                existing.deleted_at = None
            return ResolvedEndUser(existing.user_id, existing.budget_id)

        budget = requested
        if budget is None and api_key.end_user_budget_id:
            budget = await self._repositories.budgets.get(api_key.end_user_budget_id)
        user = _new_end_user(owner_user_id, external_id, budget)
        if await self._repositories.end_users.add(user):
            return ResolvedEndUser(user.user_id, user.budget_id)
        winner = await self._lost_insert(owner_user_id, external_id)
        return ResolvedEndUser(winner.user_id, winner.budget_id)

    async def get(self, api_key: APIKey, external_id: str) -> EndUserPublic:
        """The key owner's end user named ``external_id``."""
        return EndUserPublic.from_model(await self._existing(api_key, external_id))

    async def put(self, api_key: APIKey, external_id: str, budget_id: str) -> tuple[EndUserPublic, bool]:
        """Put the end user on ``budget_id``, creating it when there is none, and say whether it was created.

        An end user already on the budget keeps its period, so repeating the call changes nothing.
        """
        _require_service_key(api_key)
        _check_external_id(external_id)
        budget = await self._listed_budget(api_key, budget_id)
        owner_user_id = str(api_key.user_id)
        user = await self._repositories.end_users.find(owner_user_id, external_id)
        if user is None:
            user = _new_end_user(owner_user_id, external_id, budget)
            if await self._repositories.end_users.add(user):
                return EndUserPublic.from_model(user), True
            user = await self._lost_insert(owner_user_id, external_id)
        user.deleted_at = None
        if user.budget_id != budget.budget_id:
            _put_on_budget(user, budget)
        return EndUserPublic.from_model(user), False

    async def update(
        self, api_key: APIKey, external_id: str, *, blocked: bool | None, budget_id: str | None
    ) -> EndUserPublic:
        """Block, unblock or move one end user of the key's owner."""
        user = await self._existing(api_key, external_id)
        if budget_id is not None:
            budget = await self._listed_budget(api_key, budget_id)
            if user.budget_id != budget.budget_id:
                _put_on_budget(user, budget)
        if blocked is not None:
            user.blocked = blocked
        return EndUserPublic.from_model(user)

    async def _assignable_budget(self, budget_id: str) -> Budget:
        budget = await self._repositories.budgets.get(budget_id)
        if budget is None or budget.organization_id is not None:
            raise EndUserBudgetNotFoundError(budget_id)
        return budget

    async def _listed_budget(self, api_key: APIKey, budget_id: str) -> Budget:
        """``budget_id``, refused unless it is on the key's list of end-user budgets."""
        if budget_id not in api_key.assignable_end_user_budgets():
            raise EndUserBudgetNotAllowedError(budget_id)
        return await self._assignable_budget(budget_id)

    async def _existing(self, api_key: APIKey, external_id: str) -> User:
        _require_service_key(api_key)
        user = await self._repositories.end_users.find(str(api_key.user_id), external_id)
        if user is None or user.deleted_at is not None:
            raise EndUserNotFoundError(external_id)
        return user

    async def _lost_insert(self, owner_user_id: str, external_id: str) -> User:
        """The end user a concurrent request created first."""
        winner = await self._repositories.end_users.find(owner_user_id, external_id)
        if winner is None:
            raise RuntimeError("An end user's insert conflicted with a row that is not there")
        return winner


def _check_external_id(external_id: str) -> None:
    if len(external_id) > MAX_EXTERNAL_ID_LENGTH:
        raise EndUserIdInvalidError(MAX_EXTERNAL_ID_LENGTH)


def _require_service_key(api_key: APIKey) -> None:
    if not api_key.is_service_key or not api_key.user_id:
        raise NotAServiceKeyError(str(api_key.id))


def _put_on_budget(user: User, budget: Budget) -> None:
    """Assign ``budget`` with a period starting now, as the users API does."""
    now = datetime.now(UTC)
    window = budget_window(now, budget)
    user.budget_id = budget.budget_id
    user.budget_started_at, user.next_budget_reset_at = window if window is not None else (now, None)


def _new_end_user(owner_user_id: str, external_id: str, budget: Budget | None) -> User:
    user = User(
        user_id=f"{END_USER_ID_PREFIX}{uuid.uuid4().hex}",
        alias=external_id,
        parent_user_id=owner_user_id,
        external_id=external_id,
    )
    if budget is not None:
        _put_on_budget(user, budget)
    return user
