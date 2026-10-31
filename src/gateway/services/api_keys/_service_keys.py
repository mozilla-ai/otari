"""What an organization's own service key may be, apart from the budgets it lists.

An organization's service key belongs to the organization, not to whoever made it: it is owned by a user minted for it
alone, because end users belong to their key's user and a person's own user serves every organization they belong to.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from gateway.exceptions.api_keys_exceptions import (
    EndUserSettingsOnPlainKeyError,
    OrganizationServiceKeyUncappedError,
    ServiceKeyFlagFixedError,
)
from gateway.models.api_keys import APIKey

if TYPE_CHECKING:
    # Type-only: the budgets service imports this package, so a runtime import would be a cycle.
    from gateway.models.tenancy import Organization, User
    from gateway.services.budgets import BudgetService
    from gateway.services.tenancy import OrganizationService

# The end users an organization's service key may have until the organization
# sets another cap. Each new id a request names creates a row, so an
# organization's key always has a cap; only an operator may leave one off.
DEFAULT_ORGANIZATION_MAX_END_USERS = 10_000

END_USER_SETTINGS = frozenset({"end_user_budget_id", "end_user_budget_ids", "max_end_users"})


def refuse_end_user_settings_on_plain_key(*, is_service_key: bool, sets_end_users: bool) -> None:
    """Refuse end-user budgets or a cap on a key that is not a service key, where they would never apply."""
    if sets_end_users and not is_service_key:
        raise EndUserSettingsOnPlainKeyError


def refuse_service_key_flag_change(key: APIKey, requested: bool | None) -> None:
    """Refuse turning a key a member or an organization made into a service key or back, from either route.

    An organization's service key has an owner of its own that its end users depend on, and a member's key is owned by
    the member's own user, which every organization they belong to shares, so neither may change what it is.
    """
    if key.self_service and requested is not None and requested != key.is_service_key:
        raise ServiceKeyFlagFixedError


def organization_max_end_users(requested: int | None, *, explicit: bool = False) -> int:
    """The cap an organization's service key keeps: ``requested``, or the default when a create names none.

    ``explicit`` is an update that named the field, where null would remove the cap and is refused.
    """
    if requested is not None:
        return requested
    if explicit:
        raise OrganizationServiceKeyUncappedError
    return DEFAULT_ORGANIZATION_MAX_END_USERS


async def admit_organization_key(
    organizations: "OrganizationService",
    budgets: "BudgetService",
    *,
    user: "User",
    organization: "Organization",
    is_service_key: bool,
    requested: Mapping[str, Any],
) -> dict[str, Any]:
    """The end-user columns a key an organization creates starts with, once the caller may create it.

    ``requested`` holds the request's end-user settings. A service key takes an owner or admin and this organization's
    budgets, and always gets a cap; a plain key takes none of the settings. Runs before anything is staged, since the
    budget check commits a Unit of Work block.
    """
    refuse_end_user_settings_on_plain_key(
        is_service_key=is_service_key, sets_end_users=any(value is not None for value in requested.values())
    )
    if not is_service_key:
        return {}
    await organizations.require_active_organization_management_access(user=user, organization=organization)
    return {
        "end_user_budget_id": requested.get("end_user_budget_id"),
        "end_user_budget_ids": await budgets.check_end_user_budgets(
            requested.get("end_user_budget_ids"),
            requested.get("end_user_budget_id"),
            check_default=True,
            organization_id=organization.id,
        ),
        "max_end_users": organization_max_end_users(requested.get("max_end_users")),
    }


async def organization_key_changes(
    organizations: "OrganizationService",
    budgets: "BudgetService",
    *,
    user: "User",
    key: APIKey,
    requested: Mapping[str, Any],
    revoking: bool,
) -> dict[str, Any]:
    """The end-user columns an update to an organization's key changes, from the settings it named in ``requested``.

    Only an organization's own service key takes them, on that organization's budgets, and its cap cannot be cleared.
    The budget check's commit releases the workspace lock the caller's load took, so it is taken again.
    """
    if not requested:
        return {}
    refuse_end_user_settings_on_plain_key(is_service_key=key.is_organization_service_key(), sets_end_users=True)
    organization = await organizations.get_active_organization_for_user(user)
    changes: dict[str, Any] = {
        "end_user_budget_ids": await budgets.check_end_user_budgets(
            requested["end_user_budget_ids"] if "end_user_budget_ids" in requested else key.end_user_budget_ids,
            requested["end_user_budget_id"] if "end_user_budget_id" in requested else key.end_user_budget_id,
            check_default="end_user_budget_id" in requested,
            check_list="end_user_budget_ids" in requested,
            organization_id=organization.id,
        )
    }
    if "end_user_budget_id" in requested:
        changes["end_user_budget_id"] = requested["end_user_budget_id"]
    if "max_end_users" in requested:
        changes["max_end_users"] = organization_max_end_users(requested["max_end_users"], explicit=True)
    await organizations.hold_workspace_write_access(
        user=user, organization_id=organization.id, workspace_id=key.workspace_id, revoking=revoking
    )
    return changes
