"""What an organization's own service key may be, apart from the budgets it lists.

An organization's service key belongs to the organization, not to whoever made it: it is owned by a user minted for it
alone, because end users belong to their key's user and a person's own user serves every organization they belong to.
"""

from gateway.exceptions.api_keys_exceptions import (
    EndUserSettingsOnPlainKeyError,
    MemberKeyNotServiceKeyError,
    OrganizationServiceKeyUncappedError,
    ServiceKeyFlagFixedError,
)
from gateway.models.api_keys import DEFAULT_ORGANIZATION_MAX_END_USERS, APIKey


def refuse_end_user_settings_on_plain_key(*, is_service_key: bool, sets_end_users: bool) -> None:
    """Refuse end-user budgets or a cap on a key that is not a service key, where they would never apply."""
    if sets_end_users and not is_service_key:
        raise EndUserSettingsOnPlainKeyError


def refuse_service_key_flag_change(key: APIKey, requested: bool | None) -> None:
    """Refuse turning an organization's key into a service key or back: its owner and end users depend on it."""
    if requested is not None and requested != key.is_service_key:
        raise ServiceKeyFlagFixedError


def refuse_member_key_as_service_key(key: APIKey, requested: bool | None) -> None:
    """Refuse an operator making a member's own key a service key, whose end users their other orgs would share."""
    if requested and key.self_service and not key.is_service_key:
        raise MemberKeyNotServiceKeyError


def organization_max_end_users(requested: int | None, *, explicit: bool = False) -> int:
    """The cap an organization's service key keeps: ``requested``, or the default when a create names none.

    ``explicit`` is an update that named the field, where null would remove the cap and is refused.
    """
    if requested is not None:
        return requested
    if explicit:
        raise OrganizationServiceKeyUncappedError
    return DEFAULT_ORGANIZATION_MAX_END_USERS
