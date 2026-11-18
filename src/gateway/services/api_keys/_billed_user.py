"""Who a request is billed to: the user its API key is bound to, or the one the master key names."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.models.api_keys import APIKey


# What a pass-through request is answered with when resolve_user_id refuses it.
MASTER_KEY_USER_REQUIRED_DETAIL = "When using master key, 'user' field is required in request body"
API_KEY_VALIDATION_FAILED_DETAIL = "API key validation failed"
API_KEY_NO_USER_DETAIL = "API key has no associated user"
USER_MISMATCH_DETAIL = "'user' field does not match the authenticated API key's user"


def resolve_user_id(
    user_id_from_request: str | None,
    api_key: APIKey | None,
    is_master_key: bool,
    *,
    master_key_error: Exception,
    no_api_key_error: Exception,
    no_user_error: Exception,
    forbidden_user_error: Exception,
    reject_mismatch: bool = True,
) -> str:
    """Resolve the effective user_id from request context.

    The resolution order is:
    1. If master key is used, the request *must* supply a user_id, and may
       name any user (the master key is trusted to act on behalf of others).
    2. For a non-master key, spend is *always* bound to the key's own user.
       The request may echo the same user_id (e.g. OpenAI's ``user`` field for
       tracking). Naming a *different* user is rejected by default. That refusal
       is a misuse signal, not a budget control: spend binds to the key's own user
       whether the name is rejected or ignored.

    Args:
        user_id_from_request: User identifier extracted from the request body
        api_key: Authenticated API key object (None when using master key)
        is_master_key: Whether the request was authenticated with a master key
        master_key_error: Raised when master key is used but no user_id is provided
        no_api_key_error: Raised when no API key is available
        no_user_error: Raised when the API key has no associated user
        forbidden_user_error: Raised when a non-master key names a user other
            than its own (only when ``reject_mismatch`` is True)
        reject_mismatch: The deployment-wide default. When True, a non-master key
            naming a different user is rejected. When False, the mismatch is
            ignored and spend is still bound to the key's own user (the client
            ``user`` is treated as a provider-side tag only). A key whose own
            ``reject_user_mismatch`` is not None overrides this in either
            direction. Spend is bound to the key's user however this resolves;
            leniency never lets a key charge another user.

    Returns:
        Resolved user_id string

    """
    if is_master_key:
        if not user_id_from_request:
            raise master_key_error
        return user_id_from_request

    if api_key is None:
        raise no_api_key_error
    if not api_key.user_id:
        raise no_user_error
    key_user_id = str(api_key.user_id)

    # A non-master key is bound to its own user. Allow the request to echo that
    # same id; a different id is rejected (strict) or ignored (lenient); either
    # way spend binds to key_user_id, so a key can never charge another user.
    # A key may override the deployment-wide default in either direction (NULL =
    # inherit), so one client whose ``user`` field is telemetry can be let through
    # without relaxing the check for every key, and a deployment that relaxed it
    # globally can still pin an individual key strict.
    if api_key.reject_user_mismatch is not None:
        reject_mismatch = api_key.reject_user_mismatch
    if reject_mismatch and user_id_from_request and user_id_from_request != key_user_id:
        raise forbidden_user_error

    return key_user_id


def names_end_user(api_key: APIKey | None, user_id_from_request: str | None) -> bool:
    """Whether a request names an end user of its service key: anyone but the key's own user."""
    return bool(
        api_key is not None
        and api_key.is_service_key
        and api_key.user_id
        and user_id_from_request
        and user_id_from_request != api_key.user_id
    )
