from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import HTTPException

    from gateway.db import APIKey


def resolve_user_id(
    user_id_from_request: str | None,
    api_key: APIKey | None,
    is_master_key: bool,
    *,
    master_key_error: HTTPException,
    no_api_key_error: HTTPException,
    no_user_error: HTTPException,
    forbidden_user_error: HTTPException,
    reject_mismatch: bool = True,
) -> str:
    """Resolve the effective user_id from request context.

    The resolution order is:
    1. If master key is used, the request *must* supply a user_id, and may
       name any user (the master key is trusted to act on behalf of others).
    2. For a non-master key, spend is *always* bound to the key's own user.
       The request may echo the same user_id (e.g. OpenAI's ``user`` field for
       tracking), but naming a *different* user is rejected — otherwise any key
       could charge spend to, and exhaust the budget of, another user.

    Args:
        user_id_from_request: User identifier extracted from the request body
        api_key: Authenticated API key object (None when using master key)
        is_master_key: Whether the request was authenticated with a master key
        master_key_error: Raised when master key is used but no user_id is provided
        no_api_key_error: Raised when no API key is available
        no_user_error: Raised when the API key has no associated user
        forbidden_user_error: Raised when a non-master key names a user other
            than its own (only when ``reject_mismatch`` is True)
        reject_mismatch: When True (default), a non-master key naming a different
            user is rejected. When False, the mismatch is ignored and spend is
            still bound to the key's own user (the client ``user`` is treated as
            a provider-side tag only). Spend is bound to the key's user either
            way — leniency never lets a key charge another user.

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
    # same id; a different id is rejected (strict) or ignored (lenient) — either
    # way spend binds to key_user_id, so a key can never charge another user.
    if reject_mismatch and user_id_from_request and user_id_from_request != key_user_id:
        raise forbidden_user_error

    return key_user_id
