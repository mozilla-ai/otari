"""What an API key may do: the one list of scopes a key can hold.

A scope names a group of endpoints by what a leaked key could do with them, so
a key holds the least it needs. Each endpoint an API key can reach requires
exactly one scope or is open to every key (read-only discovery: the model, tool
and price catalogs). The master key holds every scope.

Reads and writes are separate wherever a write can spend, change or destroy
something, and ``end-users:delete`` is separate from ``end-users:write`` because
deleting an end user erases its records. No scope lets a key manage keys:
a key that could mint keys would outlive its own revocation.
"""

from collections.abc import Iterable
from enum import StrEnum


class KeyScope(StrEnum):
    """One group of endpoints an API key may be granted, named ``resource:action``.

    Ordered by resource, not A-Z: the order is what a stored list is normalized
    to and what the dashboard renders, and the end-user scopes come last because
    only a service key may hold them.
    """

    INFERENCE_RUN = "inference:run"
    FILES_READ = "files:read"
    FILES_WRITE = "files:write"
    BATCHES_READ = "batches:read"
    BATCHES_WRITE = "batches:write"
    WEB_SEARCH = "web:search"
    WEB_FETCH = "web:fetch"
    CODE_EXECUTE = "code:execute"
    MCP_READ = "mcp:read"
    MCP_EXECUTE = "mcp:execute"
    TELEMETRY_WRITE = "telemetry:write"
    AGENTS_WRITE = "agents:write"
    USAGE_READ = "usage:read"
    USAGE_WRITE = "usage:write"
    END_USERS_READ = "end-users:read"
    END_USERS_WRITE = "end-users:write"
    END_USERS_DELETE = "end-users:delete"


# What a key is given when its creator names no scopes: calling models, which is
# what most keys are for.
DEFAULT_KEY_SCOPES: tuple[KeyScope, ...] = (KeyScope.INFERENCE_RUN,)

# Only a service key has end users, so only a service key may hold these.
END_USER_SCOPES: frozenset[KeyScope] = frozenset(
    {KeyScope.END_USERS_READ, KeyScope.END_USERS_WRITE, KeyScope.END_USERS_DELETE}
)

# Every scope that existed when keys were first scoped, which is what a key that
# predates scopes could reach, and what a config.yml key naming none still gets.
# Frozen: a scope added later is never granted to a key that did not ask for it,
# so this list must not grow with the enum.
UNSCOPED_KEY_SCOPES: tuple[KeyScope, ...] = (
    KeyScope.INFERENCE_RUN,
    KeyScope.FILES_READ,
    KeyScope.FILES_WRITE,
    KeyScope.BATCHES_READ,
    KeyScope.BATCHES_WRITE,
    KeyScope.WEB_SEARCH,
    KeyScope.WEB_FETCH,
    KeyScope.CODE_EXECUTE,
    KeyScope.MCP_READ,
    KeyScope.MCP_EXECUTE,
    KeyScope.TELEMETRY_WRITE,
    KeyScope.AGENTS_WRITE,
    KeyScope.USAGE_READ,
    KeyScope.USAGE_WRITE,
    KeyScope.END_USERS_READ,
    KeyScope.END_USERS_WRITE,
    KeyScope.END_USERS_DELETE,
)


def unscoped_key_scopes(*, is_service_key: bool) -> list[str]:
    """What a key that names no scopes of its own holds: every original scope its kind of key may hold."""
    return normalize_scopes(scope for scope in UNSCOPED_KEY_SCOPES if is_service_key or scope not in END_USER_SCOPES)


def normalize_scopes(scopes: Iterable[KeyScope | str]) -> list[str]:
    """``scopes`` as stored: each once, in the enum's order.

    Raises:
        ValueError: an entry names no scope.
    """
    held = {KeyScope(scope) for scope in scopes}
    return [scope.value for scope in KeyScope if scope in held]


def checked_scopes(scopes: Iterable[KeyScope | str], *, is_service_key: bool) -> list[str]:
    """``scopes`` as a key of this kind stores them, normalized.

    Raises:
        ValueError: ``scopes`` is empty or names one this kind of key may not hold.
    """
    normalized = normalize_scopes(scopes)
    if not normalized:
        raise ValueError("A key needs at least one scope.")
    misplaced = [scope for scope in normalized if KeyScope(scope) in END_USER_SCOPES]
    if misplaced and not is_service_key:
        raise ValueError(f"Only a service key may hold {', '.join(misplaced)}: a key that is not one has no end users.")
    return normalized


def updated_scopes(
    stored: list[str], requested: Iterable[KeyScope | str] | None, *, was_service_key: bool, is_service_key: bool
) -> list[str]:
    """The scopes a key holds after an update naming ``requested`` (None leaves them) and whether it is a service key.

    Checked against what the key is after the update, so one request may make a key a service key and grant it the
    end-user scopes. A key that stops being one loses those scopes rather than keeping them for the day it is one again.

    Raises:
        ValueError: as ``checked_scopes`` does, or turning a service key into a plain one leaves it with no scope.
    """
    if requested is not None:
        return checked_scopes(requested, is_service_key=is_service_key)
    if is_service_key or not was_service_key:
        return stored
    end_user_scopes = {scope.value for scope in END_USER_SCOPES}
    return checked_scopes((scope for scope in stored if scope not in end_user_scopes), is_service_key=False)


def known_scopes(stored: Iterable[str]) -> list[KeyScope]:
    """The stored scopes this gateway knows, leaving out any a newer one wrote before a downgrade."""
    return [KeyScope(scope) for scope in stored if scope in KeyScope._value2member_map_]
