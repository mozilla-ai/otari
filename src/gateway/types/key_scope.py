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

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Self


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


@dataclass(frozen=True)
class KeyScopeSet:
    """The scopes one key holds, each once and in the enum's order.

    Built only through the constructors below, so a set that exists has been
    through the rule its constructor names. ``in`` answers whether a key may
    reach a scope, and iteration yields the scopes in order.
    """

    _held: frozenset[KeyScope]

    @classmethod
    def read(cls, stored: Iterable[str]) -> Self:
        """What a stored list names here, leaving out any scope a newer gateway wrote before a downgrade.

        NOTE: an unknown scope is dropped rather than refused, so a set read back may hold fewer than were written.
        """
        return cls(frozenset(KeyScope(scope) for scope in stored if scope in KeyScope))

    @classmethod
    def checked(cls, scopes: Iterable[KeyScope | str], *, is_service_key: bool) -> Self:
        """What a key of this kind may hold, from a caller's own list.

        Raises:
            ValueError: ``scopes`` is empty or names one this kind of key may not hold, or no scope at all.
        """
        held = cls(frozenset(KeyScope(scope) for scope in scopes))
        if not held:
            raise ValueError("A key needs at least one scope.")
        misplaced = [scope.value for scope in held if scope in END_USER_SCOPES]
        if misplaced and not is_service_key:
            msg = f"Only a service key may hold {', '.join(misplaced)}: a key that is not one has no end users."
            raise ValueError(msg)
        return held

    @classmethod
    def unscoped(cls, *, is_service_key: bool) -> Self:
        """What a key that names none of its own holds: every original scope its kind of key may hold."""
        return cls(frozenset(s for s in UNSCOPED_KEY_SCOPES if is_service_key or s not in END_USER_SCOPES))

    def updated(
        self, requested: Iterable[KeyScope | str] | None, *, was_service_key: bool, is_service_key: bool
    ) -> Self:
        """What the key holds after an update naming ``requested``, which ``None`` leaves as it is.

        Checked against what the key is after the update, so one request may make a key a service key and
        grant it the end-user scopes.
        A key that stops being one loses those scopes rather than keeping them for the day it is one again.

        Raises:
            ValueError: as ``checked`` does, or turning a service key into a plain one leaves it with no scope.
        """
        if requested is not None:
            return type(self).checked(requested, is_service_key=is_service_key)
        if is_service_key or not was_service_key:
            return self
        return type(self).checked((s for s in self if s not in END_USER_SCOPES), is_service_key=False)

    @property
    def stored(self) -> list[str]:
        """The set as a column holds it."""
        return [scope.value for scope in self]

    def __and__(self, other: Iterable[KeyScope]) -> Self:
        """The scopes this set and ``other`` share, which can only be fewer than either."""
        return type(self)(self._held & frozenset(other))

    def __contains__(self, scope: object) -> bool:
        return scope in self._held

    def __iter__(self) -> Iterator[KeyScope]:
        return (scope for scope in KeyScope if scope in self._held)

    def __len__(self) -> int:
        return len(self._held)
