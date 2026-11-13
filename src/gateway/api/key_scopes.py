"""Which scope an API key needs to reach a route, declared on the route itself.

A route an API key may reach passes ``key_scope(...)`` as its ``openapi_extra``,
naming the scope it requires, or ``None`` where every key may reach it. The
declaration is read back at request time by the credential check, which refuses
a key on a route that declares nothing, so a route added without one fails
closed rather than open. It is also published in the OpenAPI document, as the
``x-otari-key-scopes`` extension on each operation: the scopes a key must hold,
empty where every key may reach the route.
"""

from typing import Any

from starlette.routing import BaseRoute

from gateway.types.key_scope import KeyScope

# NOTE: The x- prefix is OAS's rather than ours. RFC 6648 deprecated it for
# protocol parameters, which is why Otari's own headers are Otari-*, but an
# OpenAPI specification extension is a different namespace and OAS still
# requires ^x- as of 3.2.1.
KEY_SCOPE_EXTENSION = "x-otari-key-scopes"


class UndeclaredKeyScope:
    """A route that declares no key scope."""


UNDECLARED = UndeclaredKeyScope()


def key_scope(scope: KeyScope | None, /, **openapi_extra: Any) -> dict[str, Any]:
    """The ``openapi_extra`` of a route an API key may reach: ``scope`` to require, or ``None`` to admit every key.

    Any other ``openapi_extra`` the route needs is passed through alongside it.
    """
    return {**openapi_extra, KEY_SCOPE_EXTENSION: [] if scope is None else [scope.value]}


def declared_key_scope(route: BaseRoute | None) -> KeyScope | None | UndeclaredKeyScope:
    """The scope ``route`` declares, ``None`` where it admits every key, or ``UNDECLARED``."""
    extra = getattr(route, "openapi_extra", None) or {}
    if KEY_SCOPE_EXTENSION not in extra:
        return UNDECLARED
    required = extra[KEY_SCOPE_EXTENSION]
    return KeyScope(required[0]) if required else None
