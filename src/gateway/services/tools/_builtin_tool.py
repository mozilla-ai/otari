"""The shape of a tool the gateway runs itself, as the tool registry lists it."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from gateway.core.config import GatewayConfig
from gateway.services.tools._native import Dialect, NativeRendering


class NativeCheck(Protocol):
    """Whether ``provider`` runs ``tool_entry``, a declaration in some provider's own words, itself in ``dialect``."""

    def __call__(self, tool_entry: dict[str, Any] | None, *, provider: str | None, dialect: Dialect) -> bool: ...


@dataclass(frozen=True)
class BuiltinTool:
    """One tool the gateway runs itself when the model calls it.

    ``name`` is the function name the model calls.
    It is also the name a call is metered and priced under, so renaming a tool orphans its pricing and usage history.
    ``definition`` returns a new function definition in the Chat Completions shape on every call.
    ``configured`` answers whether this deployment has a backend that can run the tool.
    ``runs_natively`` answers, for one attempt's provider and dialect, whether that provider runs a
    declaration of the tool itself. ``None`` for the provider reads as not native.
    ``native`` holds the tool's rendering for each dialect it is announced in, and omits the
    rest. A dialect that is absent announces nothing through the registry.
    """

    name: str
    definition: Callable[[], dict[str, Any]]
    configured: Callable[[GatewayConfig], bool]
    runs_natively: NativeCheck = field(compare=False)
    native: Mapping[Dialect, NativeRendering] = field(default=MappingProxyType({}), compare=False)
