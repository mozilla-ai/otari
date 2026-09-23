"""The shape of a tool the gateway runs itself, as the tool registry lists it."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from gateway.core.config import GatewayConfig
from gateway.services.tools._native import Dialect, NativeRendering


@dataclass(frozen=True)
class BuiltinTool:
    """One tool the gateway runs itself when the model calls it.

    ``name`` is the function name the model calls.
    It is also the name a call is metered and priced under, so renaming a tool orphans its pricing and usage history.
    ``definition`` returns a new function definition in the Chat Completions shape on every call.
    ``configured`` answers whether this deployment has a backend that can run the tool.
    ``native`` holds the tool's rendering for each dialect it is announced in, and omits the
    rest. A dialect that is absent announces nothing through the registry.
    """

    name: str
    definition: Callable[[], dict[str, Any]]
    configured: Callable[[GatewayConfig], bool]
    native: Mapping[Dialect, NativeRendering] = field(default=MappingProxyType({}), compare=False)
