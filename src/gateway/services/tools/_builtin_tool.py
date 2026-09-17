"""The shape of a tool the gateway runs itself, as the tool registry lists it."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from gateway.core.config import GatewayConfig


@dataclass(frozen=True)
class BuiltinTool:
    """One tool the gateway runs itself when the model calls it.

    ``name`` is the function name the model calls.
    It is also the name a call is metered and priced under, so renaming a tool orphans its pricing and usage history.
    ``definition`` returns a new function definition in the Chat Completions shape on every call.
    ``configured`` answers whether this deployment has a backend that can run the tool.
    """

    name: str
    definition: Callable[[], dict[str, Any]]
    configured: Callable[[GatewayConfig], bool]
