"""How a gateway-run tool call is announced in a wire format's own server-tool vocabulary.

A provider that runs a tool itself reports it in dedicated wire items rather than as
an ordinary tool call. The gateway runs the call instead, so a caller who asked in
that vocabulary is owed the same items back. A tool declares one rendering per
dialect that can describe it honestly, and none for a dialect that cannot.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gateway.services._tool_loop import ToolBackend


class Dialect(StrEnum):
    """A wire format the gateway answers in."""

    CHAT = "chat"
    MESSAGES = "messages"
    RESPONSES = "responses"


# Marks a Messages ``server_tool_use`` ID as one this gateway minted. Anthropic issues
# ``srvtoolu_``-prefixed IDs of its own, so a reserved prefix is what lets an echoed
# transcript be told apart from one describing a call the provider really ran.
SERVER_TOOL_USE_ID_PREFIX = "otari_srvtoolu_"


@dataclass(frozen=True)
class NativeCall:
    """One gateway-run tool call, as a rendering needs to describe it.

    ``id`` is the ID the caller's own transcript gave the call. A rendering that needs an
    ID of its own mints one carrying the gateway's prefix instead. ``failed`` is whether the
    call ran and answered with a tool error. ``token`` is minted once per call: a rendering
    that announces a call both when it starts and when it ends builds the ID of both from
    it, so a caller can match the two.
    """

    name: str
    id: str
    arguments: Mapping[str, Any]
    failed: bool = False
    token: str = field(default_factory=lambda: uuid.uuid4().hex, compare=False)


class NativeRendering(Protocol):
    """One tool's gateway-run calls, in one dialect's server-tool vocabulary."""

    def declared(self, tool_entry: Mapping[str, Any] | None) -> bool:
        """Whether a caller declaring the tool this way expects this rendering back."""
        ...

    def started(self, call: NativeCall) -> list[Any]:
        """Wire items announcing a call the gateway is about to run, empty where the dialect has none."""
        ...

    def ran(self, call: NativeCall, pool: ToolBackend) -> list[Any]:
        """Wire items announcing a call the gateway ran, empty where there is nothing to report."""
        ...

    def refused(self, call: NativeCall) -> list[Any]:
        """Wire items announcing a call the request's use cap refused, empty where the dialect has none."""
        ...
