"""Wire and internal models for MCP server configuration."""

from __future__ import annotations

import hashlib
import json
import uuid

from pydantic import BaseModel, ConfigDict, Field, StrictBool

# Ceiling on ``mcp_server_ids`` in one request. A workspace can hold at most 50
# configured servers, in standalone (``MAX_MCP_SERVERS_PER_WORKSPACE`` in
# ``services/tenancy/workspace_mcp_server_service.py``) and on the platform,
# which uses the same value, so no reachable request loses anything. It is
# declared separately from that constant rather than imported: a wire bound has
# to be enforceable at parse time, with no database and no mode to consult, and
# a models module reaching into a service to learn its own limits would be the
# wrong direction. Without it a single request could hand the resolver an
# unbounded ``IN`` list, which is why the usage filters carry the same kind of
# ceiling (``core/sql.MAX_FILTER_VALUES``).
MAX_MCP_SERVER_IDS = 50

# Hex characters kept from the revision digest. 32 leaves collision risk far
# below the chance of a stored-server change going unnoticed for any other
# reason, and stays well inside the 1-to-128-character wire format (R-DISC-3).
REVISION_LENGTH = 32


class McpServerConfig(BaseModel):
    """Inline MCP server configuration accepted by generation requests.

    Streamable HTTP transport. The `url` must be reachable from the gateway process.

    URL safety (SSRF guard against private/link-local/reserved IP ranges, plus
    rejecting plain ``http://`` when ``authorization_token`` is set) is
    enforced by :func:`gateway.services.url_safety.validate_mcp_url`, called
    from the async request pipeline (``prepare_gateway_tools``) rather than
    here at parse time: the safety check does a DNS lookup, which must be
    awaited so it can't block the event loop, and Pydantic validators run
    synchronously during request-body parsing.
    """

    name: str = Field(min_length=1, max_length=128)
    url: str = Field(min_length=1)
    authorization_token: str | None = None
    purpose_hint: str | None = Field(default=None, max_length=2000)
    allowed_tools: list[str] | None = None


class ResolvedMcpServer(BaseModel):
    """One stored MCP server, resolved for the authenticated workspace.

    The internal counterpart of :class:`McpServerConfig`: the same connection
    details, plus the two fields the stored-server endpoints need and an inline
    request body cannot carry. ``id`` is what the exactly-one-matching-entry
    check compares, and ``enabled`` is what separates the disabled-server 404
    from a server the caller cannot see at all.

    Lives here rather than beside either resolver because both modes produce it:
    ``_platform._resolve_platform_mcp_server`` in hybrid mode and
    ``workspace_mcp_server_service.resolve_workspace_mcp_server`` in standalone,
    and a service may not import the API layer.
    """

    # Extra keys are ignored so a platform that grows a field does not break a
    # gateway that has not learned about it. ``enabled`` defaults true for legacy
    # peers that return only enabled server configs, but an explicit value is
    # strict: lax coercion would read a resolver that answered ``"no"`` as enabled.
    model_config = ConfigDict(extra="ignore")

    id: uuid.UUID
    name: str = Field(min_length=1, max_length=128)
    url: str = Field(min_length=1)
    authorization_token: str | None = None
    enabled: StrictBool = True
    purpose_hint: str | None = Field(default=None, max_length=2000)
    allowed_tools: list[str] | None = None

    @property
    def revision(self) -> str:
        """The opaque revision of this server's execution-relevant configuration.

        Derived rather than stored (R-RES-3): a pure function of the four fields
        that change what an execution does, so every worker and replica agrees
        by construction and the value moves atomically with the configuration it
        covers. ``name`` and ``purpose_hint`` are excluded, so retitling a server
        does not invalidate an authorization an application is still holding.

        The credential contributes only as a digest, and the whole value is a
        digest, so neither the token nor the URL can be read back out of a
        revision that travels to a caller.
        """
        canonical = json.dumps(
            {
                "url": self.url,
                "credential": _credential_fingerprint(self.authorization_token),
                "enabled": self.enabled,
                "allowed_tools": None if self.allowed_tools is None else sorted(self.allowed_tools),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:REVISION_LENGTH]


def _credential_fingerprint(token: str | None) -> str:
    """Distinguish credentials without carrying one into the revision input."""
    if token is None:
        return ""
    return hashlib.sha256(token.encode()).hexdigest()
