"""Request-body models for inline MCP server configuration on /v1/chat/completions."""

from __future__ import annotations

from pydantic import BaseModel, Field

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


class McpServerConfig(BaseModel):
    """Inline MCP server configuration accepted on the chat completions request.

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
