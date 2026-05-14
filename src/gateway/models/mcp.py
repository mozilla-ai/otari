"""Request-body models for inline MCP server configuration on /v1/chat/completions."""

from __future__ import annotations

from pydantic import BaseModel, Field


class McpServerConfig(BaseModel):
    """Inline MCP server configuration accepted on the chat completions request.

    Streamable HTTP transport. The `url` must be reachable from the gateway process
    (public, sidecar, cluster-internal, or localhost are all fine).
    """

    name: str = Field(min_length=1, max_length=128)
    url: str = Field(min_length=1)
    authorization_token: str | None = None
    purpose_hint: str | None = Field(default=None, max_length=2000)
    allowed_tools: list[str] | None = None
