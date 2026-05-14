"""Request-body models for inline MCP server configuration on /v1/chat/completions."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from gateway.services.url_safety import UnsafeURLError, validate_mcp_url


class McpServerConfig(BaseModel):
    """Inline MCP server configuration accepted on the chat completions request.

    Streamable HTTP transport. The `url` must be reachable from the gateway process.

    URL safety is enforced at parse time:

    * SSRF guard rejects private, link-local, and reserved IP ranges. Loopback is
      allowed by default (sidecars, dev) — set ``GATEWAY_MCP_ALLOW_LOOPBACK=false`` to disable.
    * Plain ``http://`` is rejected when ``authorization_token`` is set, to keep
      bearer tokens off the wire in cleartext.
    """

    name: str = Field(min_length=1, max_length=128)
    url: str = Field(min_length=1)
    authorization_token: str | None = None
    purpose_hint: str | None = Field(default=None, max_length=2000)
    allowed_tools: list[str] | None = None

    @model_validator(mode="after")
    def _check_url_safety(self) -> "McpServerConfig":
        try:
            validate_mcp_url(self.url, has_authorization_token=bool(self.authorization_token))
        except UnsafeURLError as exc:
            raise ValueError(str(exc)) from exc
        return self
