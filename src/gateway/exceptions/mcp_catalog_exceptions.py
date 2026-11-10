"""Errors that the mcp-catalog domain may raise.

Each carries the HTTP status it renders as.
"""

import uuid

from gateway.exceptions import TenancyConflictError, TenancyNotFoundError


class McpCatalogServerNotFoundError(TenancyNotFoundError):
    def __init__(self, server_id: uuid.UUID):
        super().__init__(f"MCP catalog server '{server_id}' not found")


class McpCatalogFeaturedNotFoundError(TenancyNotFoundError):
    def __init__(self, featured_id: uuid.UUID):
        super().__init__(f"Featured MCP server '{featured_id}' not found")


class McpCatalogFeaturedLimitReachedError(TenancyConflictError):
    def __init__(self, limit: int):
        super().__init__(f"An organization features at most {limit} MCP servers; remove one first")


class McpCatalogSyncDisabledError(TenancyConflictError):
    """The deployment's ``mcp_catalog_sync`` setting is ``off``, so nothing may contact the registry."""

    def __init__(self) -> None:
        super().__init__("The MCP catalog sync is switched off on this deployment (mcp_catalog_sync is 'off')")
