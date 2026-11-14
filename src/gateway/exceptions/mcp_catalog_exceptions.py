"""Errors that the mcp-catalog domain may raise.

Each carries the HTTP status it renders as.
"""

from fastapi import status

from gateway.exceptions import TenancyConflictError, TenancyError, TenancyForbiddenError


class McpCatalogSyncDisabledError(TenancyConflictError):
    """The deployment's ``mcp_catalog_sync`` setting is ``off``, so nothing may contact the registry."""

    def __init__(self) -> None:
        super().__init__("The MCP catalog is turned off on this deployment (mcp_catalog_sync is 'off')")


class McpCatalogSyncForbiddenError(TenancyForbiddenError):
    """The caller neither operates the deployment nor named a workspace they may add MCP servers to."""

    def __init__(self) -> None:
        super().__init__("Loading the MCP catalog takes a workspace you may add MCP servers to")


class McpCatalogSyncTooSoonError(TenancyError):
    """A sync ended too recently for a workspace admin to ask for another."""

    status_code = status.HTTP_429_TOO_MANY_REQUESTS

    def __init__(self, minutes: int) -> None:
        super().__init__(f"The MCP catalog was loaded recently; try again in {minutes} minutes")
