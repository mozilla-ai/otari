"""The mcp-catalog domain lists remote MCP servers a workspace can add, synced from the official MCP Registry.

It also finds each server's logo, the catalog's or one from a server's own host, and serves it from this gateway.
"""

from gateway.services.mcp_catalog._logos import Logo, LogoTile, LogoTone
from gateway.services.mcp_catalog._registry import FetchPage, RegistryError, RegistryPage
from gateway.services.mcp_catalog._service import (
    LogoLookBudget,
    LogoRef,
    McpCatalogService,
    WorkspaceManagement,
    WorkspaceServers,
)

__all__ = [
    "FetchPage",
    "Logo",
    "LogoLookBudget",
    "LogoRef",
    "LogoTile",
    "LogoTone",
    "McpCatalogService",
    "RegistryError",
    "RegistryPage",
    "WorkspaceManagement",
    "WorkspaceServers",
]
