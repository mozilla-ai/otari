"""The mcp-catalog domain lists remote MCP servers a workspace can add, synced from the official MCP Registry."""

from gateway.services.mcp_catalog._registry import (
    REGISTRY_URL,
    FetchPage,
    RegistryClient,
    RegistryError,
    RegistryPage,
    entry_from_registry_item,
    fallback_title,
)
from gateway.services.mcp_catalog._service import (
    MAX_FEATURED_PER_ORGANIZATION,
    McpCatalogService,
    OrganizationAccess,
)
from gateway.services.mcp_catalog._worker import run_mcp_catalog_sync_worker

__all__ = [
    "MAX_FEATURED_PER_ORGANIZATION",
    "REGISTRY_URL",
    "FetchPage",
    "McpCatalogService",
    "OrganizationAccess",
    "RegistryClient",
    "RegistryError",
    "RegistryPage",
    "entry_from_registry_item",
    "fallback_title",
    "run_mcp_catalog_sync_worker",
]
