"""Settings for the MCP server catalog and its sync from the MCP Registry."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from gateway.core.settings_view import SettingsGroup, Shown


class McpCatalogSettings(BaseModel):
    mcp_catalog_enabled: Annotated[bool, Shown(SettingsGroup.TOOLS)] = Field(
        default=True,
        description=(
            "Offer the MCP server catalog: Browse catalog on the MCP servers page, and the logos it knows. "
            "Off removes them, their routes and the sync worker. Requires restart."
        ),
    )
    mcp_catalog_sync: Annotated[Literal["off", "manual", "scheduled"], Shown(SettingsGroup.TOOLS)] = Field(
        default="manual",
        description=(
            "Whether the gateway copies the official MCP Registry (registry.modelcontextprotocol.io) into the "
            "catalog. 'off': never, for a deployment with no egress; the catalog keeps what it holds. "
            "'manual': only when someone who may add MCP servers loads or refreshes it from Browse catalog, "
            "or an operator asks through the API. 'scheduled': also every mcp_catalog_sync_interval_hours."
        ),
    )
    mcp_catalog_sync_interval_hours: Annotated[int, Shown(SettingsGroup.TOOLS)] = Field(
        default=24,
        ge=1,
        le=720,
        description="Hours between syncs when mcp_catalog_sync is scheduled.",
    )
