"""Request and response models of the mcp-catalog domain."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SyncMode = Literal["off", "manual", "scheduled"]
SyncOutcome = Literal["complete", "partial", "failed"]
# How a featured entry compares with the catalog today: the same address, a different one, or no longer listed.
CatalogStatus = Literal["current", "changed", "removed"]


class McpCatalogServerPublic(BaseModel):
    """One remote MCP server the catalog lists."""

    id: uuid.UUID
    name: str = Field(description="The source's own identifier, such as `com.example/mcp`.")
    title: str
    description: str | None
    url: str = Field(description="The server's streamable HTTP endpoint.")
    requires_token: bool = Field(
        description=(
            "The source says a call needs an Authorization header. False means nothing was declared, not that "
            "the server is open: one that signs users in with OAuth declares nothing."
        )
    )
    website_url: str | None
    version: str | None
    source: str = Field(description="Where the entry came from. `registry` is the official MCP Registry.")


class McpCatalogServersPublic(BaseModel):
    """A page of catalog servers. `count` is how many match in all."""

    data: list[McpCatalogServerPublic]
    count: int


class McpCatalogFeaturedPublic(BaseModel):
    """A server the organization recommends, as it was when it was featured."""

    id: uuid.UUID
    name: str
    title: str
    description: str | None
    url: str
    requires_token: bool
    website_url: str | None
    source: str
    catalog_status: CatalogStatus = Field(
        description=(
            "`current` while the catalog lists this server at the same URL, `changed` once it lists another "
            "URL, `removed` once it lists none. The featured entry keeps the URL it was featured with."
        )
    )
    catalog_server_id: uuid.UUID | None = Field(
        description=(
            "The catalog server this entry was copied from, null once the catalog no longer lists it. "
            "Feature it again to take the copy afresh."
        )
    )
    created_at: datetime


class McpCatalogFeaturedListPublic(BaseModel):
    """An organization's featured shelf, by title."""

    data: list[McpCatalogFeaturedPublic]
    count: int


class McpCatalogFeatureRequest(BaseModel):
    """The catalog server to feature."""

    model_config = ConfigDict(extra="forbid")

    server_id: uuid.UUID = Field(description="The id of a catalog server.")


class McpCatalogSyncStatus(BaseModel):
    """Whether the catalog syncs, what the sync in progress or the last one did, and how many servers it holds."""

    mode: SyncMode = Field(description="The deployment's `mcp_catalog_sync` setting.")
    running: bool
    requested: bool = Field(description="A sync was asked for and no worker has started it yet.")
    outcome: SyncOutcome | None = Field(
        description=(
            "How the last sync ended, null before the first. `partial` kept the pages it read and removed "
            "nothing; the next sync resumes where it stopped."
        )
    )
    error: str | None = Field(description="Why the last sync was partial or failed.")
    started_at: datetime | None
    finished_at: datetime | None
    last_completed_at: datetime | None = Field(description="When a sync last read the whole registry.")
    next_sync_at: datetime | None = Field(description="When the next scheduled sync is due, if the mode is scheduled.")
    pages: int
    fetched: int = Field(description="Registry entries read.")
    kept: int = Field(description="Entries stored: remote servers Otari can connect to.")
    removed: int = Field(description="Servers a complete sync deleted because the registry no longer lists them.")
    server_count: int = Field(description="Servers in the catalog now.")


class McpCatalogEntry(BaseModel):
    """One server as the sync read it from its source, ready to store."""

    model_config = ConfigDict(frozen=True)

    name: str
    title: str
    description: str | None
    url: str
    requires_token: bool
    website_url: str | None
    version: str | None
    source: str
