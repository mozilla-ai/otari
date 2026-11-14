"""Request and response models of the mcp-catalog domain."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SyncMode = Literal["off", "manual", "scheduled"]
SyncOutcome = Literal["complete", "partial", "failed"]
LogoToneValue = Literal["dark", "light", "tile"]
LOGO_TONE_DESCRIPTION = (
    "Where the logo needs a tile to stay visible: `dark` ink on a dark background, `light` ink on a light "
    "background, `tile` on every background. Null when it reads on both."
)


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
    icon_url: str | None = Field(
        description="The https image the registry entry names as the server's icon, null when it names none."
    )
    logo_url: str | None = Field(
        description=(
            "Where this gateway serves the server's logo, which it fetched once and keeps. Null when it looked "
            "and found none. A URL without `v` has not been looked for yet, and asking for it starts the look."
        )
    )
    logo_tone: LogoToneValue | None = Field(description=LOGO_TONE_DESCRIPTION)
    publisher_kind: Literal["vendor", "community"] = Field(
        description=(
            "`vendor` when the server is its maker's own: the endpoint lives on a domain the publisher owns, "
            "the server is named for that domain and nothing else, and the domain is not a hosting platform's. "
            "A few makers' servers the rule cannot recognize are listed by hand. `community` for every other, "
            "such as a gateway to another company's API. The rule reads names and vouches for nothing."
        )
    )
    version: str | None
    source: str = Field(
        description=(
            "Where the entry came from. `registry` is the official MCP Registry; `curated` is a short list "
            "bundled with Otari of well-known servers the registry does not list."
        )
    )


class McpCatalogServersPublic(BaseModel):
    """A page of catalog servers. `count` is how many match in all."""

    data: list[McpCatalogServerPublic]
    count: int


class McpCatalogSyncRequest(BaseModel):
    """Who asks for a sync, for a caller who does not operate the deployment."""

    model_config = ConfigDict(extra="forbid")

    workspace_id: uuid.UUID | None = Field(
        default=None,
        description=(
            "A workspace the caller may add MCP servers to. Required unless the caller operates the deployment."
        ),
    )


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
    is_stale: bool = Field(
        description="The catalog is empty, or no sync has read the whole registry into it in the last day."
    )
    next_request_at: datetime | None = Field(
        description=(
            "Before this time a workspace admin's sync request is refused with a 429, because a sync ended "
            "recently. Null when one may be asked for now. An operator is not held to it."
        )
    )


class McpCatalogEntry(BaseModel):
    """One server as the sync read it from its source, ready to store."""

    model_config = ConfigDict(frozen=True)

    name: str
    title: str
    description: str | None
    url: str
    requires_token: bool
    website_url: str | None
    icon_url: str | None = None
    version: str | None
    source: str
    is_official: bool = False
