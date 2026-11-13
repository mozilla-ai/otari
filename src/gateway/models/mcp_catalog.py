"""ORM tables for the MCP server catalog: the synced servers, the sync's state, and the logos the gateway fetched."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import Boolean, CheckConstraint, Float, Integer, LargeBinary, String, Text, Uuid, false, func
from sqlalchemy.orm import Mapped, mapped_column

from gateway.models.base import Base, UtcDateTime

SYNC_STATE_ROW_ID = 1

# Where a catalog entry came from: the MCP Registry, or the short list bundled with Otari of well-known
# servers the registry does not list.
SOURCE_REGISTRY = "registry"
SOURCE_CURATED = "curated"

# How the last sync ended. Null on the row until one has.
OUTCOME_COMPLETE = "complete"
OUTCOME_PARTIAL = "partial"
OUTCOME_FAILED = "failed"


class McpCatalogServer(Base):
    """One remote MCP server the catalog knows, as the sync last read it.

    Deployment-wide and owned by the sync: a complete sync rewrites every row
    it sees and deletes the ones it no longer does, and stores the bundled
    list again each time it starts. Nothing a person decided lives here.
    """

    __tablename__ = "mcp_catalog_servers"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # The source's own identifier, reverse-DNS in the MCP Registry (``com.example/mcp``).
    name: Mapped[str] = mapped_column(String, unique=True)
    title: Mapped[str] = mapped_column(String)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    # Indexed: the Popular shelf and the logo lookup find servers by URL.
    url: Mapped[str] = mapped_column(String, index=True)
    # The source says a call needs an Authorization header. False is "not declared", not "open".
    requires_token: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    website_url: Mapped[str | None] = mapped_column(String, default=None)
    # An https image the registry entry itself names as the server's icon: the first place its logo is looked for.
    icon_url: Mapped[str | None] = mapped_column(String, default=None)
    version: Mapped[str | None] = mapped_column(String, default=None)
    source: Mapped[str] = mapped_column(String, default=SOURCE_REGISTRY, server_default=SOURCE_REGISTRY)
    # Its maker's own server, as the sync that stored it judged. Kept only so a search can list those first.
    is_official: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    # The start of the sync that last listed this server, which is how a finished sync finds the ones it did not.
    last_seen_at: Mapped[datetime] = mapped_column(UtcDateTime(), index=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=lambda: datetime.now(UTC))


class McpCatalogSyncState(Base):
    """The one row that says whether a sync is running, what the last one did, and where to resume.

    Also the claim: every replica's worker polls this row, and a sync starts
    only for the replica whose conditional update set ``running``. The counters
    describe the sync in progress, or the last one once it has ended.
    """

    __tablename__ = "mcp_catalog_sync_state"
    __table_args__ = (CheckConstraint(f"id = {SYNC_STATE_ROW_ID}", name="ck_mcp_catalog_sync_state_one_row"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=SYNC_STATE_ROW_ID)
    sync_requested: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    running: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    # Moved by every page a running sync stores. A running row whose heartbeat stopped belongs to a replica that died.
    heartbeat_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), default=None)
    next_sync_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), default=None)
    # When the crawl these counters describe began. A resumed crawl keeps it, so its rows share one ``last_seen_at``.
    crawl_started_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), default=None)
    # The source's cursor for the next page, kept while a crawl is unfinished so the next sync resumes there.
    cursor: Mapped[str | None] = mapped_column(String, default=None)
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), default=None)
    last_completed_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), default=None)
    outcome: Mapped[str | None] = mapped_column(String, default=None)
    error: Mapped[str | None] = mapped_column(String, default=None)
    pages: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    fetched: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    kept: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    removed: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime(),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        server_default=func.now(),
    )


# Whether the last look for a logo found one.
LOGO_OK = "ok"
LOGO_MISSING = "missing"


class McpLogo(Base):
    """A logo the gateway fetched once and serves from its own origin, or the record that none was found.

    Keyed by what it is the logo of: a catalog server by name (``catalog:com.example/mcp``), or the host
    of a server added by URL (``host:mcp.example.com``). Deployment-wide, and never written by a person:
    an admin's own logo for a workspace's server is stored beside that server, not here.
    """

    __tablename__ = "mcp_logos"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String)
    # The image as fetched, its type sniffed from its bytes. Null when none was found.
    content: Mapped[bytes | None] = mapped_column(LargeBinary, default=None)
    content_type: Mapped[str | None] = mapped_column(String, default=None)
    # A digest of ``content``; the served URL carries it, so a changed logo is a new URL.
    etag: Mapped[str | None] = mapped_column(String, default=None)
    # How bright the logo's ink is, 0 to 1, or null when it could not be told (or it carries its own background).
    luminance: Mapped[float | None] = mapped_column(Float, default=None)
    source_url: Mapped[str | None] = mapped_column(String, default=None)
    fetched_at: Mapped[datetime] = mapped_column(UtcDateTime())
    # Not looked for again before this, found or not.
    retry_at: Mapped[datetime] = mapped_column(UtcDateTime())
