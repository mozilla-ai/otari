"""ORM tables for the MCP server catalog: the synced servers, the featured shelves, and the sync's state."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import Boolean, CheckConstraint, ForeignKey, Integer, String, Text, UniqueConstraint, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from gateway.models.base import Base, UtcDateTime

SYNC_STATE_ROW_ID = 1

# Where a catalog entry came from. One value today; the column is what lets a second source join the table.
SOURCE_REGISTRY = "registry"

# How the last sync ended. Null on the row until one has.
OUTCOME_COMPLETE = "complete"
OUTCOME_PARTIAL = "partial"
OUTCOME_FAILED = "failed"


class McpCatalogServer(Base):
    """One remote MCP server the catalog knows, as the sync last read it.

    Deployment-wide and owned by the sync: a complete sync rewrites every row
    it sees and deletes the ones it no longer does. Nothing a person decided
    lives here; a featured entry is a copy in ``mcp_catalog_featured``.
    """

    __tablename__ = "mcp_catalog_servers"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # The source's own identifier, reverse-DNS in the MCP Registry (``com.example/mcp``).
    name: Mapped[str] = mapped_column(String, unique=True)
    title: Mapped[str] = mapped_column(String, index=True)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    url: Mapped[str] = mapped_column(String)
    # The source says a call needs an Authorization header. False is "not declared", not "open".
    requires_token: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    website_url: Mapped[str | None] = mapped_column(String, default=None)
    version: Mapped[str | None] = mapped_column(String, default=None)
    source: Mapped[str] = mapped_column(String, default=SOURCE_REGISTRY, server_default=SOURCE_REGISTRY)
    # The start of the sync that last listed this server, which is how a finished sync finds the ones it did not.
    last_seen_at: Mapped[datetime] = mapped_column(UtcDateTime(), index=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=lambda: datetime.now(UTC))


class McpCatalogFeatured(Base):
    """A server an organization's admins recommend to its workspaces.

    Holds its own copy of the entry, taken when it was featured, so what an
    admin vetted is what a workspace adds: a sync that changes or removes the
    catalog row leaves this one as it was.

    CASCADE: a shelf means nothing without its organization.
    """

    __tablename__ = "mcp_catalog_featured"
    __table_args__ = (UniqueConstraint("organization_id", "name", name="uq_mcp_catalog_featured_organization_name"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("organization.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String)
    title: Mapped[str] = mapped_column(String)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    url: Mapped[str] = mapped_column(String)
    requires_token: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    website_url: Mapped[str | None] = mapped_column(String, default=None)
    source: Mapped[str] = mapped_column(String, default=SOURCE_REGISTRY, server_default=SOURCE_REGISTRY)
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
