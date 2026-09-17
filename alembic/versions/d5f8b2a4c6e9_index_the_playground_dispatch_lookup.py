"""Index the hosted Playground's dispatch-key lookup.

``services/playground_dispatch.resolve_dispatch_key`` runs on every message the
hosted Playground sends. It reads ``api_keys`` by owner and workspace among the
rows this deployment minted for itself, and the single-column indexes on
``user_id`` and ``workspace_id`` leave that read walking the owner's ordinary
keys and sorting whatever it finds.

Partial on ``internal_secret IS NOT NULL``, which is what marks those rows. There
is at most one per owner per workspace, so the index stays the size of the thing
it answers for rather than the size of a table that holds every key on the
deployment. Both engines support a partial index, and the predicate matches the
query's exactly, so neither has to be talked into using it.

``created_at`` and ``id`` trail the two equality columns because they are the
resolver's tie-break, so the read is ordered by the index rather than sorted
after it. They earn their place only where the race that resolver tolerates has
left a second row, and they cost nothing the rest of the time: neither column is
ever updated, so the per-message write that refreshes the row's allow-list does
not touch this index.

Separate from ``c4e7a9b1d3f6``, which added the column: that revision is on
``main`` and has been applied, so a database that already ran it would never see
an index added to it in place.

Revision ID: d5f8b2a4c6e9
Revises: c4e7a9b1d3f6
Create Date: 2026-09-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d5f8b2a4c6e9"
down_revision: str | Sequence[str] | None = "c4e7a9b1d3f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_DISPATCH_INDEX = "ix_api_keys_internal_dispatch"


def upgrade() -> None:
    op.create_index(
        _DISPATCH_INDEX,
        "api_keys",
        ["user_id", "workspace_id", "created_at", "id"],
        unique=False,
        postgresql_where=sa.text("internal_secret IS NOT NULL"),
        sqlite_where=sa.text("internal_secret IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(_DISPATCH_INDEX, table_name="api_keys")
