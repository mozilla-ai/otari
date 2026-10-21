"""Drop ``api_keys.internal_secret`` and its dispatch index.

The column held the encrypted plaintext of the key a hosted control plane
minted to forward its Playground to a data-plane gateway. Hosted mode is gone
and the Playground runs its completions in-process, so nothing presents those
keys any more. The rows that carry one are deleted first: without the column
they would read as ordinary keys, appearing on the key pages and still
authenticating. Their usage rows keep their user and workspace; only the link
to the deleted key is cleared, as deleting any key does. Those rows were
written from the data plane's usage report, labeled like any other request, and
only the key marked them as Playground traffic, so they are relabeled with the
Playground endpoint first: unmarked, they would close a workspace's activation
guide as if its own code had called the gateway.

The downgrade restores the column and the index but not the deleted rows,
whose plaintext is gone.

Revision ID: 5ab5a1699e8b
Revises: dbd76ac9d948
Create Date: 2026-10-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "5ab5a1699e8b"
down_revision: str | Sequence[str] | None = "dbd76ac9d948"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_DISPATCH_INDEX = "ix_api_keys_internal_dispatch"
_PLAYGROUND_USAGE_ENDPOINT = "/v1/playground/chat/completions"


def upgrade() -> None:
    op.execute(
        sa.text(
            "UPDATE usage_logs SET endpoint = :endpoint"
            " WHERE api_key_id IN (SELECT id FROM api_keys WHERE internal_secret IS NOT NULL)"
        ).bindparams(endpoint=_PLAYGROUND_USAGE_ENDPOINT)
    )
    op.execute(sa.text("DELETE FROM api_keys WHERE internal_secret IS NOT NULL"))
    op.drop_index(_DISPATCH_INDEX, table_name="api_keys")
    op.drop_column("api_keys", "internal_secret")


def downgrade() -> None:
    op.add_column("api_keys", sa.Column("internal_secret", sa.String(), nullable=True))
    op.create_index(
        _DISPATCH_INDEX,
        "api_keys",
        ["user_id", "workspace_id", "created_at", "id"],
        unique=False,
        postgresql_where=sa.text("internal_secret IS NOT NULL"),
        sqlite_where=sa.text("internal_secret IS NOT NULL"),
    )
