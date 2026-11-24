"""Hold ``search_tool_credentials.options`` encrypted.

The column carries free-form provider options for a search or fetch instance,
forwarded to the backend as keyword arguments. A backend that wants a second
credential takes it there, so the column is as much a credential store as the
API key beside it, and it was held in clear.

The clear column is dropped in the same upgrade rather than emptied, so the
migrated schema holds no cleartext. NOTE: a task still running the previous
release reads a column that is gone, so this upgrade needs the old tasks
drained rather than a rolling replacement. NOTE: dropping a column on
PostgreSQL is a catalog change, and the sealing update leaves the row version
it replaced behind, so the cleartext stays readable in the table's pages until
the deployment rewrites the table with ``VACUUM FULL`` or ``pg_repack``.

``secret_box`` is imported rather than reimplemented here, because a SQLite
deployment that sets no ``OTARI_SECRET_KEY`` keeps its key in a file beside the
database file and that module is what knows where to look.

Revision ID: d4f6a8c0e2b5
Revises: c3e5a7b9d1f4
Create Date: 2026-11-21
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine import Connection

from gateway.services.secret_box import (
    configured_secret_keys,
    decrypt_mapping,
    encrypt_mapping,
    secret_key_file_for,
    use_local_secret_key,
)

revision: str = "d4f6a8c0e2b5"
down_revision: str | Sequence[str] | None = "c3e5a7b9d1f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

tools = sa.table(
    "search_tool_credentials",
    sa.column("name", sa.String()),
    sa.column("options", sa.JSON()),
    sa.column("encrypted_options", sa.String()),
)


def _require_the_deployments_key(bind: Connection) -> None:
    """Load the key this deployment serves with, so rows seal with the key that opens them."""
    if configured_secret_keys():
        return
    url = bind.engine.url
    if url.get_backend_name() == "sqlite" and url.database and url.database != ":memory:":
        use_local_secret_key(secret_key_file_for(Path(url.database)), may_generate=False)
    if not configured_secret_keys():
        msg = (
            "search_tool_credentials rows carry options that have to be encrypted, but no OTARI_SECRET_KEY is "
            "configured. Set it to the key the stored tool keys were sealed with, then run this upgrade again."
        )
        raise RuntimeError(msg)


def _mapping(value: Any) -> dict[str, Any] | None:
    """The options a stored value holds, or None when it holds none worth keeping."""
    return value if isinstance(value, dict) and value else None


def upgrade() -> None:
    with op.batch_alter_table("search_tool_credentials") as batch_op:
        batch_op.add_column(sa.Column("encrypted_options", sa.String(), nullable=True))

    bind = op.get_bind()
    stored = [
        (row.name, values)
        for row in bind.execute(sa.select(tools.c.name, tools.c.options).where(tools.c.options.is_not(None)))
        if (values := _mapping(row.options)) is not None
    ]
    if stored:
        _require_the_deployments_key(bind)
        for name, values in stored:
            bind.execute(sa.update(tools).where(tools.c.name == name).values(encrypted_options=encrypt_mapping(values)))

    with op.batch_alter_table("search_tool_credentials") as batch_op:
        batch_op.drop_column("options")


def downgrade() -> None:
    """Put the options back in clear, which is the only shape the previous release reads.

    The column comes back NOT NULL as it was, which needs a server default the
    original did not carry: a row holding no options has to say so somehow.
    """
    with op.batch_alter_table("search_tool_credentials") as batch_op:
        batch_op.add_column(sa.Column("options", sa.JSON(), nullable=False, server_default=sa.text("'{}'")))

    bind = op.get_bind()
    sealed = list(
        bind.execute(sa.select(tools.c.name, tools.c.encrypted_options).where(tools.c.encrypted_options.is_not(None)))
    )
    if sealed:
        _require_the_deployments_key(bind)
        for row in sealed:
            bind.execute(
                sa.update(tools).where(tools.c.name == row.name).values(options=decrypt_mapping(row.encrypted_options))
            )

    with op.batch_alter_table("search_tool_credentials") as batch_op:
        batch_op.drop_column("encrypted_options")
