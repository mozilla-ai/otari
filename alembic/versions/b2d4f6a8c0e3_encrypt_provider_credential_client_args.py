"""Hold ``provider_credentials.client_args`` encrypted.

The retired Deployment providers page stored SDK client extras here in clear,
and a Bedrock instance keeps its ``aws_secret_access_key`` among them. A row
survives only where the startup move onto a hosted provider could not take it,
so these are the rows an operator has not dealt with yet and the ones most
likely to be forgotten in a backup.

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

Revision ID: b2d4f6a8c0e3
Revises: a1c3e5b7d9f2
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

revision: str = "b2d4f6a8c0e3"
down_revision: str | Sequence[str] | None = "a1c3e5b7d9f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

credentials = sa.table(
    "provider_credentials",
    sa.column("instance", sa.String()),
    sa.column("client_args", sa.JSON()),
    sa.column("encrypted_client_args", sa.String()),
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
            "provider_credentials rows carry client_args that have to be encrypted, but no OTARI_SECRET_KEY is "
            "configured. Set it to the key the stored provider credentials were sealed with, then run this "
            "upgrade again."
        )
        raise RuntimeError(msg)


def _mapping(value: Any) -> dict[str, Any] | None:
    """The settings a stored value holds, or None when it holds none worth keeping."""
    return value if isinstance(value, dict) and value else None


def upgrade() -> None:
    with op.batch_alter_table("provider_credentials") as batch_op:
        batch_op.add_column(sa.Column("encrypted_client_args", sa.String(), nullable=True))

    bind = op.get_bind()
    stored = [
        (row.instance, values)
        for row in bind.execute(
            sa.select(credentials.c.instance, credentials.c.client_args).where(credentials.c.client_args.is_not(None))
        )
        if (values := _mapping(row.client_args)) is not None
    ]
    if stored:
        _require_the_deployments_key(bind)
        for instance, values in stored:
            bind.execute(
                sa.update(credentials)
                .where(credentials.c.instance == instance)
                .values(encrypted_client_args=encrypt_mapping(values))
            )

    with op.batch_alter_table("provider_credentials") as batch_op:
        batch_op.drop_column("client_args")


def downgrade() -> None:
    """Put the settings back in clear, which is the only shape the previous release reads.

    The column comes back NOT NULL as it was, which needs a server default the
    original did not carry: a row holding no settings has to say so somehow.
    """
    with op.batch_alter_table("provider_credentials") as batch_op:
        batch_op.add_column(sa.Column("client_args", sa.JSON(), nullable=False, server_default=sa.text("'{}'")))

    bind = op.get_bind()
    sealed = list(
        bind.execute(
            sa.select(credentials.c.instance, credentials.c.encrypted_client_args).where(
                credentials.c.encrypted_client_args.is_not(None)
            )
        )
    )
    if sealed:
        _require_the_deployments_key(bind)
        for row in sealed:
            bind.execute(
                sa.update(credentials)
                .where(credentials.c.instance == row.instance)
                .values(client_args=decrypt_mapping(row.encrypted_client_args))
            )

    with op.batch_alter_table("provider_credentials") as batch_op:
        batch_op.drop_column("encrypted_client_args")
