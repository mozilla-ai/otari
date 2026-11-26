"""Hold ``org_provider_keys.client_args`` encrypted.

The column carries arbitrary settings an organization admin sets, and a
credential legitimately lives in it: Bedrock's ``aws_secret_access_key``, or an
authentication header an edge in front of the endpoint expects. It was stored
in clear, so anyone who could read the database recovered it, and every backup
taken since the row was written holds it. The settings now travel the way the
API key beside them already does, sealed with ``OTARI_SECRET_KEY``.

The clear column is dropped in the same upgrade rather than emptied, so the
migrated schema holds no cleartext. NOTE: a task still running the previous
release reads a column that is gone, so this upgrade needs the old tasks
drained rather than a rolling replacement. NOTE: dropping a column on
PostgreSQL is a catalog change, and the sealing update leaves the row version
it replaced behind, so the cleartext stays readable in the table's pages until
the deployment rewrites the table with ``VACUUM FULL`` or ``pg_repack``.

``secret_box`` is imported rather than reimplemented here. A SQLite deployment
that sets no ``OTARI_SECRET_KEY`` keeps its key in a file beside the database
file, and that module is what knows where to look; a migration that read only
the environment variable would refuse to run on such a deployment.

Revision ID: a1c3e5b7d9f2
Revises: b4d6f8a0c2e5
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

revision: str = "a1c3e5b7d9f2"
down_revision: str | Sequence[str] | None = "e2b7c4d9f1a3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

keys = sa.table(
    "org_provider_keys",
    sa.column("id", sa.Uuid()),
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
            "org_provider_keys rows carry client_args that have to be encrypted, but no OTARI_SECRET_KEY is "
            "configured. Set it to the key the stored provider credentials were sealed with, then run this "
            "upgrade again."
        )
        raise RuntimeError(msg)


def _mapping(value: Any) -> dict[str, Any] | None:
    """The settings a stored value holds, or None when it holds none worth keeping."""
    return value if isinstance(value, dict) and value else None


def upgrade() -> None:
    with op.batch_alter_table("org_provider_keys") as batch_op:
        batch_op.add_column(sa.Column("encrypted_client_args", sa.String(), nullable=True))

    bind = op.get_bind()
    stored = [
        (row.id, values)
        for row in bind.execute(sa.select(keys.c.id, keys.c.client_args).where(keys.c.client_args.is_not(None)))
        if (values := _mapping(row.client_args)) is not None
    ]
    if stored:
        _require_the_deployments_key(bind)
        for key_id, values in stored:
            bind.execute(
                sa.update(keys).where(keys.c.id == key_id).values(encrypted_client_args=encrypt_mapping(values))
            )

    with op.batch_alter_table("org_provider_keys") as batch_op:
        batch_op.drop_column("client_args")


def downgrade() -> None:
    """Put the settings back in clear, which is the only shape the previous release reads."""
    with op.batch_alter_table("org_provider_keys") as batch_op:
        batch_op.add_column(sa.Column("client_args", sa.JSON(), nullable=True))

    bind = op.get_bind()
    sealed = list(
        bind.execute(
            sa.select(keys.c.id, keys.c.encrypted_client_args).where(keys.c.encrypted_client_args.is_not(None))
        )
    )
    if sealed:
        _require_the_deployments_key(bind)
        for row in sealed:
            bind.execute(
                sa.update(keys)
                .where(keys.c.id == row.id)
                .values(client_args=decrypt_mapping(row.encrypted_client_args))
            )

    with op.batch_alter_table("org_provider_keys") as batch_op:
        batch_op.drop_column("encrypted_client_args")
