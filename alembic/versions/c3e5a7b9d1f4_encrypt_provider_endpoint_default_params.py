"""Hold ``provider_endpoints.default_params`` encrypted.

The column carries fields sent in every request body beneath the caller's own.
A model server behind a content delivery network or an identity-aware proxy
wants that edge's credential among them, so the column is as much a credential
store as the API key beside it, and it was held in clear.

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

Revision ID: c3e5a7b9d1f4
Revises: b2d4f6a8c0e3
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

revision: str = "c3e5a7b9d1f4"
down_revision: str | Sequence[str] | None = "b2d4f6a8c0e3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

endpoints = sa.table(
    "provider_endpoints",
    sa.column("id", sa.Uuid()),
    sa.column("default_params", sa.JSON()),
    sa.column("encrypted_default_params", sa.String()),
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
            "provider_endpoints rows carry default_params that have to be encrypted, but no OTARI_SECRET_KEY is "
            "configured. Set it to the key the stored endpoint keys were sealed with, then run this upgrade again."
        )
        raise RuntimeError(msg)


def _mapping(value: Any) -> dict[str, Any] | None:
    """The fields a stored value holds, or None when it holds none worth keeping."""
    return value if isinstance(value, dict) and value else None


def upgrade() -> None:
    with op.batch_alter_table("provider_endpoints") as batch_op:
        batch_op.add_column(sa.Column("encrypted_default_params", sa.String(), nullable=True))

    bind = op.get_bind()
    stored = [
        (row.id, values)
        for row in bind.execute(
            sa.select(endpoints.c.id, endpoints.c.default_params).where(endpoints.c.default_params.is_not(None))
        )
        if (values := _mapping(row.default_params)) is not None
    ]
    if stored:
        _require_the_deployments_key(bind)
        for endpoint_id, values in stored:
            bind.execute(
                sa.update(endpoints)
                .where(endpoints.c.id == endpoint_id)
                .values(encrypted_default_params=encrypt_mapping(values))
            )

    with op.batch_alter_table("provider_endpoints") as batch_op:
        batch_op.drop_column("default_params")


def downgrade() -> None:
    """Put the fields back in clear, which is the only shape the previous release reads."""
    with op.batch_alter_table("provider_endpoints") as batch_op:
        batch_op.add_column(sa.Column("default_params", sa.JSON(), nullable=True))

    bind = op.get_bind()
    sealed = list(
        bind.execute(
            sa.select(endpoints.c.id, endpoints.c.encrypted_default_params).where(
                endpoints.c.encrypted_default_params.is_not(None)
            )
        )
    )
    if sealed:
        _require_the_deployments_key(bind)
        for row in sealed:
            bind.execute(
                sa.update(endpoints)
                .where(endpoints.c.id == row.id)
                .values(default_params=decrypt_mapping(row.encrypted_default_params))
            )

    with op.batch_alter_table("provider_endpoints") as batch_op:
        batch_op.drop_column("encrypted_default_params")
