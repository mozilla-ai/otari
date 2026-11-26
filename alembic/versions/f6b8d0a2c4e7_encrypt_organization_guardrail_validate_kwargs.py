"""Hold ``organization_guardrails.validate_kwargs`` encrypted.

The column carries the per-check arguments a mandate sends with every
guardrail call. A guardrail class can take its vendor key as one of them, which
is why the read already masked the column by entry name, and it was the second
place a credential lives on the row without a column of its own.

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

Revision ID: f6b8d0a2c4e7
Revises: e5a7c9b1d3f6
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

revision: str = "f6b8d0a2c4e7"
down_revision: str | Sequence[str] | None = "e5a7c9b1d3f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

guardrails = sa.table(
    "organization_guardrails",
    sa.column("id", sa.Uuid()),
    sa.column("validate_kwargs", sa.JSON()),
    sa.column("encrypted_validate_kwargs", sa.Text()),
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
            "organization_guardrails rows carry validate_kwargs that have to be encrypted, but no "
            "OTARI_SECRET_KEY is configured. Set it to the key the stored guardrail credentials were sealed "
            "with, then run this upgrade again."
        )
        raise RuntimeError(msg)


def _mapping(value: Any) -> dict[str, Any] | None:
    """The arguments a stored value holds, or None when it holds none worth keeping."""
    return value if isinstance(value, dict) and value else None


def upgrade() -> None:
    with op.batch_alter_table("organization_guardrails") as batch_op:
        batch_op.add_column(sa.Column("encrypted_validate_kwargs", sa.Text(), nullable=True))

    bind = op.get_bind()
    stored = [
        (row.id, values)
        for row in bind.execute(
            sa.select(guardrails.c.id, guardrails.c.validate_kwargs).where(guardrails.c.validate_kwargs.is_not(None))
        )
        if (values := _mapping(row.validate_kwargs)) is not None
    ]
    if stored:
        _require_the_deployments_key(bind)
        for guardrail_id, values in stored:
            bind.execute(
                sa.update(guardrails)
                .where(guardrails.c.id == guardrail_id)
                .values(encrypted_validate_kwargs=encrypt_mapping(values))
            )

    with op.batch_alter_table("organization_guardrails") as batch_op:
        batch_op.drop_column("validate_kwargs")


def downgrade() -> None:
    """Put the arguments back in clear, which is the only shape the previous release reads."""
    with op.batch_alter_table("organization_guardrails") as batch_op:
        batch_op.add_column(sa.Column("validate_kwargs", sa.JSON(), nullable=True))

    bind = op.get_bind()
    sealed = list(
        bind.execute(
            sa.select(guardrails.c.id, guardrails.c.encrypted_validate_kwargs).where(
                guardrails.c.encrypted_validate_kwargs.is_not(None)
            )
        )
    )
    if sealed:
        _require_the_deployments_key(bind)
        for row in sealed:
            bind.execute(
                sa.update(guardrails)
                .where(guardrails.c.id == row.id)
                .values(validate_kwargs=decrypt_mapping(row.encrypted_validate_kwargs))
            )

    with op.batch_alter_table("organization_guardrails") as batch_op:
        batch_op.drop_column("encrypted_validate_kwargs")
