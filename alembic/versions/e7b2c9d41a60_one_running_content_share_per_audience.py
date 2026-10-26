"""Hold a session to one running content share per audience.

A read resolves the newest running share of an audience, so one share per
audience is what makes ending the share an owner can see end the reading.
The index is partial on ``revoked_at IS NULL``, so ended shares stay for the
record of who read through them.

The backfill ends every duplicate but the newest of each audience, which is
the one reads resolve and the only one an owner is shown as running.

Revision ID: e7b2c9d41a60
Revises: c4e1a7b90d35
Create Date: 2026-10-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e7b2c9d41a60"
down_revision: str | Sequence[str] | None = "c4e1a7b90d35"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "trace_content_shares"
_INDEX = "uq_trace_content_shares_running"
_RUNNING = sa.text("revoked_at IS NULL")

_shares = sa.table(
    _TABLE,
    sa.column("id"),
    sa.column("workspace_id"),
    sa.column("trace_id"),
    sa.column("audience"),
    sa.column("created_at"),
    sa.column("revoked_at"),
)


def _newest_per_audience() -> sa.Select[tuple[object]]:
    ranked = (
        sa.select(
            _shares.c.id,
            sa.func.row_number()
            .over(
                partition_by=[_shares.c.workspace_id, _shares.c.trace_id, _shares.c.audience],
                order_by=[_shares.c.created_at.desc(), _shares.c.id.desc()],
            )
            .label("rank"),
        )
        .where(_shares.c.revoked_at.is_(None))
        .subquery()
    )
    return sa.select(ranked.c.id).where(ranked.c.rank == 1)


def upgrade() -> None:
    op.execute(
        _shares.update()
        .where(_shares.c.revoked_at.is_(None), _shares.c.id.notin_(_newest_per_audience()))
        .values(revoked_at=sa.func.now())
    )
    op.create_index(
        _INDEX,
        _TABLE,
        ["workspace_id", "trace_id", "audience"],
        unique=True,
        postgresql_where=_RUNNING,
        sqlite_where=_RUNNING,
    )


def downgrade() -> None:
    op.drop_index(_INDEX, table_name=_TABLE)
