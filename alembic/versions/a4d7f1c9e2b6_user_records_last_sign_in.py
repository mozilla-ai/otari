"""Record when an identity last signed in to the dashboard.

One nullable timestamp on ``user``, stamped wherever a dashboard session is
minted (``services/dashboard_session_service.create_dashboard_session``) and read
by the deployment-wide user administration surface (``/v1/admin/users``), which
is the first page that has to answer "is this account still in use".

A stored column rather than ``max(dashboard_sessions.created_at)``, which would
have needed no migration at all. Session rows are pruned once they expire, so the
derived answer decays from "signed in three weeks ago" to "never signed in" with
nothing distinguishing the two, and "never signed in" is the reading an operator
would act on. NULL here means never, and goes on meaning it.

No backfill. Every existing row reads NULL, which is honest: the sessions that
would have dated them are gone or will be, and inventing a timestamp from a live
session row would date the deployment's upgrade rather than the sign-in.

**No table rebuild**, for the reason ``f2a4c6d8b0e3`` gives: one ``ADD COLUMN``
with no constraint and no index, which both engines take as plain DDL, so
``batch_alter_table`` (and, on SQLite, recreating a table four others hold
foreign keys into) is not needed.

Timezone-aware, matching the other timestamps this tree adds to ``user``.

Revision ID: a4d7f1c9e2b6
Revises: d2f5b8c0e4a7
Create Date: 2026-08-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a4d7f1c9e2b6"
down_revision: str | Sequence[str] | None = "d2f5b8c0e4a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("user", sa.Column("last_sign_in_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("user", "last_sign_in_at")
