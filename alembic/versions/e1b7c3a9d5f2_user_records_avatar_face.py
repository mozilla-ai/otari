"""Keep the avatar an identity picked on its profile.

One nullable integer on ``user``: the index of the preset face the dashboard
draws for this identity. Until now the pick lived in one browser's storage, so
nobody else in the organization saw it and it did not follow its owner to
another device.

No backfill. NULL means no pick, and the dashboard then draws the face the
identity's id hashes to, which is what every row showed before. Picks already
made in a browser are uploaded by that browser the next time it loads.

**No table rebuild**, for the reason ``a4d7f1c9e2b6`` gives: one ``ADD COLUMN``
with no constraint and no index is plain DDL on both engines.

Revision ID: e1b7c3a9d5f2
Revises: c4f9e2a7d5b8
Create Date: 2026-11-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e1b7c3a9d5f2"
down_revision: str | Sequence[str] | None = "c4f9e2a7d5b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("user", sa.Column("avatar_face", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("user", "avatar_face")
