"""Let an identity point at no organization.

An identity whose every membership refuses it has nowhere to act, and the wire contract and the
sign-in response already answer ``null`` for that. The column could not hold it, so the row went on
naming an organization the identity may not act in and every reader of the column believed it.

The downgrade points anyone holding ``null`` at the deployment's default organization, because the
column it restores cannot hold the state they are in. It fails where no default organization
exists, rather than choosing one of theirs on their behalf.

Revision ID: 4eee3a7ee90f
Revises: 10727bf67704
Create Date: 2026-11-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "4eee3a7ee90f"
down_revision: str | Sequence[str] | None = "10727bf67704"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_DEFAULT_ORGANIZATION_SLUG = "default"


def upgrade() -> None:
    with op.batch_alter_table("user") as batch_op:
        batch_op.alter_column("active_organization_id", existing_type=sa.Uuid(), nullable=True)


def downgrade() -> None:
    op.execute(
        sa.text(
            'UPDATE "user" SET active_organization_id = '
            "(SELECT id FROM organization WHERE slug = :slug) "
            "WHERE active_organization_id IS NULL"
        ).bindparams(slug=_DEFAULT_ORGANIZATION_SLUG)
    )
    with op.batch_alter_table("user") as batch_op:
        batch_op.alter_column("active_organization_id", existing_type=sa.Uuid(), nullable=False)
