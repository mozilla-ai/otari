"""Record on an app authorization grant the scopes the person approved.

The exchange read the scopes from the app's live registration, so widening that
registration between the approval and the redemption minted a key beyond what
the registration held when the person approved it. The grant now carries them.

In-flight grants are deleted rather than backfilled. A grant lives two minutes
and holds no credential, so the worst a caller sees is one refused redemption
and a second trip through consent, which is better than minting a key against
scopes nobody approved. The column takes no default, for the same reason.

Revision ID: d4a7b2c9e510
Revises: c3f8a1d6e2b9
Create Date: 2026-11-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d4a7b2c9e510"
down_revision: str | Sequence[str] | None = "c3f8a1d6e2b9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_TABLE = "oauth_app_grants"


def upgrade() -> None:
    """Clear the grants approved before scopes existed, then add the column they owe."""
    op.execute(sa.text(f"DELETE FROM {_TABLE}"))
    # The column takes no default, so an insert that omits the scopes fails
    # rather than storing a grant that redeems to a key reaching nothing.
    # NOTE: SQLite accepts a NOT NULL column with no default only on an empty
    # table, which the delete above guarantees.
    with op.batch_alter_table(_TABLE) as batch_op:
        batch_op.add_column(sa.Column("scopes", sa.JSON(), nullable=False))


def downgrade() -> None:
    """Drop the column; the exchange reads the app's registration again."""
    op.execute(sa.text(f"DELETE FROM {_TABLE}"))
    with op.batch_alter_table(_TABLE) as batch_op:
        batch_op.drop_column("scopes")
