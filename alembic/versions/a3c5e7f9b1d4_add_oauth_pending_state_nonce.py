"""Keep the OpenID nonce on a pending OAuth authorization.

apron-auth 0.17 sends a nonce with every OpenID authorization request and,
at the code exchange, refuses an ID token that does not carry it, and refuses
outright a pending state that has lost it. Google sign-in is OpenID, so the
row that carries the state between authorize and callback must carry the
nonce too, or every Google callback is refused.

Nullable, because GitHub sign-in is not OpenID and sends none. Rows already
in flight at the upgrade have none either; they expire within the state TTL,
and a Google sign-in started before the upgrade is simply restarted.

Revision ID: a3c5e7f9b1d4
Revises: e4410dfaf0db
Create Date: 2026-10-17 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a3c5e7f9b1d4"
down_revision: str | Sequence[str] | None = "e4410dfaf0db"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("oauth_pending_state", sa.Column("nonce", sa.String(length=128), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("oauth_pending_state") as batch:
        batch.drop_column("nonce")
