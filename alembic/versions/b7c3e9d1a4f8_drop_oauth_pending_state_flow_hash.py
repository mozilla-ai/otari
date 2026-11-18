"""Move ``oauth_pending_state`` onto the library's own pending-state fields.

``flow_hash`` held the digest of a cookie-held secret, compared after the row was
claimed, so that a redirect URL read out of an access log or a history entry
could not finish a sign-in from another browser. apron-auth now derives the
``state`` from a token it mints per authorization request and refuses a callback
whose state that token does not mint, which answers the same threat before the
row is claimed and before the provider is called. One mechanism is kept and the
column goes.

``openid_requested`` arrives in the same change, because the library's state
store contract asks for every saved field back and this is the one the row did
not hold. It is NOT NULL and takes no default: a default would answer "the
request asked for no OpenID", which is the answer that leaves an ID token's
nonce unchecked.

Rows in flight across the upgrade cannot be redeemed, because their state was
not minted from a browser token, and they expire within their TTL and are swept
by the next authorization. They are deleted here rather than defaulted, for the
reason above. The downgrade restores the column and deletes the rows first for
the same reason in reverse: none of them has a digest to restore.

Revision ID: b7c3e9d1a4f8
Revises: d5a1c7e3f9b2
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b7c3e9d1a4f8"
down_revision: str | Sequence[str] | None = "d5a1c7e3f9b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "oauth_pending_state"


def upgrade() -> None:
    op.execute(sa.text(f"DELETE FROM {_TABLE}"))
    # Batch mode because SQLite cannot add a NOT NULL column without a default
    # in place; it rebuilds the emptied table instead.
    with op.batch_alter_table(_TABLE) as batch_op:
        batch_op.drop_column("flow_hash")
        batch_op.add_column(sa.Column("openid_requested", sa.Boolean(), nullable=False))


def downgrade() -> None:
    op.execute(sa.text(f"DELETE FROM {_TABLE}"))
    with op.batch_alter_table(_TABLE) as batch_op:
        batch_op.drop_column("openid_requested")
        batch_op.add_column(sa.Column("flow_hash", sa.String(length=64), nullable=False))
