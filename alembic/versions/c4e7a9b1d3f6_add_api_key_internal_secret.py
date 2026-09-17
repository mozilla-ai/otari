"""Add ``api_keys.internal_secret``.

One nullable column, holding the encrypted plaintext of a key this deployment
minted for itself and has to present again later. The hosted Playground is the
first and so far only caller (``services/playground_dispatch``): a control plane
serves no inference, so it forwards the page's completion to its data-plane
gateway, and the gateway authenticates an API key. Verifying a credential needs
only the hash beside this column; presenting one needs the credential, which is
why it is stored, and it is stored the way provider credentials are, through
``secret_box``.

NULL on every existing row and on every key anybody creates. That is what the
key listings filter on, so the Playground's own row never appears on a page of
credentials somebody is expected to manage, and no endpoint returns the value.

Plain ``ADD COLUMN`` and ``DROP COLUMN``, no constraint and no index: the lookup
that reads it is already covered by the ``user_id`` index, and a rebuild of
``api_keys`` on SQLite would recreate a table two usage tables hold foreign keys
into.

Revision ID: c4e7a9b1d3f6
Revises: b2d4f6a8c0e2
Create Date: 2026-09-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4e7a9b1d3f6"
down_revision: str | Sequence[str] | None = "b2d4f6a8c0e2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Unbounded: the stored value is Fernet ciphertext, whose length follows the
    # token it wraps and the format's own framing rather than anything this
    # schema should bet on.
    op.add_column("api_keys", sa.Column("internal_secret", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("api_keys", "internal_secret")
