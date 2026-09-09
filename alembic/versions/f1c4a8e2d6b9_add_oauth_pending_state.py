"""Add the pending-state table that gives OAuth sign-in PKCE and a server-side state check.

The dashboard's Google and GitHub sign-in shipped in otari#765 with PKCE off and
the CSRF ``state`` checked only in the browser, against a value in
``sessionStorage`` that no server ever saw. The stated reason was that authorize
and callback are independent requests with nothing kept between them, so a
verifier minted at authorize time had nowhere to live. This table is that
somewhere, and it is the only thing the flow was missing: apron-auth has carried
a ``StateStore`` protocol and full PKCE support since the version otari#765
pinned.

One row per authorization in flight, deleted as it is consumed, so a replayed
state matches nothing, and bound to the browser that started it through the
digest of a cookie-held flow secret, so a redirect URL read out of an access log
or a history entry cannot finish the sign-in from anywhere else. Keyed by the SHA-256 of the state rather than the state
itself, which is where it departs from ``webauthn_challenge``: that table stores
its nonce in the clear because nothing is stored *under* it, while a row here
holds the PKCE ``code_verifier``, and a reader of this table must not come away
with a state they could present to the callback.

Nothing cascades from ``user`` here, unlike the passkey tables. A pending state
is minted before anybody is identified: whose sign-in it is is precisely what
the flow has not established yet.

Purely additive, so there is no data step and the downgrade is a drop.

Revision ID: f1c4a8e2d6b9
Revises: d5b7f9a1c3e6
Create Date: 2026-09-09 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f1c4a8e2d6b9"
down_revision: str | Sequence[str] | None = "d5b7f9a1c3e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "oauth_pending_state",
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        # SHA-256 as hex, so 64 characters exactly.
        sa.Column("state_hash", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        # SHA-256 as hex of the browser's flow-cookie secret.
        sa.Column("flow_hash", sa.String(length=64), nullable=False),
        # RFC 7636 caps a verifier at 128 characters. Nullable because
        # apron-auth's pending state models it that way for providers that
        # cannot do PKCE, not because either provider offered here needs it.
        sa.Column("code_verifier", sa.String(length=128), nullable=True),
        sa.Column("redirect_uri", sa.String(length=2048), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("state_hash"),
    )
    # The sweep's column: expired rows are pruned by range, and every consume
    # reads one row by primary key, so this is the only scan the table takes.
    op.create_index(
        op.f("ix_oauth_pending_state_expires_at"),
        "oauth_pending_state",
        ["expires_at"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_oauth_pending_state_expires_at"), table_name="oauth_pending_state")
    op.drop_table("oauth_pending_state")
