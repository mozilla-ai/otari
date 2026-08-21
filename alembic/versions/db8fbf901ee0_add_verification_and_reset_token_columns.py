"""Add the email-verification and password-reset token columns to ``user``.

Four nullable columns: ``email_verification_token_hash``,
``email_verification_token_expires_at``, ``password_reset_token_hash``,
``password_reset_token_expires_at``. otari#650 is the flow that reads and
writes them.

Neither reuses the existing ``email_verification_token`` column (added by
``f2a4c6d8b0e3``): that one is carried verbatim for hosted-edition parity and
stores a raw token per its own comment, and changing what it holds would be an
undocumented divergence from whatever the hosted platform's production column
still expects of it, for a column nothing in this codebase reads today. These
four are additive instead, following the shape ``Invitation.token_hash``
already established: only a SHA-256 hash is stored, and single-use is
enforced by clearing the hash and expiry to ``NULL`` on success rather than by
a status column, so a replayed token simply matches no row.

**No table rebuild**, for the same reason ``f2a4c6d8b0e3`` needed none: four
``ADD COLUMN`` statements and two ``CREATE UNIQUE INDEX`` statements are plain
DDL on both engines, and rebuilding ``user`` on SQLite would mean recreating
the table four other tables hold foreign keys into. Both hash columns are
unique through an index rather than a constraint, and ``downgrade`` drops
both indexes before their columns because SQLite refuses ``DROP COLUMN``
while one covers it.

Revision ID: db8fbf901ee0
Revises: c8e2a4f6b0d3
Create Date: 2026-08-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "db8fbf901ee0"
down_revision: str | Sequence[str] | None = "c8e2a4f6b0d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_VERIFICATION_TOKEN_HASH_INDEX = "ix_user_email_verification_token_hash"
_RESET_TOKEN_HASH_INDEX = "ix_user_password_reset_token_hash"


def upgrade() -> None:
    # 64 hex characters: the fixed width of a SHA-256 digest, matching
    # ``Invitation.token_hash``.
    op.add_column("user", sa.Column("email_verification_token_hash", sa.String(length=64), nullable=True))
    op.add_column("user", sa.Column("email_verification_token_expires_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("user", sa.Column("password_reset_token_hash", sa.String(length=64), nullable=True))
    op.add_column("user", sa.Column("password_reset_token_expires_at", sa.DateTime(timezone=True), nullable=True))
    # Unique so two identities cannot hold one token, and repeated NULLs stay
    # legal on both engines, which is what every existing row will have.
    op.create_index(
        op.f(_VERIFICATION_TOKEN_HASH_INDEX),
        "user",
        ["email_verification_token_hash"],
        unique=True,
    )
    op.create_index(
        op.f(_RESET_TOKEN_HASH_INDEX),
        "user",
        ["password_reset_token_hash"],
        unique=True,
    )


def downgrade() -> None:
    # Before their columns: SQLite refuses ``DROP COLUMN`` while an index covers it.
    op.drop_index(op.f(_RESET_TOKEN_HASH_INDEX), table_name="user")
    op.drop_index(op.f(_VERIFICATION_TOKEN_HASH_INDEX), table_name="user")
    op.drop_column("user", "password_reset_token_expires_at")
    op.drop_column("user", "password_reset_token_hash")
    op.drop_column("user", "email_verification_token_expires_at")
    op.drop_column("user", "email_verification_token_hash")
