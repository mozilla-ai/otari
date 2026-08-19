"""Add the nullable credential columns to ``user``.

``hashed_password``, ``terms_accepted_at``, ``oauth_provider``,
``email_verification_token`` and ``email_verified_at``, all nullable, all
unread. They exist in the platform's production ``user`` table and in no table
here, and the overlay contributes no tables, so landing them now gives the
re-parenting migration (otari-ai#1644) one target schema instead of one per
edition. otari-ai#1716 settled the flow they belong to: the master key stays the
API credential and sessions become the dashboard login, so a standalone row with
all five null is the normal state rather than an unmigrated one. No
authentication behavior changes here.

``email`` is deliberately untouched: it stays nullable and uniquely indexed,
because a standalone operator identity is a label rather than a sign-in address.

Three things this revision settles, all of them stated in ``models/tenancy.py``
next to the columns:

- **No table rebuild.** Five ``ADD COLUMN`` statements and one
  ``CREATE UNIQUE INDEX``, which both engines take as plain DDL.
  ``batch_alter_table`` is what a constraint would have cost, and rebuilding
  ``user`` on SQLite would mean recreating the table four other tables hold
  foreign keys into, plus the half of the ``organization`` cycle that points at
  it. ``email_verification_token`` is therefore unique through an index rather
  than a constraint, which is what the platform does too, and the index is
  dropped before the column in ``downgrade`` because SQLite refuses
  ``DROP COLUMN`` on an indexed column.
- **``oauth_provider`` is a VARCHAR, not the platform's native ``oauthprovider``
  enum.** ``op.add_column`` does not create a PostgreSQL enum type for you, and
  the same type renders as VARCHAR plus a CHECK on SQLite, which the OSS edition
  ships by default. The vocabulary belongs to the OAuth flow that has not
  rehomed; the tenancy tables already store their own vocabularies as strings.
- **The two timestamps are timezone-aware**, unlike the platform's naive
  ``DateTime``, matching the departure the tenancy revision already applied to
  every ``created_at`` and ``updated_at`` in this schema.

Revision ID: f2a4c6d8b0e3
Revises: a3c7e1b9d5f2
Create Date: 2026-08-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f2a4c6d8b0e3"
down_revision: str | Sequence[str] | None = "a3c7e1b9d5f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_VERIFICATION_TOKEN_INDEX = "ix_user_email_verification_token"


def upgrade() -> None:
    # Unbounded, as the platform has it: a hash carries its own algorithm and
    # cost parameters, so a ceiling here would be a bet on which one the session
    # flow picks.
    op.add_column("user", sa.Column("hashed_password", sa.String(), nullable=True))
    op.add_column("user", sa.Column("terms_accepted_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("user", sa.Column("oauth_provider", sa.String(length=50), nullable=True))
    op.add_column("user", sa.Column("email_verification_token", sa.String(), nullable=True))
    op.add_column("user", sa.Column("email_verified_at", sa.DateTime(timezone=True), nullable=True))
    # Unique so two identities cannot hold one token, and repeated NULLs stay
    # legal on both engines, which is what every existing row will have.
    op.create_index(
        op.f(_VERIFICATION_TOKEN_INDEX),
        "user",
        ["email_verification_token"],
        unique=True,
    )


def downgrade() -> None:
    # Before the column: SQLite refuses ``DROP COLUMN`` while an index covers it.
    op.drop_index(op.f(_VERIFICATION_TOKEN_INDEX), table_name="user")
    op.drop_column("user", "email_verified_at")
    op.drop_column("user", "email_verification_token")
    op.drop_column("user", "oauth_provider")
    op.drop_column("user", "terms_accepted_at")
    op.drop_column("user", "hashed_password")
