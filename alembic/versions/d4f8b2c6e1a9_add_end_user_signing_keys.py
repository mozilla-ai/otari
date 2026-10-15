"""Let a service key sign end-user tokens offline.

Adds ``api_key_signing_keys``, the Ed25519 public keys a service key registers,
and ``api_key_token_redemptions``, one row per token used, keyed by the service
key and the token's ``jti`` so concurrent first uses converge on one row.

Revision ID: d4f8b2c6e1a9
Revises: c9e1a3f5b7d2
Create Date: 2026-10-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d4f8b2c6e1a9"
down_revision: str | Sequence[str] | None = "c9e1a3f5b7d2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SIGNING_KEYS = "api_key_signing_keys"
_REDEMPTIONS = "api_key_token_redemptions"


def upgrade() -> None:
    """Create the signing keys and the redemptions."""
    op.create_table(
        _SIGNING_KEYS,
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("api_key_id", sa.String(), nullable=False),
        sa.Column("public_key", sa.Text(), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["api_key_id"], ["api_keys.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_api_key_signing_keys_api_key_id", _SIGNING_KEYS, ["api_key_id"], unique=False)
    op.create_table(
        _REDEMPTIONS,
        sa.Column("parent_key_id", sa.String(), nullable=False),
        sa.Column("jti", sa.String(length=128), nullable=False),
        sa.Column("child_key_id", sa.String(), nullable=True),
        sa.Column("signing_key_id", sa.String(), nullable=False),
        sa.Column("claims_digest", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["parent_key_id"], ["api_keys.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["child_key_id"], ["api_keys.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("parent_key_id", "jti"),
    )
    op.create_index("ix_api_key_token_redemptions_child_key_id", _REDEMPTIONS, ["child_key_id"], unique=False)
    op.create_index("ix_api_key_token_redemptions_signing_key_id", _REDEMPTIONS, ["signing_key_id"], unique=False)
    op.create_index("ix_api_key_token_redemptions_expires_at", _REDEMPTIONS, ["expires_at"], unique=False)


def downgrade() -> None:
    """Drop the keys tokens were materialized as, then both tables.

    Such a key has a secret nobody was given, so without its redemption it is
    dead weight in every listing.
    """
    op.execute(f"DELETE FROM api_keys WHERE id IN (SELECT child_key_id FROM {_REDEMPTIONS})")  # noqa: S608
    op.drop_index("ix_api_key_token_redemptions_expires_at", table_name=_REDEMPTIONS)
    op.drop_index("ix_api_key_token_redemptions_signing_key_id", table_name=_REDEMPTIONS)
    op.drop_index("ix_api_key_token_redemptions_child_key_id", table_name=_REDEMPTIONS)
    op.drop_table(_REDEMPTIONS)
    op.drop_index("ix_api_key_signing_keys_api_key_id", table_name=_SIGNING_KEYS)
    op.drop_table(_SIGNING_KEYS)
