"""Add the hosted provider tables.

A hosted provider is the deployment's own upstream credential for one any-llm
implementation: what serves a request that brings no BYO key and names no
configured instance. ``hosted_provider_models`` is the roster the deployment
offers on each, with a serving switch per model; the rate stays in
``model_pricing``.

The roster keys on the provider's name rather than on a foreign key, because a
hosted provider may also be declared in configuration with no row of its own,
and its roster still has to live somewhere. A seeded rate is told from a chosen
one by the price version's ``origin``, so the roster carries no timestamp.

Revision ID: e4410dfaf0db
Revises: d4f8b2a6c1e9
Create Date: 2026-10-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e4410dfaf0db"
down_revision: str | Sequence[str] | None = "d4f8b2a6c1e9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_PROVIDERS = "hosted_providers"
_MODELS = "hosted_provider_models"


def upgrade() -> None:
    op.create_table(
        "hosted_providers",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("encrypted_api_key", sa.String(), nullable=False),
        sa.Column("api_key_last4", sa.String(length=4), nullable=True),
        sa.Column("api_base", sa.String(length=1024), nullable=True),
        sa.Column("encrypted_client_args", sa.String(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider", name="uq_hosted_providers_provider"),
    )
    op.create_table(
        "hosted_provider_models",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        # Leads with ``provider``, so a per-provider read needs no index of its own.
        sa.UniqueConstraint("provider", "model", name="uq_hosted_provider_models_provider_model"),
    )


def downgrade() -> None:
    op.drop_table(_MODELS)
    op.drop_table(_PROVIDERS)
