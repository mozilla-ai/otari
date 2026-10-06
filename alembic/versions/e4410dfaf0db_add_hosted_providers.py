"""Add the hosted provider tables.

A hosted provider is the deployment's own upstream credential for one any-llm
implementation: what serves a request that brings no BYO key and names no
configured instance. ``hosted_provider_models`` is the roster the deployment
offers on each, with a serving switch per model; the rate stays in
``model_pricing``.

The roster keys on the provider's name rather than on a foreign key, because a
hosted provider may also be declared in ``config.yml`` or the environment, with
no row of its own, and its roster still has to live somewhere.

Revision ID: e4410dfaf0db
Revises: d4f7a2b9e6c1
Create Date: 2026-10-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e4410dfaf0db"
down_revision: str | Sequence[str] | None = "d4f7a2b9e6c1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_PROVIDERS = "hosted_providers"
_MODELS = "hosted_provider_models"
_MODELS_PROVIDER_INDEX = "ix_hosted_provider_models_provider"


def upgrade() -> None:
    op.create_table(
        "hosted_providers",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("encrypted_api_key", sa.String(), nullable=False),
        sa.Column("api_key_last4", sa.String(length=4), nullable=True),
        sa.Column("api_base", sa.String(length=1024), nullable=True),
        sa.Column("client_args", sa.JSON(), nullable=True),
        sa.Column("enabled", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider", name="uq_hosted_providers_provider"),
    )
    op.create_table(
        "hosted_provider_models",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("seeded_price_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider", "model", name="uq_hosted_provider_models_provider_model"),
    )
    op.create_index(_MODELS_PROVIDER_INDEX, _MODELS, ["provider"])


def downgrade() -> None:
    op.drop_index(_MODELS_PROVIDER_INDEX, table_name=_MODELS)
    op.drop_table(_MODELS)
    op.drop_table(_PROVIDERS)
