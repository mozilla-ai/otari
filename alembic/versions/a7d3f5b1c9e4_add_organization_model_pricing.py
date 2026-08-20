"""Add per-organization model pricing overrides.

``model_pricing`` has no tenancy column, so a deployment has exactly one price
list. ``organization_model_pricing`` is the layer above it: an organization's own
rate for a model, resolved ahead of the deployment row and the genai-prices
dataset. A deployment that creates no override prices exactly as it did before,
which is why this is a pure ``CREATE TABLE`` with no backfill and no change to
``model_pricing``.

Two things about the shape, both explained at length on the model
(`gateway.models.entities.OrganizationModelPricing`) and noted here because a
reader of the migration alone would find them surprising:

- The key is one ``model_key`` string, not a ``provider`` plus ``model`` pair as
  the platform's equivalent table carries, because the whole pricing chain here
  keys on ``provider:model`` and that string is sometimes a provider *instance*
  or not a model at all (``otari:web_search``).
- The period is an interval (``effective_from``, nullable ``effective_to``) where
  ``model_pricing`` carries a version series, and overlapping periods for one key
  are refused rather than shadowed. The refusal lives in the service: an
  ``EXCLUDE`` over a range type is the natural constraint and SQLite, which the
  OSS edition ships by default, has neither. The unique index here is the part
  both engines can enforce, the exact-duplicate start.

Revision ID: a7d3f5b1c9e4
Revises: c8f2a6b4e9d3
Create Date: 2026-08-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a7d3f5b1c9e4"
down_revision: str | Sequence[str] | None = "c8f2a6b4e9d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "organization_model_pricing"


def upgrade() -> None:
    op.create_table(
        _TABLE,
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("model_key", sa.String(), nullable=False),
        sa.Column("input_price_per_million", sa.Float(), nullable=False),
        sa.Column("output_price_per_million", sa.Float(), nullable=False),
        sa.Column("cache_read_price_per_million", sa.Float(), nullable=True),
        sa.Column("cache_write_price_per_million", sa.Float(), nullable=True),
        sa.Column("cache_write_1h_price_per_million", sa.Float(), nullable=True),
        sa.Column("pricing_tiers", sa.JSON(), nullable=False),
        sa.Column("effective_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("effective_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organization.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "effective_to IS NULL OR effective_to > effective_from",
            name="ck_organization_model_pricing_period_ordered",
        ),
        sa.CheckConstraint(
            "input_price_per_million >= 0",
            name="ck_organization_model_pricing_input_non_negative",
        ),
        sa.CheckConstraint(
            "output_price_per_million >= 0",
            name="ck_organization_model_pricing_output_non_negative",
        ),
        sa.CheckConstraint(
            "cache_read_price_per_million IS NULL OR cache_read_price_per_million >= 0",
            name="ck_organization_model_pricing_cache_read_non_negative",
        ),
        sa.CheckConstraint(
            "cache_write_price_per_million IS NULL OR cache_write_price_per_million >= 0",
            name="ck_organization_model_pricing_cache_write_non_negative",
        ),
        sa.CheckConstraint(
            "cache_write_1h_price_per_million IS NULL OR cache_write_1h_price_per_million >= 0",
            name="ck_organization_model_pricing_cache_write_1h_non_negative",
        ),
    )
    # One index serves as both the uniqueness constraint and the resolution
    # lookup, because both want these three columns in this order.
    op.create_index(
        "uq_organization_model_pricing_period_start",
        _TABLE,
        ["organization_id", "model_key", "effective_from"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_organization_model_pricing_period_start", table_name=_TABLE)
    op.drop_table(_TABLE)
