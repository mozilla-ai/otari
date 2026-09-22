"""Add the models an organization offers on one of its provider keys.

An organization's BYO provider key reaches every model of its provider and
lists none of them: the catalog discovers only ``config.providers`` instances,
so a key bought an admin a credential and no way to see, price or withhold what
it served. This table is the membership half of that: one row per model offered
on one key, carrying the serving switch.

Deliberately no price column. An organization's rates live in
``organization_model_pricing``, which is the store
``services.pricing_service.find_model_pricing`` already consults first for a
request whose organization is known, so a rate set for an offered model is the
rate its requests settle at, with no second store to drift. ``model_pricing`` is
the wrong home for the same reason it always was here: it carries no tenancy
column, so one row would price the model for every organization on the
deployment.

Brand new, so a single forward migration with no expand/backfill/contract.

Revision ID: b4d7f1a9c2e6
Revises: f2a6c81d9b47
Create Date: 2026-09-21 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b4d7f1a9c2e6"
down_revision: str | Sequence[str] | None = "f2a6c81d9b47"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "org_provider_key_models",
        sa.Column("id", sa.Uuid(), nullable=False),
        # Denormalized from the key's own organization, so the composite FK
        # below can pin this row to that organization at the database rather
        # than trusting every write path to re-derive it. Same reasoning as the
        # two link tables in `e1c3a5b7d9f2`.
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("org_provider_key_id", sa.Uuid(), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        # True by default, but the service offers a model nothing prices with
        # this false: a model no rate could be found for must not be billed at
        # nothing while the pricing data catches up.
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["organization_id", "org_provider_key_id"],
            ["org_provider_keys.organization_id", "org_provider_keys.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("org_provider_key_id", "model", name="uq_org_provider_key_models_key_model"),
    )
    op.create_index(
        op.f("ix_org_provider_key_models_org_provider_key_id"),
        "org_provider_key_models",
        ["org_provider_key_id"],
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_org_provider_key_models_org_provider_key_id"),
        table_name="org_provider_key_models",
    )
    op.drop_table("org_provider_key_models")
