"""Add hosted provider regions and declared data locations.

``hosted_provider_regions`` holds the regions of a provider that serves from
more than one; a provider with none serves from its own connection, as before.
``hosted_providers.data_location`` and the regions' own column declare where
the upstream keeps request data, and ``workspace_provider_policies``
gains the list of locations a workspace accepts. All nullable, so existing
rows read as undeclared and unrestricted.

Revision ID: d5a7c9e1b3f4
Revises: c3e8a1f5d7b2
Create Date: 2026-11-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d5a7c9e1b3f4"
down_revision: str | Sequence[str] | None = "c3e8a1f5d7b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("hosted_providers") as batch_op:
        batch_op.add_column(sa.Column("data_location", sa.String(length=16), nullable=True))
    with op.batch_alter_table("workspace_provider_policies") as batch_op:
        batch_op.add_column(sa.Column("allowed_data_locations", sa.JSON(), nullable=True))
    op.create_table(
        "hosted_provider_regions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("hosted_provider_id", sa.Uuid(), nullable=False),
        sa.Column("slug", sa.String(length=32), nullable=False),
        sa.Column("label", sa.String(length=64), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("data_location", sa.String(length=16), nullable=True),
        sa.Column("encrypted_api_key", sa.String(), nullable=True),
        sa.Column("api_key_last4", sa.String(length=4), nullable=True),
        sa.Column("api_base", sa.String(length=1024), nullable=True),
        sa.Column("encrypted_client_args", sa.String(), nullable=True),
        sa.Column("supported_countries", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["hosted_provider_id"], ["hosted_providers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("hosted_provider_id", "slug", name="uq_hosted_provider_regions_provider_slug"),
    )
    op.create_index(
        "ix_hosted_provider_regions_hosted_provider_id", "hosted_provider_regions", ["hosted_provider_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_hosted_provider_regions_hosted_provider_id", table_name="hosted_provider_regions")
    op.drop_table("hosted_provider_regions")
    with op.batch_alter_table("workspace_provider_policies") as batch_op:
        batch_op.drop_column("allowed_data_locations")
    with op.batch_alter_table("hosted_providers") as batch_op:
        batch_op.drop_column("data_location")
