"""Add hosted provider regions and declared data locations.

``hosted_provider_regions`` holds the regions of a provider that serves from
more than one; a provider with none serves from its own connection, as before.
``hosted_providers.data_location`` and the regions' own column declare where
the upstream keeps request data, ``workspace_provider_policies`` gains the list
of locations a workspace accepts, and ``usage_logs`` records the location a
request was served from. All nullable, so existing
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
    with op.batch_alter_table("usage_logs") as batch_op:
        batch_op.add_column(sa.Column("data_location", sa.String(length=16), nullable=True))
    with op.batch_alter_table("batches") as batch_op:
        batch_op.add_column(sa.Column("data_location", sa.String(length=16), nullable=True))
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
    op.create_index("ix_hosted_provider_regions_hosted_provider_id", "hosted_provider_regions", ["hosted_provider_id"])


def downgrade() -> None:
    # The release before this one serves a provider from its own connection
    # alone, so a region's own endpoint, key and country limits would be lost
    # and its traffic sent where none of them say.
    # Where past requests and batches were served is audit history nothing
    # can rebuild once the columns are dropped.
    recorded = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT (SELECT COUNT(*) FROM usage_logs WHERE data_location IS NOT NULL)"
                " + (SELECT COUNT(*) FROM batches WHERE data_location IS NOT NULL)"
            )
        )
        .scalar()
    )
    if recorded:
        msg = (
            f"{recorded} usage or batch row(s) record where they were served from, which this downgrade would drop. "
            "Export usage_logs.data_location and batches.data_location first, then clear them to downgrade."
        )
        raise RuntimeError(msg)
    regions = op.get_bind().execute(sa.text("SELECT COUNT(*) FROM hosted_provider_regions")).scalar()
    if regions:
        msg = (
            f"{regions} hosted provider region(s) exist, which the previous release cannot serve. Switch off each "
            "provider that has regions, then remove its regions, and keep it off until its own connection carries "
            "equivalent country and location limits: removing the last region serves from that connection at once."
        )
        raise RuntimeError(msg)
    # The release before this one cannot enforce a workspace's data locations,
    # so dropping them would quietly serve those workspaces from anywhere.
    restricted = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT COUNT(*) FROM workspace_provider_policies "
                "WHERE allowed_data_locations IS NOT NULL AND CAST(allowed_data_locations AS TEXT) != 'null'"
            )
        )
        .scalar()
    )
    if restricted:
        msg = (
            f"{restricted} workspace(s) limit hosted providers to some data locations, which the previous release "
            "cannot enforce. Clear those limits (or switch the affected providers off) before downgrading."
        )
        raise RuntimeError(msg)
    op.drop_index("ix_hosted_provider_regions_hosted_provider_id", table_name="hosted_provider_regions")
    op.drop_table("hosted_provider_regions")
    with op.batch_alter_table("batches") as batch_op:
        batch_op.drop_column("data_location")
    with op.batch_alter_table("usage_logs") as batch_op:
        batch_op.drop_column("data_location")
    with op.batch_alter_table("workspace_provider_policies") as batch_op:
        batch_op.drop_column("allowed_data_locations")
    with op.batch_alter_table("hosted_providers") as batch_op:
        batch_op.drop_column("data_location")
