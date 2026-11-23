"""Add the edge-proxy credential an organization's provider key carries.

An organization that points a key at its own ``api_base`` often sits behind a
proxy that requires a second credential. All five columns are nullable, so an
existing key keeps dialing with its provider credential alone.

The credential means nothing unless every one of those columns holds a value,
and the CHECK is what keeps that true: without it a partial write would store a
declaration with no value behind it, and the row would read back as a
credential no build can assemble. SQLite has no ``ALTER TABLE ... ADD
CONSTRAINT``, so Alembic's batch mode rebuilds the table. ``copy_from`` states
the table explicitly, so the rebuild carries the two partial unique indexes and
the three constraints from this declaration rather than from whatever
reflection recovers.

Revision ID: c7e1b9d4a260
Revises: f6b8d0a2c4e7
Create Date: 2026-11-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c7e1b9d4a260"
down_revision: str | Sequence[str] | None = "f6b8d0a2c4e7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "org_provider_keys"
_CHECK = "ck_org_provider_keys_endpoint_auth_all_or_none"
_CHECK_SQL = (
    "(endpoint_auth_method IS NULL"
    " AND endpoint_auth_header_name IS NULL"
    " AND encrypted_endpoint_auth_secret IS NULL"
    " AND endpoint_auth_last4 IS NULL"
    " AND endpoint_auth_expires_at IS NULL)"
    " OR (endpoint_auth_method IS NOT NULL"
    " AND endpoint_auth_header_name IS NOT NULL"
    " AND encrypted_endpoint_auth_secret IS NOT NULL"
    " AND endpoint_auth_last4 IS NOT NULL"
    " AND endpoint_auth_expires_at IS NOT NULL)"
)


def _org_provider_keys(*, with_endpoint_auth: bool) -> sa.Table:
    """The table as it stands on either side of this revision.

    Passed as ``copy_from`` so SQLite's rebuild takes every column, index and
    constraint from here rather than from reflection.
    """
    columns = [
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("api_base", sa.String(length=1024), nullable=True),
        sa.Column("encrypted_api_key", sa.String(), nullable=True),
        sa.Column("last4", sa.String(length=8), nullable=True),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_org_default", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("zero_data_retention", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("slug", sa.String(length=64), nullable=True),
        sa.Column("encrypted_client_args", sa.String(), nullable=True),
    ]
    if with_endpoint_auth:
        columns += [
            sa.Column("endpoint_auth_method", sa.String(length=32), nullable=True),
            sa.Column("endpoint_auth_header_name", sa.String(length=64), nullable=True),
            sa.Column("encrypted_endpoint_auth_secret", sa.String(), nullable=True),
            sa.Column("endpoint_auth_last4", sa.String(length=8), nullable=True),
            sa.Column("endpoint_auth_expires_at", sa.DateTime(timezone=True), nullable=True),
        ]

    table = sa.Table(
        _TABLE,
        sa.MetaData(),
        *columns,
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "provider", "name", name="uq_org_provider_keys_org_provider_name"),
        sa.UniqueConstraint("organization_id", "id", name="uq_org_provider_keys_org_id"),
        sa.ForeignKeyConstraint(["organization_id"], ["organization.id"], ondelete="CASCADE"),
        *([sa.CheckConstraint(_CHECK_SQL, name=_CHECK)] if with_endpoint_auth else []),
    )
    sa.Index(
        "uq_org_provider_keys_org_default",
        table.c.organization_id,
        table.c.provider,
        unique=True,
        postgresql_where=sa.text("is_org_default AND archived_at IS NULL AND slug IS NULL"),
        sqlite_where=sa.text("is_org_default AND archived_at IS NULL AND slug IS NULL"),
    )
    sa.Index(
        "uq_org_provider_keys_org_slug",
        table.c.organization_id,
        table.c.slug,
        unique=True,
        postgresql_where=sa.text("slug IS NOT NULL"),
        sqlite_where=sa.text("slug IS NOT NULL"),
    )
    sa.Index("ix_org_provider_keys_organization_id", table.c.organization_id)
    return table


def upgrade() -> None:
    with op.batch_alter_table(_TABLE, copy_from=_org_provider_keys(with_endpoint_auth=False)) as batch_op:
        batch_op.add_column(sa.Column("endpoint_auth_method", sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column("endpoint_auth_header_name", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("encrypted_endpoint_auth_secret", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("endpoint_auth_last4", sa.String(length=8), nullable=True))
        batch_op.add_column(sa.Column("endpoint_auth_expires_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.create_check_constraint(_CHECK, _CHECK_SQL)


def downgrade() -> None:
    with op.batch_alter_table(_TABLE, copy_from=_org_provider_keys(with_endpoint_auth=True)) as batch_op:
        batch_op.drop_constraint(_CHECK, type_="check")
        batch_op.drop_column("endpoint_auth_expires_at")
        batch_op.drop_column("endpoint_auth_last4")
        batch_op.drop_column("encrypted_endpoint_auth_secret")
        batch_op.drop_column("endpoint_auth_header_name")
        batch_op.drop_column("endpoint_auth_method")
