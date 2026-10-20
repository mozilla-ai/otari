"""Give price rows a kind, so a tool's rate has a namespace of its own.

A search or fetch instance was priced as ``<provider>:<instance>``, a model as
``<provider>:<model>``, and the gateway told the two apart by the provider part
alone, which breaks when a search provider is also a model provider. ``kind``
(``model``, ``search`` or ``fetch``) joins each table's key, so a tool's row is
keyed by its kind and instance name and can never be read as a model's. Every
existing row is a ``model`` one: a tool rate written under ``<provider>:<instance>``
keeps its key and is still read, as a deprecated form, until a later release.

Revision ID: d8b2f6a4c0e9
Revises: c0e83c6fe801
Create Date: 2026-10-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d8b2f6a4c0e9"
down_revision: str | Sequence[str] | None = "c0e83c6fe801"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_KIND = "kind"
_PRIMARY_KEY = "model_pricing_pkey"
_PERIOD_INDEX = "uq_organization_model_pricing_period_start"
# The scale ``a7c3e5d9b1f4`` gave every rate column, spelled out so this
# revision keeps its meaning when the application's idea of the scale changes.
_RATE_TYPE = sa.Numeric(18, 8)


def _kind_column() -> sa.Column[str]:
    """The ``kind`` column, every existing row reading as a ``model`` one."""
    return sa.Column(_KIND, sa.String(length=16), nullable=False, server_default="model")


def _model_pricing_table(*, with_kind: bool) -> sa.Table:
    """``model_pricing`` as it stands, for SQLite's batch rebuild.

    Spelled out rather than reflected, as ``a7c3e5d9b1f4`` explains: batch mode
    has mis-reflected this table's composite primary key before.
    """
    key = ["kind", "model_key", "effective_at"] if with_kind else ["model_key", "effective_at"]
    return sa.Table(
        "model_pricing",
        sa.MetaData(),
        *([_kind_column()] if with_kind else []),
        sa.Column("model_key", sa.String(), nullable=False),
        sa.Column("effective_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("input_price_per_million", _RATE_TYPE, nullable=False),
        sa.Column("output_price_per_million", _RATE_TYPE, nullable=False),
        sa.Column("cache_read_price_per_million", _RATE_TYPE, nullable=True),
        sa.Column("cache_write_price_per_million", _RATE_TYPE, nullable=True),
        sa.Column("pricing_tiers", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("unit", sa.String(length=16), nullable=False, server_default="tokens"),
        sa.Column("origin", sa.String(length=16), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint(*key, name=_PRIMARY_KEY),
    )


def upgrade() -> None:
    """Upgrade schema."""
    is_sqlite = op.get_bind().dialect.name == "sqlite"
    op.add_column("model_pricing", _kind_column())
    with op.batch_alter_table("model_pricing", copy_from=_model_pricing_table(with_kind=True)) as batch:
        if not is_sqlite:
            batch.drop_constraint(_PRIMARY_KEY, type_="primary")
        batch.create_primary_key(_PRIMARY_KEY, ["kind", "model_key", "effective_at"])

    op.add_column("organization_model_pricing", _kind_column())
    op.drop_index(_PERIOD_INDEX, table_name="organization_model_pricing")
    op.create_index(
        _PERIOD_INDEX,
        "organization_model_pricing",
        ["organization_id", "kind", "model_key", "effective_from"],
        unique=True,
    )


def downgrade() -> None:
    """Downgrade schema.

    A tool row has nowhere to go under the old key, so a database that holds
    one refuses the downgrade rather than drop a rate somebody set.
    """
    bind = op.get_bind()
    for table in ("model_pricing", "organization_model_pricing"):
        if bind.execute(sa.text(f"SELECT 1 FROM {table} WHERE kind <> 'model'")).first() is not None:  # noqa: S608
            msg = f"Cannot downgrade: {table} holds search or fetch tool prices; delete them first."
            raise RuntimeError(msg)

    op.drop_index(_PERIOD_INDEX, table_name="organization_model_pricing")
    op.create_index(
        _PERIOD_INDEX,
        "organization_model_pricing",
        ["organization_id", "model_key", "effective_from"],
        unique=True,
    )
    with op.batch_alter_table("organization_model_pricing") as batch:
        batch.drop_column(_KIND)

    is_sqlite = bind.dialect.name == "sqlite"
    with op.batch_alter_table("model_pricing", copy_from=_model_pricing_table(with_kind=True)) as batch:
        if not is_sqlite:
            batch.drop_constraint(_PRIMARY_KEY, type_="primary")
        batch.create_primary_key(_PRIMARY_KEY, ["model_key", "effective_at"])
        batch.drop_column(_KIND)
