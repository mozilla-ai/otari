"""Name hosted providers, and let a keyless one store no key.

A hosted provider was keyed on its any-llm implementation, so a deployment
could serve one endpoint per implementation. ``name`` is now what a request,
the roster and the price list know it by, backfilled with the implementation so
every existing provider keeps its name, and ``provider`` stays the
implementation without being unique. ``encrypted_api_key`` becomes nullable
for a backend called without a key.

The downgrade refuses while a row is named otherwise than its implementation,
or holds no key, because neither fits the old table: the old code finds a
roster and a price under the implementation's name only.

Revision ID: c7e1a3f5b9d2
Revises: 557fc0f5bfda
Create Date: 2026-10-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c7e1a3f5b9d2"
down_revision: str | Sequence[str] | None = "557fc0f5bfda"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "hosted_providers"
_RESTRICTION_INDEX = "ix_hosted_providers_restricted_to_organization_id"
_UNIQUE_PROVIDER = "uq_hosted_providers_provider"
_UNIQUE_NAME = "uq_hosted_providers_name"


def _hosted_providers(*, named: bool) -> sa.Table:
    """The table as it stands before each ``batch_alter_table``, so SQLite's rebuild keeps what reflection misses.

    ``named`` is the upgraded shape. Before the upgrade's batch, ``name`` is
    already there, added nullable so it could be backfilled.
    """
    columns = [
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("encrypted_api_key", sa.String(), nullable=named),
        sa.Column("api_key_last4", sa.String(length=4), nullable=True),
        sa.Column("api_base", sa.String(length=1024), nullable=True),
        sa.Column("encrypted_client_args", sa.String(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("restricted_to_organization_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["restricted_to_organization_id"], ["organization.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.Index(_RESTRICTION_INDEX, "restricted_to_organization_id"),
    ]
    columns.insert(1, sa.Column("name", sa.String(length=64), nullable=not named))
    if named:
        columns.append(sa.UniqueConstraint("name", name=_UNIQUE_NAME))
    else:
        columns.append(sa.UniqueConstraint("provider", name=_UNIQUE_PROVIDER))
    return sa.Table(_TABLE, sa.MetaData(), *columns)


def upgrade() -> None:
    op.add_column(_TABLE, sa.Column("name", sa.String(length=64), nullable=True))
    op.execute(sa.text(f"UPDATE {_TABLE} SET name = provider"))
    with op.batch_alter_table(_TABLE, copy_from=_hosted_providers(named=False)) as batch:
        batch.alter_column("name", existing_type=sa.String(length=64), nullable=False)
        batch.alter_column("encrypted_api_key", existing_type=sa.String(), nullable=True)
        batch.drop_constraint(_UNIQUE_PROVIDER, type_="unique")
        batch.create_unique_constraint(_UNIQUE_NAME, ["name"])


def downgrade() -> None:
    bind = op.get_bind()
    renamed = bind.execute(sa.text(f"SELECT name FROM {_TABLE} WHERE name <> provider")).first()  # noqa: S608
    keyless = bind.execute(sa.text(f"SELECT name FROM {_TABLE} WHERE encrypted_api_key IS NULL")).first()  # noqa: S608
    if renamed is not None or keyless is not None:
        msg = (
            "Cannot downgrade: remove the hosted providers named otherwise than their implementation, "
            "and those that hold no key, first."
        )
        raise RuntimeError(msg)
    with op.batch_alter_table(_TABLE, copy_from=_hosted_providers(named=True)) as batch:
        batch.drop_constraint(_UNIQUE_NAME, type_="unique")
        batch.create_unique_constraint(_UNIQUE_PROVIDER, ["provider"])
        batch.alter_column("encrypted_api_key", existing_type=sa.String(), nullable=False)
        batch.drop_column("name")
