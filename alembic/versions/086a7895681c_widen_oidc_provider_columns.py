"""Widen the three provider columns so a customer's own OIDC issuer URL fits.

``oauth_identity_link.provider``, ``user.oauth_provider`` and
``oauth_pending_state.provider`` each held a short fixed vocabulary
(``google``, ``github``) at their old widths. A customer's own OIDC
connection is named by its issuer URL instead, which does not fit in 50 or
32 characters, so all three widen to 255 alongside
``gateway.models.identity.MAX_PROVIDER_LENGTH``.

``batch_alter_table`` is used on all three tables because SQLite cannot
``ALTER COLUMN`` in place; it recreates the table on SQLite and renders a
plain ``ALTER COLUMN`` on PostgreSQL, so one form works on both engines.

Revision ID: 086a7895681c
Revises: 012b723efd0d
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "086a7895681c"
down_revision: str | Sequence[str] | None = "012b723efd0d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # NOTE: SQLite's batch reflection preserves this table's named unique
    # constraint and composite primary key, so copy_from is not needed here.
    with op.batch_alter_table("oauth_identity_link") as batch_op:
        batch_op.alter_column(
            "provider", existing_type=sa.String(length=50), type_=sa.String(length=255), existing_nullable=False
        )
    with op.batch_alter_table("user") as batch_op:
        batch_op.alter_column(
            "oauth_provider", existing_type=sa.String(length=50), type_=sa.String(length=255), existing_nullable=True
        )
    with op.batch_alter_table("oauth_pending_state") as batch_op:
        batch_op.alter_column(
            "provider", existing_type=sa.String(length=32), type_=sa.String(length=255), existing_nullable=False
        )


# Each column's old width, which a value stored since the upgrade can exceed.
_OLD_WIDTHS = (
    ("oauth_identity_link", "provider", 50),
    ("oauth_pending_state", "provider", 32),
    ("user", "oauth_provider", 50),
)


def _refuse_names_that_no_longer_fit() -> None:
    """Refuse the downgrade while any stored provider is too long for its old column.

    An organization's own connection is named by its issuer URL, which does not fit the old widths,
    so narrowing the column would either fail with the engine's own error or lose the name that
    identifies an account. Naming the rows is the only honest answer: nothing here can invent a
    shorter name for them.
    """
    connection = op.get_bind()
    for table, column, width in _OLD_WIDTHS:
        offending = connection.execute(
            # The identifiers are quoted because one of these tables is named ``user``, which
            # PostgreSQL reserves.
            sa.text(f'SELECT count(*) FROM "{table}" WHERE length("{column}") > :width'),  # noqa: S608 (literals above)
            {"width": width},
        ).scalar_one()
        if offending:
            raise RuntimeError(
                f"{offending} row(s) in {table}.{column} name a provider longer than {width} "
                f"characters, which the old column cannot hold. Remove those sign-in records, then "
                f"migrate again."
            )


def downgrade() -> None:
    _refuse_names_that_no_longer_fit()
    with op.batch_alter_table("oauth_pending_state") as batch_op:
        batch_op.alter_column(
            "provider", existing_type=sa.String(length=255), type_=sa.String(length=32), existing_nullable=False
        )
    with op.batch_alter_table("user") as batch_op:
        batch_op.alter_column(
            "oauth_provider", existing_type=sa.String(length=255), type_=sa.String(length=50), existing_nullable=True
        )
    with op.batch_alter_table("oauth_identity_link") as batch_op:
        batch_op.alter_column(
            "provider", existing_type=sa.String(length=255), type_=sa.String(length=50), existing_nullable=False
        )
