"""Let an organization mandate name a hosted guardrail.

``hosted_guardrail_id`` points at a guardrail the deployment hosts behind
``HostedGuardrailPort``. It has no foreign key, because the table holding
hosted guardrails belongs to whichever build binds the port, not to this chain;
the port answers whether an id is offered.

A hosted mandate is its own backend. It names no URL, no definition and no
credential, because the hosted guardrail brings its own. The new check says so,
and leaves ``ck_organization_guardrails_single_backend`` as it was.

The check needs a table rebuild on SQLite, so it goes through
``batch_alter_table`` with a complete ``copy_from``, as ``a9c4e7b2d5f8`` does.

Revision ID: c2e5a8d1f4b7
Revises: a9c4e7b2d5f8
Create Date: 2026-09-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c2e5a8d1f4b7"
down_revision: str | Sequence[str] | None = "a9c4e7b2d5f8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_DEFINITIONS = "organization_guardrail_definitions"
_MANDATES = "organization_guardrails"

_MANDATES_ORG_INDEX = "ix_organization_guardrails_organization_id"
_MANDATES_UQ_PROFILE = "uq_organization_guardrails_org_profile"
_MANDATES_FK_DEFINITION = "fk_organization_guardrails_definition"
_MANDATES_CK_BACKEND = "ck_organization_guardrails_single_backend"
_MANDATES_CK_HOSTED = "ck_organization_guardrails_hosted_alone"
_MANDATES_HOSTED_INDEX = "ix_organization_guardrails_hosted_guardrail_id"

_BACKEND_CHECK = "NOT (url IS NOT NULL AND definition_id IS NOT NULL)"
_HOSTED_CHECK = (
    "hosted_guardrail_id IS NULL OR (url IS NULL AND definition_id IS NULL AND encrypted_credential IS NULL)"
)


def _mandates(*, hosted: bool) -> sa.Table:
    """``organization_guardrails`` as it stands on one side of this revision, complete for ``copy_from``."""
    meta = sa.MetaData()
    columns: list[sa.Column[object] | sa.schema.SchemaItem] = [
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("profile", sa.String(), nullable=False),
        sa.Column("url", sa.String(), nullable=True),
        sa.Column("encrypted_credential", sa.Text(), nullable=True),
        sa.Column("mode", sa.String(), nullable=False),
        sa.Column("on_unavailable", sa.String(), nullable=False),
        sa.Column("validate_kwargs", sa.JSON(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("applies_to_all_workspaces", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("definition_id", sa.Uuid(), nullable=True),
    ]
    constraints: list[sa.schema.SchemaItem] = [
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "profile", name=_MANDATES_UQ_PROFILE),
        sa.ForeignKeyConstraint(["organization_id"], ["organization.id"], ondelete="CASCADE"),
        sa.Index(_MANDATES_ORG_INDEX, "organization_id"),
        sa.ForeignKeyConstraint(
            ["organization_id", "definition_id"],
            [f"{_DEFINITIONS}.organization_id", f"{_DEFINITIONS}.id"],
            name=_MANDATES_FK_DEFINITION,
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint(_BACKEND_CHECK, name=_MANDATES_CK_BACKEND),
    ]
    if hosted:
        columns.append(sa.Column("hosted_guardrail_id", sa.Uuid(), nullable=True))
        constraints += [
            sa.Index(_MANDATES_HOSTED_INDEX, "hosted_guardrail_id"),
            sa.CheckConstraint(_HOSTED_CHECK, name=_MANDATES_CK_HOSTED),
        ]
    return sa.Table(_MANDATES, meta, *columns, *constraints)


def upgrade() -> None:
    with op.batch_alter_table(_MANDATES, copy_from=_mandates(hosted=False)) as batch:
        batch.add_column(sa.Column("hosted_guardrail_id", sa.Uuid(), nullable=True))
        batch.create_index(_MANDATES_HOSTED_INDEX, ["hosted_guardrail_id"])
        batch.create_check_constraint(_MANDATES_CK_HOSTED, _HOSTED_CHECK)


def downgrade() -> None:
    # Without its column a hosted mandate would read as one falling back to the
    # deployment's ``guardrails_url``, a check against a service that never
    # heard of the profile. Dropping the row is the honest downgrade.
    mandates = sa.table(_MANDATES, sa.column("hosted_guardrail_id", sa.Uuid()))
    op.execute(mandates.delete().where(mandates.c.hosted_guardrail_id.is_not(None)))
    with op.batch_alter_table(_MANDATES, copy_from=_mandates(hosted=True)) as batch:
        batch.drop_constraint(_MANDATES_CK_HOSTED, type_="check")
        batch.drop_index(_MANDATES_HOSTED_INDEX)
        batch.drop_column("hosted_guardrail_id")
