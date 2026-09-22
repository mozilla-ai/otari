"""Add the organization guardrail definitions table and the mandate's link to it.

An ``organization_guardrails`` row has always been a *mandate*: a profile name
some other service already serves, plus the failure modes and the workspace
scope. It could never define the check itself, so the catalog listed guardrails
Otari could call over a hosted API with nowhere to put their vendor API key.

``organization_guardrail_definitions`` is that missing half: the
``any_guardrail`` class to build and the arguments to build it with, secrets in
one encrypted map. Its own table rather than three more columns on the mandate,
because the build arguments depend on ``guardrail_name`` while a mandate is
identified by ``(organization_id, profile)``; fused, one guardrail under two
policies would hold two copies of one vendor key.

Two constraints are worth naming. ``uq_org_guardrail_definitions_org_id`` exists
only so the mandate's foreign key can be *composite*, which is what stops one
organization mandating another's definition (and its credential) however a write
path behaves. And ``ck_organization_guardrails_single_backend`` is deliberately
not an XOR: both columns NULL is the ordinary pre-existing row, the one falling
back to the deployment's ``guardrails_url``, so an XOR would refuse every
mandate already stored.

Adding a check constraint and a foreign key to an existing table needs a table
rebuild on SQLite, which has no ``ADD CONSTRAINT``, so the second half goes
through ``batch_alter_table`` with an explicit ``copy_from``. The declaration
has to be complete: ``copy_from`` replaces reflection wholesale, so a constraint
or index left out of it is one the rebuild drops on the floor.

Nothing reads the new rows yet. This revision is storage only.

Revision ID: a9c4e7b2d5f8
Revises: b8d4f0a2c6e1
Create Date: 2026-09-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a9c4e7b2d5f8"
down_revision: str | Sequence[str] | None = "b8d4f0a2c6e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_DEFINITIONS = "organization_guardrail_definitions"
_MANDATES = "organization_guardrails"

_DEFINITIONS_ORG_INDEX = "ix_organization_guardrail_definitions_organization_id"
_DEFINITIONS_UQ_NAME = "uq_org_guardrail_definitions_org_name"
_DEFINITIONS_UQ_ORG_ID = "uq_org_guardrail_definitions_org_id"

_MANDATES_ORG_INDEX = "ix_organization_guardrails_organization_id"
_MANDATES_UQ_PROFILE = "uq_organization_guardrails_org_profile"
_MANDATES_FK_DEFINITION = "fk_organization_guardrails_definition"
_MANDATES_CK_BACKEND = "ck_organization_guardrails_single_backend"


def _mandates(*, linked: bool) -> sa.Table:
    """``organization_guardrails`` as it stands on one side of this revision.

    Handed to ``batch_alter_table`` as ``copy_from``, so it must describe the
    table completely: the unique constraint, the organization foreign key and
    the organization index are all things SQLite's rebuild would otherwise lose.
    """
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
    ]
    constraints: list[sa.schema.SchemaItem] = [
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "profile", name=_MANDATES_UQ_PROFILE),
        sa.ForeignKeyConstraint(["organization_id"], ["organization.id"], ondelete="CASCADE"),
        sa.Index(_MANDATES_ORG_INDEX, "organization_id"),
    ]
    if linked:
        columns.append(sa.Column("definition_id", sa.Uuid(), nullable=True))
        constraints += [
            sa.ForeignKeyConstraint(
                ["organization_id", "definition_id"],
                [f"{_DEFINITIONS}.organization_id", f"{_DEFINITIONS}.id"],
                name=_MANDATES_FK_DEFINITION,
                ondelete="RESTRICT",
            ),
            sa.CheckConstraint(
                "NOT (url IS NOT NULL AND definition_id IS NOT NULL)",
                name=_MANDATES_CK_BACKEND,
            ),
        ]
    return sa.Table(_MANDATES, meta, *columns, *constraints)


def upgrade() -> None:
    op.create_table(
        _DEFINITIONS,
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("guardrail_name", sa.String(), nullable=False),
        sa.Column("create_kwargs", sa.JSON(), nullable=False),
        sa.Column("encrypted_create_secrets", sa.Text(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["organization_id"], ["organization.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("organization_id", "name", name=_DEFINITIONS_UQ_NAME),
        sa.UniqueConstraint("organization_id", "id", name=_DEFINITIONS_UQ_ORG_ID),
    )
    op.create_index(_DEFINITIONS_ORG_INDEX, _DEFINITIONS, ["organization_id"])

    with op.batch_alter_table(_MANDATES, copy_from=_mandates(linked=False)) as batch:
        batch.add_column(sa.Column("definition_id", sa.Uuid(), nullable=True))
        batch.create_foreign_key(
            _MANDATES_FK_DEFINITION,
            _DEFINITIONS,
            ["organization_id", "definition_id"],
            ["organization_id", "id"],
            ondelete="RESTRICT",
        )
        batch.create_check_constraint(_MANDATES_CK_BACKEND, "NOT (url IS NOT NULL AND definition_id IS NOT NULL)")


def downgrade() -> None:
    with op.batch_alter_table(_MANDATES, copy_from=_mandates(linked=True)) as batch:
        batch.drop_constraint(_MANDATES_CK_BACKEND, type_="check")
        batch.drop_constraint(_MANDATES_FK_DEFINITION, type_="foreignkey")
        batch.drop_column("definition_id")

    op.drop_index(_DEFINITIONS_ORG_INDEX, table_name=_DEFINITIONS)
    op.drop_table(_DEFINITIONS)
