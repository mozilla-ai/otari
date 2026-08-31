"""Give a budget an owning organization, so an admin can define one.

The roles matrix (otari-ai#1943) puts Spend & budgets at Edit for an
organization admin, and until now a budget was deployment-global with
``/v1/budgets`` gated on ``require_deployment_operator``: there was no column
saying whose a budget was, so there was nothing for a tenant-scoped surface to
scope to. This adds it, and ``/v1/organizations/me/budgets`` writes it.

**Nullable, with no backfill, and that is the contract rather than a shortcut.**
NULL means the deployment's own budget. Two populations read NULL and both
should:

- every budget that already exists, which was defined deployment-wide;
- every budget the otari-ai cutover migration mints. That migration shares one
  budget per distinct ``(max_budget, budget_duration_sec, reset_alignment)``
  shape across all the ceilings it writes (see ``docs/budget-migration.md``
  there), so a shape held by two tenants' ceilings has no single owner and
  inventing one would hand one tenant's admin control over the other's cap.

Nothing tenant-facing ever lists, offers or repoints a NULL row, so from an
organization's side a deployment budget does not exist. That is also what keeps
the cutover migration working untouched: its preflight refuses on a *missing*
table or column and its inserts name their columns explicitly, so a new nullable
column is invisible to it and needs no re-transcription and no backfill pass.

**Column, index, then the constraint in a batch**, following
``7c5ba82a601b``: SQLite has no ALTER TABLE ADD CONSTRAINT, so the foreign key
goes on inside ``batch_alter_table`` while the plain ADD COLUMN and the index do
not need it.

**The one backfill, and why it is narrow.** A deployment with exactly one
organization gets its budgets assigned to it, when nothing outside that
organization names them. There the operator *is* that organization's owner
already, so the assignment grants no authority anybody lacked, and without it the
new Spend page would open read-only on every existing self-hosted deployment.

A deployment with two or more organizations is left entirely alone. Assigning
there would hand one tenant's admins control over a cap set above them, and for a
budget shared by two tenants' ceilings there is no single right answer anyway.
That population is the hosted one, and it is the cutover's to resolve, where the
planner knows each ceiling's organization; see ``docs/budget-migration.md`` in
otari-ai. Until then such a ceiling reads back with ``manageable`` false, which
is the honest description of a real ceiling whose figure is set elsewhere.

A budget named by a ``users`` row is skipped even on a single-organization
deployment: that table is the deployment's own, with no tenancy column, so a
budget handed to a gateway user is not the organization's to redefine. A budget
named by a ``budget_reset_logs`` row is skipped for the same reason one step
removed: a reset record outlives the assignment that produced it, so such a
budget *was* handed to a gateway user even where no live ``users`` row still
points at it, and the reference goes on refusing a delete either way.

``ondelete="CASCADE"`` matches ``workspace.organization_id``, the neighbouring
column in the tenancy model. Nothing can reach it yet: this gateway serves no
organization-delete route. Whoever adds one has to clear the tenant's ceilings
first, the way ``WorkspaceService._delete_scoped_budgets_for`` already does for a
workspace, because ``scoped_budgets.budget_id`` is RESTRICT and a cascade
arriving here would otherwise fail on it rather than on anything that names the
organization.

Revision ID: b7e1c4a9d2f5
Revises: a4d7f1c9e2b6
Create Date: 2026-08-31
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b7e1c4a9d2f5"
down_revision: str | Sequence[str] | None = "a4d7f1c9e2b6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("budgets", sa.Column("organization_id", sa.Uuid(), nullable=True))
    op.create_index(op.f("ix_budgets_organization_id"), "budgets", ["organization_id"], unique=False)
    with op.batch_alter_table("budgets") as batch:
        batch.create_foreign_key(
            "fk_budgets_organization_id",
            "organization",
            ["organization_id"],
            ["id"],
            ondelete="CASCADE",
        )
    _assign_sole_organization(op.get_bind())


def _assign_sole_organization(bind: sa.Connection) -> None:
    """Give every eligible budget to the deployment's organization, if it has just one.

    Reflected tables rather than the ORM models: a migration has to keep working
    when those models move on, so it names the columns it needs and nothing else.
    """
    organizations = list(bind.execute(sa.text("SELECT id FROM organization")).scalars())
    if len(organizations) != 1:
        # Two or more tenants, so there is nothing unambiguous to assign. Zero is
        # the same branch and no chain reaches it (an earlier data migration
        # seeds "Default organization"), but it is one condition rather than two
        # so that a database whose tenancy seed was removed does not crash the
        # upgrade here.
        return
    bind.execute(
        sa.text(
            "UPDATE budgets SET organization_id = :organization_id "
            "WHERE organization_id IS NULL "
            "AND budget_id NOT IN (SELECT budget_id FROM users WHERE budget_id IS NOT NULL) "
            "AND budget_id NOT IN (SELECT budget_id FROM budget_reset_logs)"
        ),
        {"organization_id": organizations[0]},
    )


def downgrade() -> None:
    with op.batch_alter_table("budgets") as batch:
        batch.drop_constraint("fk_budgets_organization_id", type_="foreignkey")
    op.drop_index(op.f("ix_budgets_organization_id"), table_name="budgets")
    op.drop_column("budgets", "organization_id")
