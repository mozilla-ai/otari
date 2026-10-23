"""Name the foreign keys on workspace_member.

``WorkspaceMemberRepository.add`` tells a vanished workspace apart from a
duplicate membership by matching the constraint a refused insert reports. It
matched PostgreSQL's generated name for the workspace foreign key, which the
schema never declared, so a later migration that rebuilt or renamed the table
would change that name and turn the advertised 404 into an unhandled 500. Name
both foreign keys so the schema stays the single source of the name, and a
future rebuild has to carry it or fail visibly.

SQLite is skipped because it carries no constraint name on an integrity error
at all, so naming the keys would change nothing there. The model declares the
same names, so a ``create_all`` schema and a migrated one agree.

Revision ID: c078bfb59446
Revises: c8e0f2a4b6d3
Create Date: 2026-10-23 13:52:13.000000
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c078bfb59446"
down_revision: str | Sequence[str] | None = "c8e0f2a4b6d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_WORKSPACE_ID_FK = "fk_workspace_member_workspace_id"
_USER_ID_FK = "fk_workspace_member_user_id"
# The names PostgreSQL generated for the two unnamed foreign keys the tenancy
# migration created.
_GENERATED_WORKSPACE_ID_FK = "workspace_member_workspace_id_fkey"
_GENERATED_USER_ID_FK = "workspace_member_user_id_fkey"


def upgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        return
    for generated, declared, table, column in (
        (_GENERATED_WORKSPACE_ID_FK, _WORKSPACE_ID_FK, "workspace", "workspace_id"),
        (_GENERATED_USER_ID_FK, _USER_ID_FK, "user", "user_id"),
    ):
        op.drop_constraint(generated, "workspace_member", type_="foreignkey")
        op.create_foreign_key(declared, "workspace_member", table, [column], ["id"], ondelete="CASCADE")


def downgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        return
    for declared, generated, table, column in (
        (_WORKSPACE_ID_FK, _GENERATED_WORKSPACE_ID_FK, "workspace", "workspace_id"),
        (_USER_ID_FK, _GENERATED_USER_ID_FK, "user", "user_id"),
    ):
        op.drop_constraint(declared, "workspace_member", type_="foreignkey")
        op.create_foreign_key(generated, "workspace_member", table, [column], ["id"], ondelete="CASCADE")
