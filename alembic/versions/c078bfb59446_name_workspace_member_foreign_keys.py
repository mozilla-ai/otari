"""Name the foreign keys on workspace_member.

``WorkspaceMemberRepository.add`` tells a vanished workspace apart from a
duplicate membership by matching the constraint a refused insert reports. It
matched PostgreSQL's generated name for the workspace foreign key, which the
schema never declared, so a later migration that rebuilt or renamed the table
would change that name and turn the advertised 404 into an unhandled 500. Name
both foreign keys, in the model too, so a rebuild has to carry the name or fail
visibly. A rename is metadata only: it keeps each key's definition and does not
revalidate rows or lock the referenced tables.

SQLite is skipped and diverges on purpose: ``create_all`` there names the keys
while a migrated schema keeps them unnamed, which is harmless because SQLite
reports no constraint name on an integrity error.

Revision ID: c078bfb59446
Revises: 5bdadb1401a1
Create Date: 2026-10-23 13:52:13.000000
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c078bfb59446"
down_revision: str | Sequence[str] | None = "5bdadb1401a1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# PostgreSQL's generated name for each unnamed key, and the name it now carries.
_RENAMES = (
    ("workspace_member_workspace_id_fkey", "fk_workspace_member_workspace_id"),
    ("workspace_member_user_id_fkey", "fk_workspace_member_user_id"),
)


def _rename(old: str, new: str) -> None:
    op.execute(f'ALTER TABLE workspace_member RENAME CONSTRAINT "{old}" TO "{new}"')


def upgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        return
    for generated, declared in _RENAMES:
        _rename(generated, declared)


def downgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        return
    for generated, declared in _RENAMES:
        _rename(declared, generated)
