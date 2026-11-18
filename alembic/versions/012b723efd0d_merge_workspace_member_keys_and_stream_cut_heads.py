"""Rejoin the two migration heads into one.

Two revisions were authored against ``5bdadb1401a1`` in parallel and both
landed: ``c078bfb59446`` (the named ``workspace_member`` foreign keys) and
``c7d2e9a4b6f1`` (``usage_logs.stream_cut_by``). They touch different tables,
but leaving both as heads makes ``alembic upgrade head`` refuse to run at all
("Multiple head revisions are present").

This revision has no schema of its own. It only names both parents, so ``head``
resolves to a single revision again and the two branches apply in either order.
As at every merge point, ``alembic downgrade -1`` from here is ambiguous;
downgrade to a named revision instead.

Revision ID: 012b723efd0d
Revises: c078bfb59446, c7d2e9a4b6f1
Create Date: 2026-11-09
"""

from collections.abc import Sequence

revision: str = "012b723efd0d"
down_revision: str | Sequence[str] | None = ("c078bfb59446", "c7d2e9a4b6f1")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Nothing to apply: this revision only rejoins the two branches."""


def downgrade() -> None:
    """Nothing to undo: this revision only rejoins the two branches."""
