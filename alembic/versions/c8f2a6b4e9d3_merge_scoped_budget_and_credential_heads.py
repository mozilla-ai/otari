"""Rejoin the two migration heads into one.

Two revisions were authored against ``a3c7e1b9d5f2`` in parallel and both
landed: ``b6e8c2a4d7f1`` (scoped-budget reset alignment) and ``f2a4c6d8b0e3``
(the nullable credential columns on ``user``). Neither is wrong and they touch
different tables, but leaving both as heads makes ``alembic upgrade head``
refuse to run at all ("Multiple head revisions are present"), so no deployment
can upgrade past either one.

This revision has no schema of its own. It exists only to name both parents, so
``head`` resolves to a single revision again and the two branches apply in
either order.

One consequence is worth knowing rather than rediscovering: ``alembic downgrade
-1`` from here reports "Ambiguous walk", because a relative step cannot say
which of the two branches to walk down. That is how Alembic treats every merge
point, not a defect of this revision. Downgrade to a named revision instead
(``alembic downgrade b6e8c2a4d7f1``), which is unambiguous and reversible.

Revision ID: c8f2a6b4e9d3
Revises: b6e8c2a4d7f1, f2a4c6d8b0e3
Create Date: 2026-08-20
"""

from collections.abc import Sequence

revision: str = "c8f2a6b4e9d3"
down_revision: str | Sequence[str] | None = ("b6e8c2a4d7f1", "f2a4c6d8b0e3")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Nothing to apply: this revision only rejoins the two branches."""


def downgrade() -> None:
    """Nothing to undo: this revision only rejoins the two branches."""
