"""Rejoin the two migration heads into one.

``6664b766de86`` (``api_keys.max_end_users``) and ``d8b2f6a4c0e9`` (the price
rows' ``kind``) were each chained onto the head their pull request saw, and both
landed, so ``alembic upgrade head`` refuses to run. They touch different tables.

This revision has no schema of its own: it names both parents so ``head``
resolves to one revision again. As at every merge point, ``alembic downgrade
-1`` from here is ambiguous; downgrade to a named revision instead.

Revision ID: 6c3606e98ed9
Revises: 6664b766de86, d8b2f6a4c0e9
Create Date: 2026-10-20
"""

from collections.abc import Sequence

revision: str = "6c3606e98ed9"
down_revision: str | Sequence[str] | None = ("6664b766de86", "d8b2f6a4c0e9")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
