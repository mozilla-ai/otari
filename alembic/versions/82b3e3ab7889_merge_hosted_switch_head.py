"""Rejoin the two migration heads into one.

``6c3606e98ed9`` rejoined the heads it saw, and ``ae8d42792fd8`` (how hosted
models are switched) landed beside it onto ``c0e83c6fe801``, so ``alembic
upgrade head`` refuses to run again. They touch different tables.

This revision has no schema of its own: it names both parents so ``head``
resolves to one revision again. As at every merge point, ``alembic downgrade
-1`` from here is ambiguous; downgrade to a named revision instead.

Revision ID: 82b3e3ab7889
Revises: 6c3606e98ed9, ae8d42792fd8
Create Date: 2026-10-20
"""

from collections.abc import Sequence

revision: str = "82b3e3ab7889"
down_revision: str | Sequence[str] | None = ("6c3606e98ed9", "ae8d42792fd8")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
